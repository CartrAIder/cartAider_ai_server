"""Sweep occlusion (item count) x camera quality and report detection recall,
recognition top-1, and end-to-end accuracy on detected crops."""
import numpy as np, cv2
from collections import defaultdict
from pathlib import Path
from ultralytics import YOLO
from cartgate import config
from cartgate.embed import get_embedder
from cartgate.gallery import build_gallery
from cartgate.synth import synth_cart_frames
from cartgate.match import sku_similarity


def load_cutouts(d="out/cut_rembg"):
    cut = defaultdict(list)
    for p in sorted(Path(d).glob("*.png")):
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is not None and im.ndim == 3 and im.shape[2] == 4:
            cut[p.stem.split("__")[0]].append(im)
    return dict(cut)


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1-ix0)*max(0, iy1-iy0)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0


def degrade(img, mode, rng):
    if mode == "clean":
        return img
    h, w = img.shape[:2]
    if "lowres" in mode:                      # cheap / far camera
        s = rng.uniform(0.30, 0.5)
        img = cv2.resize(cv2.resize(img, (max(1, int(w*s)), max(1, int(h*s)))), (w, h))
    if "blur" in mode:
        img = cv2.GaussianBlur(img, (int(rng.choice([5, 7, 9])),)*2, 0)
    if "noise" in mode:
        img = np.clip(img.astype(np.float32) + rng.normal(0, 18, img.shape), 0, 255).astype(np.uint8)
    if "dark" in mode:
        img = (img.astype(np.float32) * rng.uniform(0.4, 0.6)).astype(np.uint8)
    if "jpeg" in mode:
        _, e = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(18, 38))])
        img = cv2.imdecode(e, 1)
    return img


def run_cfg(model, gallery, embedder, cutouts, n_items, qual, dev, n_carts=40, seed=0):
    rng = np.random.default_rng(seed)
    skus = list(cutouts.keys())
    vis = found = recog = buried = 0
    for _ in range(n_carts):
        cart = list(rng.choice(skus, size=n_items, replace=True))
        for f in synth_cart_frames(cart, cutouts, rng, n_frames=1, size=(640, 640)):
            buried += len(cart) - len(f.objects)             # occlusion-aware GT dropped these
            img = degrade(f.image, qual, rng)
            res = model.predict(img, conf=config.DET_CONF, verbose=False, device=dev)[0]
            dets = [tuple(int(v) for v in b) for b in res.boxes.xyxy.cpu().numpy()]
            for o in f.objects:
                vis += 1
                best, bx = 0.4, None
                for d in dets:
                    j = iou(d, o.box)
                    if j > best:
                        best, bx = j, d
                if bx is None:
                    continue
                found += 1
                crop = img[max(0, bx[1]):bx[3], max(0, bx[0]):bx[2]]
                if crop.size == 0:
                    continue
                vec = embedder.embed(crop, None)
                dist = [s for s in skus if s != o.sku]
                receipt = [o.sku] + list(rng.choice(dist, size=5, replace=False))
                pred = max(receipt, key=lambda s: sku_similarity(vec, gallery, s))
                recog += int(pred == o.sku)
    return dict(vis=vis, buried=buried, det_recall=found/max(vis, 1),
               recog_top1=recog/max(found, 1), e2e=recog/max(vis, 1))


def main():
    dev = 0
    emb = get_embedder("dino_arc.onnx", pad=True)          # DINOv2+ArcFace
    model = YOLO("runs/detector/best.pt")
    gallery = build_gallery("dataset/images", emb, "out", remove_bg=False, enrich_synth=16)
    cutouts = load_cutouts()
    print(f"\n{'items':>5} {'quality':<16} {'buried%':>7} {'det_recall':>10} {'recog_top1':>10} {'end2end':>8}")
    print("-" * 62)
    for n_items in (4, 8, 14):
        for qual in ("clean", "lowres+noise", "bad(lowres+blur+noise+dark+jpeg)"):
            r = run_cfg(model, gallery, emb, cutouts, n_items, qual, dev, seed=n_items)
            bur = r["buried"] / max(r["buried"] + r["vis"], 1)
            print(f"{n_items:>5} {qual:<16.16} {bur:>6.0%} {r['det_recall']:>10.1%} "
                  f"{r['recog_top1']:>10.1%} {r['e2e']:>8.1%}")


if __name__ == "__main__":
    main()
