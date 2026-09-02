"""Render detection + recognition on hard (occluded/degraded) cart scenes, annotating
each box with predicted name, confidence, and correctness. Saves PNGs to out/viz/.
Colours: green=correct & confident, amber=correct & low-conf, red=wrong, grey=missed.
"""
import csv, glob
from collections import defaultdict
from pathlib import Path

import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from cartgate.config import PASS_SIM, REVIEW_SIM      # single source of truth
from cartgate.embed import get_embedder
from cartgate.gallery import build_gallery
from cartgate.synth import synth_cart_frames
from cartgate.match import sku_similarity

FONT_PATH = sorted(glob.glob("/usr/share/fonts/truetype/nanum/NanumGothic*.ttf"))[0]
COL = {"ok": (34, 197, 94), "review": (234, 179, 8), "wrong": (239, 68, 68), "miss": (148, 163, 184)}


def names_map():
    m = {}
    for r in csv.DictReader(open("products.csv", encoding="utf-8-sig")):
        m[r["sku_id"]] = r["name"]
    return m


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
    h, w = img.shape[:2]
    if "lowres" in mode:
        s = 0.32; img = cv2.resize(cv2.resize(img, (int(w*s), int(h*s))), (w, h))
    if "blur" in mode:
        img = cv2.GaussianBlur(img, (11, 1), 0)                 # horizontal motion
    if "noise" in mode:
        img = np.clip(img.astype(np.float32) + rng.normal(0, 16, img.shape), 0, 255).astype(np.uint8)
    if "dark" in mode:
        img = (img.astype(np.float32) * 0.42).astype(np.uint8)
    if "jpeg" in mode:
        _, e = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 22]); img = cv2.imdecode(e, 1)
    return img


def draw(img_bgr, dets, misses, header):
    im = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT_PATH, 19); fh = ImageFont.truetype(FONT_PATH, 22)
    for (box, col, label) in misses + dets:                    # misses first (under)
        x0, y0, x1, y1 = box
        d.rectangle([x0, y0, x1, y1], outline=col, width=3)
        tb = d.textbbox((0, 0), label, font=f)
        tw, th = tb[2]-tb[0], tb[3]-tb[1]
        ly = max(0, y0-th-8)
        d.rectangle([x0, ly, x0+tw+10, ly+th+8], fill=col)
        d.text((x0+5, ly+2), label, fill=(15, 23, 32), font=f)
    d.rectangle([0, 0, im.width, 34], fill=(15, 23, 32))
    d.text((10, 6), header, fill=(240, 244, 248), font=fh)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def main():
    nm = names_map()
    emb = get_embedder("dino_arc.onnx", pad=True)          # DINOv2+ArcFace, letterbox preprocessing
    model = YOLO("runs/detector/best.pt")
    gallery = build_gallery("dataset/images", emb, "out", remove_bg=False, enrich_synth=16)
    cutouts = load_cutouts()
    outdir = Path("out/viz"); outdir.mkdir(parents=True, exist_ok=True)

    scenes = [
        ("여유 · 깨끗한 카메라", 4, "clean"),
        ("혼잡 · 물건 겹침", 8, "clean"),
        ("쌓임 · 심한 겹침", 13, "clean"),
        ("저해상도 카메라", 7, "lowres+noise"),
        ("움직임 블러 (지나가는 카트)", 7, "blur+noise"),
        ("저조도", 7, "dark+noise"),
        ("최악 복합 (저해상도+블러+노이즈+저조도+jpeg)", 10, "lowres+blur+noise+dark+jpeg"),
    ]
    for i, (title, n, qual) in enumerate(scenes):
        rng = np.random.default_rng(100 + i)
        skus = list(cutouts.keys())
        cart = list(rng.choice(skus, size=n, replace=False))
        frame = synth_cart_frames(cart, cutouts, rng, n_frames=1, size=(720, 720))[0]
        img = degrade(frame.image, qual, rng)
        receipt = list(set(cart))
        res = model.predict(img, conf=0.25, verbose=False, device=0)[0]
        boxes = [tuple(int(v) for v in b) for b in res.boxes.xyxy.cpu().numpy()]

        # keep the single best-IoU detection per GT object (drop duplicate boxes)
        bygt = {}
        for bx in boxes:
            best, gt = 0.4, None
            for o in frame.objects:
                j = iou(bx, o.box)
                if j > best: best, gt = j, o
            if gt is not None and (gt.track_id not in bygt or best > bygt[gt.track_id][0]):
                bygt[gt.track_id] = (best, bx, gt)

        dets, recog = [], 0
        for _, bx, gt in bygt.values():
            crop = img[max(0, bx[1]):bx[3], max(0, bx[0]):bx[2]]
            if crop.size == 0:
                continue
            vec = emb.embed(crop, None)
            sim, pred = max(((sku_similarity(vec, gallery, s), s) for s in receipt))
            correct = pred == gt.sku
            band = "ok" if sim >= PASS_SIM else ("review" if sim >= REVIEW_SIM else "wrong")
            col = COL["wrong"] if not correct else (COL["ok"] if band == "ok" else COL["review"])
            recog += int(correct)
            dets.append((bx, col, f"{nm.get(pred, pred)} {sim*100:.0f}% {'O' if correct else 'X'}"))
        matched_gt = set(bygt)
        misses = [(o.box, COL["miss"], "미검출(가림)") for o in frame.objects
                  if o.track_id not in matched_gt]

        vis = len(frame.objects)
        header = f"{title}  —  검출 {len(matched_gt)}/{vis} · 인식 {recog}/{len(matched_gt)}"
        out = draw(img, dets, misses, header)
        cv2.imwrite(str(outdir / f"scene_{i}.png"), out)
        print(f"  scene_{i}: {title} | items={vis} detected={len(matched_gt)} recognized={recog}")
    print(f"\nsaved to {outdir}")


if __name__ == "__main__":
    main()
