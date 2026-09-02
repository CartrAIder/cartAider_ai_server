"""Generate the synthetic cart dataset — one build serving two purposes:

  1) decision-layer benchmark:  carts.json (receipt vs contents, per-camera GT, verdict)
  2) detector fine-tuning:       YOLO labels + data.yaml (single "product" class)

Two upper-diagonal cameras (opposite sides) film the SAME pile for several
consecutive frames, so the set can also exercise tracking and cross-camera
fusion — a one-frame-per-camera set cannot (nothing reaches min_frames).
Scenarios mix normal / unpaid / quantity-excess so verdicts are known.

  python scripts/make_cart_dataset.py --num 500 --frames 4   # 500 x 2 x 4 = 4000 imgs
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from cartgate import calibrate_plane
from cartgate.synth import CART_PLANE_CM, camera_matrix, synth_cart_views

FORMAT_VERSION = 2          # 1 = one image per camera, 2 = frames[] per camera


def write_calibration(path: Path, dirs, names, size: int) -> dict:
    """Emulate the one-time gate calibration for this synthetic geometry.

    Exactly what an ArUco sheet gives on the real gate: four points whose
    cart-plane position (cm) is known, imaged through each camera. Because the
    synthetic cameras are fixed, one homography per camera is valid for the whole
    set — which is what lets the decision layer run in assignment mode.
    """
    Wcm, Hcm = CART_PLANE_CM
    plane_px = [(0, 0), (size, 0), (size, size), (0, size)]       # canvas corners
    plane_cm = [(x * Wcm / size, y * Hcm / size) for x, y in plane_px]
    H, report = {}, {}
    for d, name in zip(dirs, names):
        M = camera_matrix(float(d), size, size)
        img_pts = [calibrate_plane.project(M, p) for p in plane_px]
        H[name] = calibrate_plane.calibrate_from_points(img_pts, plane_cm)
        report[name] = calibrate_plane.check_calibration(H[name], img_pts, plane_cm)
    calibrate_plane.save(str(path), H, meta={
        "plane_cm": f"{Wcm:g}x{Hcm:g} cart opening", "source": "synthetic gate geometry",
        "merge_radius_cm": 12.0, "reprojection": {k: round(v["max_cm"], 4) for k, v in report.items()}})
    return report


def load_cutouts(d="out/cut_rembg"):
    cut = defaultdict(list)
    for p in sorted(Path(d).glob("*.png")):
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is not None and im.ndim == 3 and im.shape[2] == 4:
            cut[p.stem.split("__")[0]].append(im)
    return dict(cut)


def make_cart(skus, rng):
    """Return (receipt {sku:qty}, contents [sku...], scenario, verdict, reason)."""
    k = int(rng.integers(2, 6))
    paid = list(rng.choice(skus, size=k, replace=False))
    receipt = {s: int(rng.integers(1, 3)) for s in paid}
    contents = [s for s, q in receipt.items() for _ in range(q)]
    r = rng.random()
    if r < 0.60:
        return receipt, contents, "normal", "PASS", "cart matches receipt"
    if r < 0.80:
        extra = next(s for s in rng.permutation(skus) if s not in receipt)
        contents.append(extra)
        return receipt, contents, "unpaid_item", "FLAG", f"unpaid item present: {extra}"
    over = paid[int(rng.integers(len(paid)))]
    contents.append(over)
    return receipt, contents, "quantity_excess", "FLAG", f"more '{over}' than paid"


def camera_names(n: int) -> list[str]:
    """Two cameras get the deployment names the pipeline uses."""
    return ["cam_left", "cam_right"] if n == 2 else [f"cam{k}" for k in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=500, help="carts (each = cameras x frames images)")
    ap.add_argument("--cameras", type=int, default=2)
    ap.add_argument("--frames", type=int, default=4, help="consecutive frames per camera")
    ap.add_argument("--out", default="cart_dataset")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--seed-offset", type=int, default=1000,
                    help="cart i uses seed seed_offset+i; change it for a disjoint set")
    args = ap.parse_args()

    out = Path(args.out)
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    cut = load_cutouts()
    skus = list(cut.keys())
    dirs = list(np.linspace(-1.0, 1.0, args.cameras)) if args.cameras > 1 else [0.0]
    names = camera_names(args.cameras)

    carts, counts = [], Counter()
    for i in range(args.num):
        rng = np.random.default_rng(args.seed_offset + i)
        receipt, contents, scenario, verdict, reason = make_cart(skus, rng)
        counts[scenario] += 1
        views = synth_cart_views(contents, cut, rng, cameras=dirs, n_frames=args.frames,
                                 size=(args.size, args.size))
        split = "val" if rng.random() < args.val_frac else "train"
        cams = []
        for d, cam_name in zip(dirs, names):
            frames_meta = []
            for fi, f in enumerate(views[float(d)]):
                name = f"cart_{i:05d}_{cam_name}_f{fi}"
                cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), f.image)
                lines = []
                for o in f.objects:
                    x0, y0, x1, y1 = o.box
                    cx, cy = (x0 + x1) / 2 / args.size, (y0 + y1) / 2 / args.size
                    bw, bh = (x1 - x0) / args.size, (y1 - y0) / args.size
                    lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
                frames_meta.append({
                    "image": f"images/{split}/{name}.jpg",
                    "objects": [{"sku": o.sku, "track_id": int(o.track_id), "box": list(o.box)}
                                for o in f.objects]})
            cams.append({"camera_id": cam_name, "direction": float(d), "frames": frames_meta})
        carts.append({"id": f"cart_{i:05d}", "format_version": FORMAT_VERSION,
                      "scenario": scenario, "verdict": verdict, "reason": reason,
                      "receipt": receipt, "contents": dict(Counter(contents)),
                      "split": split, "cameras": cams})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.num}", flush=True)

    json.dump(carts, open(out / "carts.json", "w"), ensure_ascii=False, indent=1)
    rep = write_calibration(out / "gate_calib.json", dirs, names, args.size)
    print("calibration ->", out / "gate_calib.json",
          "| reprojection max_cm:", {k: round(v["max_cm"], 3) for k, v in rep.items()})
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: product\n")
    ntr = len(list((out / "images" / "train").glob("*.jpg")))
    nva = len(list((out / "images" / "val").glob("*.jpg")))
    total = args.num * args.cameras * args.frames
    print(f"\n{args.num} carts x {args.cameras} cams x {args.frames} frames = {total} images -> {out}/")
    print("scenarios:", dict(counts), "| train/val imgs:", ntr, nva)


if __name__ == "__main__":
    main()
