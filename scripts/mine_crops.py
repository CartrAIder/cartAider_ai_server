"""Mine recognition training crops from the cart benchmark.

The recognizer is currently trained on single-object cutouts composited onto
random backgrounds — clean, isolated, fully visible. At the gate it sees
something else entirely: a partly buried object inside a cart, with neighbouring
products in the same box. Measured gap: 90.8% top-1 on the old distribution vs
72-74% on real tracks.

This takes the benchmark's ground-truth boxes and writes exactly the crops the
pipeline would feed the embedder, labelled by SKU, so training can match
deployment.

    python scripts/mine_crops.py --carts cart_dataset/carts.json --out out/crops_train
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carts", default="cart_dataset/carts.json")
    ap.add_argument("--out", default="out/crops_train")
    ap.add_argument("--pad", type=float, default=0.02, help="box padding fraction")
    ap.add_argument("--min-side", type=int, default=24, help="skip crops smaller than this")
    ap.add_argument("--splits", default="train,val", help="which cart splits to mine")
    args = ap.parse_args()

    carts_path = Path(args.carts)
    carts = json.loads(carts_path.read_text())
    if carts and carts[0].get("format_version", 1) < 2:
        raise SystemExit("needs carts.json v2 (per-camera frames)")
    splits = set(args.splits.split(","))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    per_sku, skipped = Counter(), 0
    for cart in carts:
        if cart.get("split", "train") not in splits:
            continue
        for cam in cart["cameras"]:
            for fi, frame in enumerate(cam["frames"]):
                img = cv2.imread(str(carts_path.parent / frame["image"]))
                if img is None:
                    continue
                H, W = img.shape[:2]
                for oi, o in enumerate(frame["objects"]):
                    x0, y0, x1, y1 = o["box"]
                    px, py = int((x1 - x0) * args.pad), int((y1 - y0) * args.pad)
                    x0, y0 = max(0, x0 - px), max(0, y0 - py)
                    x1, y1 = min(W, x1 + px), min(H, y1 + py)
                    if x1 - x0 < args.min_side or y1 - y0 < args.min_side:
                        skipped += 1
                        continue
                    crop = img[y0:y1, x0:x1]
                    d = out / o["sku"]
                    d.mkdir(exist_ok=True)
                    cv2.imwrite(str(d / f"{cart['id']}_{cam['camera_id']}_f{fi}_{oi}.jpg"), crop)
                    per_sku[o["sku"]] += 1

    total = sum(per_sku.values())
    print(f"mined {total} crops over {len(per_sku)} SKUs -> {out}/ (skipped {skipped} tiny)")
    if per_sku:
        c = per_sku.most_common()
        print(f"  per-SKU: max {c[0][1]} ({c[0][0]}), min {c[-1][1]} ({c[-1][0]}), "
              f"median {sorted(per_sku.values())[len(per_sku)//2]}")


if __name__ == "__main__":
    main()
