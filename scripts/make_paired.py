# -*- coding: utf-8 -*-
"""Pair every dataset image with its BiRefNet cutout under dataset_paired/.

Mirrors the dataset/images/ layout; per image writes <stem><ext> (original, copied)
and <stem>.cut.png (RGBA cutout at the original resolution, so it overlays exactly).

  conda run -n cartgate python make_paired.py
"""
import shutil
from pathlib import Path
import numpy as np, cv2
from PIL import Image
from rembg import new_session, remove

SRC = Path("dataset/images")
DST = Path("dataset_paired")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    DST.mkdir(exist_ok=True)
    sess = new_session("birefnet-general")
    folders = pairs = 0
    for d in sorted(p for p in SRC.iterdir() if p.is_dir()):
        out = DST / d.name
        out.mkdir(exist_ok=True)
        folders += 1
        for ip in sorted(d.iterdir()):
            if ip.suffix.lower() not in IMG_EXT:
                continue
            shutil.copy2(ip, out / ip.name)                     # original, untouched
            bgr = cv2.imread(str(ip))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = remove(Image.fromarray(rgb), session=sess, post_process_mask=True)
            rgba = np.array(res.convert("RGBA"))                # full-size, aligned with original
            cv2.imwrite(str(out / f"{ip.stem}.cut.png"), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
            pairs += 1
        print(f"  {d.name}: {pairs}", flush=True) if folders % 10 == 0 else None
    print(f"\ndone: {folders} SKU folders, {pairs} (original + cutout) pairs -> {DST}/", flush=True)


if __name__ == "__main__":
    main()
