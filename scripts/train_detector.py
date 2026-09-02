"""Train the class-agnostic 'product' detector (YOLO11n) on synthetic carts.
Copies the best checkpoint to a fixed path: runs/detector/best.pt."""
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

print("cuda available:", torch.cuda.is_available())
dev = 0 if torch.cuda.is_available() else "cpu"
m = YOLO("yolo11n.pt")
m.train(data="cart_dataset/data.yaml", epochs=60, imgsz=640, batch=32,
        device=dev, project="runs", name="detector_train", exist_ok=True,
        workers=8, patience=15, verbose=True,
        # robustness augmentation (moving carts, free arrangement, occlusion, multi-view)
        degrees=10.0, translate=0.1, scale=0.5, shear=2.0, perspective=0.0005,
        flipud=0.2, fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        mosaic=1.0, close_mosaic=10, mixup=0.15, copy_paste=0.1, erasing=0.4)
metrics = m.val(data="cart_dataset/data.yaml", device=dev)
print("VAL mAP50:", round(float(metrics.box.map50), 4), "mAP50-95:", round(float(metrics.box.map), 4))

Path("runs/detector").mkdir(parents=True, exist_ok=True)
shutil.copy(m.trainer.best, "runs/detector/best.pt")
print("best weights ->", "runs/detector/best.pt")
