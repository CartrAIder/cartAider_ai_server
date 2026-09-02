"""Export the product detector to ONNX at 640 and 512 input sizes for edge deployment.
512 is the Nano-friendly setting. The TensorRT engine is hardware-specific and must be
built on the Jetson from the ONNX.
"""
import shutil
from pathlib import Path

from ultralytics import YOLO

WEIGHTS = "runs/detector/best.pt"


def main():
    m = YOLO(WEIGHTS)
    out = Path("deploy")
    out.mkdir(exist_ok=True)
    print(f"model params: {sum(p.numel() for p in m.model.parameters())/1e6:.2f}M")
    for imgsz in (640, 512):
        p = Path(m.export(format="onnx", imgsz=imgsz, opset=12, simplify=True,
                          dynamic=False, device="cpu"))
        dst = out / f"product_det_{imgsz}.onnx"
        shutil.move(str(p), dst)
        print(f"  ONNX imgsz={imgsz}: {dst}  ({dst.stat().st_size/1e6:.1f} MB)")

    print("\nOn the Jetson (Xavier AGX / Orin Nano), build the TensorRT engine from ONNX:")
    print("  trtexec --onnx=product_det_512.onnx --saveEngine=product_det_512_int8.engine \\")
    print("          --int8 --fp16 --workspace=2048   # INT8 needs a small calibration set")
    print("  (FP16 ~6-12MB engine; INT8 ~3-5MB; CUDA/TRT runtime context dominates RAM ~0.3-0.7GB)")


if __name__ == "__main__":
    main()
