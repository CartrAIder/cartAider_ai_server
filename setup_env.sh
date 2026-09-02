#!/usr/bin/env bash
# Fresh, GPU-capable conda env for CartGate (driver CUDA 12.6 -> pytorch cu12.4).
# Keeps opencv-contrib (SIFT) as the cv2 in the env, not the opencv-python that
# ultralytics would otherwise pull in.
set -e
CONDA=/opt/anaconda3/bin/conda
ENV=cartgate

echo "=== [1/5] create env (python 3.11) ==="
$CONDA create -y -n $ENV python=3.11

echo "=== [2/5] pytorch + torchvision (CUDA 12.4, works on 12.6 driver) ==="
$CONDA install -y -n $ENV -c pytorch -c nvidia pytorch torchvision pytorch-cuda=12.4

echo "=== [3/5] pip: pipeline deps ==="
$CONDA run -n $ENV pip install --no-input \
  onnxruntime opencv-contrib-python pillow-heif scikit-image scipy rembg ultralytics onnxscript

echo "=== [4/5] force opencv-contrib to own cv2 (SIFT) ==="
$CONDA run -n $ENV pip uninstall -y opencv-python || true
$CONDA run -n $ENV pip install --force-reinstall --no-deps opencv-contrib-python

echo "=== [5/5] verify ==="
$CONDA run -n $ENV python - <<'PY'
import torch, cv2, numpy as np
print("torch", torch.__version__, "| cuda.is_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  device0:", torch.cuda.get_device_name(0))
print("cv2", cv2.__version__, "| SIFT", hasattr(cv2, "SIFT_create"))
import onnxruntime, rembg, ultralytics, skimage, PIL, scipy
print("onnxruntime", onnxruntime.__version__, "| ultralytics", ultralytics.__version__, "| numpy", np.__version__)
print("ENV OK")
PY
echo "@@@ SETUP DONE @@@"
