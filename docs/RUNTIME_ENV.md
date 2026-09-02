# Runtime environment notes (GPU / onnxruntime)

Measured on the dev box: NVIDIA L40S 46GB, Intel Xeon Platinum 8568Y+ (192 vCPU),
driver 560.35.05 (CUDA 12.6), torch 2.5.1+cu124, ultralytics 8.4.104.

## onnxruntime version is pinned, and the pin matters

`requirements.txt` pins **`onnxruntime-gpu==1.22.0`**.

- `onnxruntime-gpu>=1.23` (incl. 1.27) is built against **CUDA 13** and fails to
  load here with `ImportError: libcudart.so.13: cannot open shared object file`.
- The plain `onnxruntime` (CPU) wheel *installs and runs fine*, which is the
  trap: `InferenceSession(..., providers=["CUDAExecutionProvider", ...])`
  emits one warning and silently falls back to CPU. That cost 22.8 ms/crop
  instead of 3.6 ms/crop and was invisible in the pipeline output.

Always confirm what actually loaded, never what was requested:

```python
emb = get_embedder("dino_arc.onnx", pad=True)
print(emb.providers)     # -> ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

`scripts/pipeline.py` prints this on every run for the same reason.

## The CUDA EP needs cuDNN 9 on the loader path

onnxruntime-gpu does not ship cuDNN. On this box the only copy is the one
bundled inside the torch wheel (`torch/lib/libcudnn.so.9`), so the EP loads
**only if `torch` was imported first**:

```
# torch imported first          -> ['CUDAExecutionProvider', 'CPUExecutionProvider']
# onnxruntime alone             -> Failed to create CUDAExecutionProvider.
#                                  Require cuDNN 9.* and CUDA 12.*  -> CPU fallback
```

`cartgate/embed.py:_preload_cuda_libs()` imports torch before creating the
session for exactly this reason. Do not remove it without providing cuDNN some
other way.

**On Jetson (AGX / Orin) or any torch-free deployment**, provide cuDNN
explicitly — either

```bash
pip install nvidia-cudnn-cu12          # then the loader finds it via site-packages
```

or point the loader at an existing install:

```bash
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
```

JetPack ships CUDA/cuDNN system-wide, and the Jetson build of onnxruntime-gpu
comes from the NVIDIA index rather than PyPI — the version pin above is for
x86 dev boxes, not for the device image.

## Measured latency (median)

| Stage | CPU | CUDA EP |
|---|---|---|
| Detector, 1 frame @640 (YOLO11n) | 40.1 ms | **6.9 ms** |
| Embedder, 1 crop (loop) | 23.5 ms | **3.6 ms** |
| Embedder, batched B=8 | — | **1.72 ms/crop** |
| Embedder, batched B=16 | — | 2.27 ms/crop (regresses) |

`cartgate/embed.py:embed_batch()` is the batched path; `scripts/pipeline.py`
uses it with `EMBED_BATCH = 8`. Batch and loop agree to 2e-4 max abs diff.

## Gallery build

`build_gallery()` embeds ~2310 views (51 SKUs × studio rotations + 16 synthetic
composites each). Cold build is ~108 s on GPU; the fingerprinted cache in
`out/gallery.pkl` + `out/gallery.key.json` makes subsequent runs ~0.01 s. The
cache invalidates automatically when any source photo's path/mtime/size changes.
