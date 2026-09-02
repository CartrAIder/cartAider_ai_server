# Handoff

Code is on GitHub; data and models are too large for git and ship as separate
cloud zips. Below is what to send and to whom.

## Deliverables

| # | Item | Size | Contents | For |
|---|------|------|----------|-----|
| 1 | GitHub repo | — | all code (`cartgate/`, `scripts/`, `docs/`, configs, `products.csv`) | everyone |
| 2 | `cart_dataset.zip` | ~310M | format **v2**: 500 carts x 2 cameras x 4 consecutive frames (4000 imgs) + `carts.json` (decision benchmark, tracking GT) + `gate_calib.json` (per-camera homography) + YOLO `labels/` + `data.yaml` | decision-layer teammate · detector training |
| 3 | `cjs_data_bundle.zip` | ~460M | `dataset/` (gallery) + models (`dino_arc.onnx`, detector `best.pt`, `yolo11n.pt`) + `out/cut_rembg/` (recognition-training cutouts) + `products.csv` | running / retraining the vision pipeline |
| 4 | `service_products.zip` | small | product master (barcode · name · category) | backend teammate |
| 5 | `cartgate_models.zip` | ~102M | **models to deploy**: `dino_arc.onnx`, detector `best.pt` + `deploy/*.onnx`, prebuilt `out/gallery.pkl` (no photo set needed), `products.csv`, sample calibration, integration docs | backend / whoever serves the model |

## Setup for a recipient

```bash
git clone <repo> && cd <repo>
conda create -n cartgate python=3.11 -y && conda activate cartgate
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt && pip install -e .
# unzip the bundles you were given into the repo root, then:
python scripts/pipeline.py                 # needs bundle #3
```

## Who needs what

- **Decision-layer teammate**: #1 + #2 (develop reconciliation on `carts.json` GT).
  Add #3 if they want to run the real detector/recognizer to get live model output.
  Start at `docs/DECISION_LAYER.md`.
- **Detector / vision (retraining)**: #1 + #2 (`yolo detect train ... data=cart_dataset/data.yaml`)
  + #3 for the gallery, recognition cutouts and current models.
- **Backend (serving the model)**: #1 + #5. Start at `docs/BACKEND_INTEGRATION.md`: it maps every
  file to "runtime" or "offline", gives the model I/O specs and a verified serving snippet.
- **Backend (payment/catalog side)**: #4 (product master) — `sku_id` links it to the vision output.

Regenerate the cart dataset: `python scripts/make_cart_dataset.py --num 500 --frames 4 --cameras 2`.

Start any interface work from `docs/CONTRACT_v1.1.md`; `scripts/eval_carts.py`
reports the operating point (false-stop / miss) the whole stack currently sits at.
