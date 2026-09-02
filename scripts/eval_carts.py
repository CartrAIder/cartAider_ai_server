"""End-to-end evaluation on the synthetic cart benchmark (cart_dataset/carts.json).

What it measures — an operating point, not accuracy:

    false-stop rate = GT PASS carts that got FLAG or REVIEW
    miss rate       = GT FLAG carts that got PASS

REVIEW counts as a stop: from the customer's side, being held for a staff check
is the same experience as being flagged. Those two numbers have very different
business costs, so they are reported separately and never averaged.

Ground truth is used ONLY to score the final verdict. Tracking runs on the image
sequences through ByteTrack exactly as it would on real footage — no GT boxes
feed the pipeline, so the numbers are not self-fulfilling.

Vision runs once per cart and the per-frame similarities are kept raw, so the
CAND_AGG x threshold sweep afterwards is pure post-processing.

    python scripts/eval_carts.py --limit 500
"""
import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from ultralytics import YOLO

from cartgate import vision_fusion
from cartgate.embed import get_embedder
from cartgate.gallery import build_gallery
from cartgate.verification import reference_verify

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "cg_pipeline", str(Path(__file__).with_name("pipeline.py")))
pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline)

AGGS = ["max", "mean", "top2"]
GRID = [(s, w) for s in (0.45, 0.50, 0.55, 0.60, 0.65)
        for w in (0.30, 0.35, 0.42, 0.50) if w < s]


# --------------------------------------------------------------------------
# vision pass (run once, keep raw per-frame similarities)
# --------------------------------------------------------------------------

def observe_cart(cart: dict, root: Path, model, embedder, gallery, dev) -> dict:
    """Run detection+tracking+embedding over every camera of one cart."""
    receipt = {str(s): int(q) for s, q in cart["receipt"].items()}
    per_cam_raw, meta = {}, {}
    t0 = time.perf_counter()
    for cam in cart["cameras"]:
        frames = []
        for f in cam["frames"]:
            img = cv2.imread(str(root / f["image"]))
            if img is not None:
                frames.append(SimpleNamespace(image=img, objects=[]))
        if not frames:
            continue
        raw = {}
        dets, _ = pipeline.resolve_camera(model, frames, embedder, gallery,
                                          list(receipt.keys()), dev,
                                          camera_id=cam["camera_id"], raw_out=raw)
        per_cam_raw[cam["camera_id"]] = raw
        meta[cam["camera_id"]] = {d.track_id: {"n_frames": d.n_frames, "box": list(d.box),
                                               "det_conf": d.det_conf} for d in dets}
    return {"id": cart["id"], "scenario": cart["scenario"], "gt": cart["verdict"],
            "receipt": receipt, "raw": per_cam_raw, "meta": meta,
            "vision_ms": (time.perf_counter() - t0) * 1000}


def detections_for(rec: dict, agg: str) -> dict:
    """Rebuild Detections from the stored raw similarities under one aggregation."""
    per_cam = {}
    for cam, tracks in rec["raw"].items():
        dets = []
        for tid, sims in tracks.items():
            m = rec["meta"][cam][tid]
            dets.append(vision_fusion.Detection(
                camera_id=cam, track_id=tid,
                candidates={s: round(pipeline._aggregate(v, agg), 4) for s, v in sims.items()},
                n_frames=m["n_frames"], box=tuple(m["box"]), det_conf=m["det_conf"]))
        per_cam[cam] = dets
    return per_cam


def verdict_for(rec: dict, agg: str, fusion, strong: float, weak: float) -> str:
    reference_verify.SIM_STRONG, reference_verify.SIM_WEAK = strong, weak
    obs = vision_fusion.build_observation(
        detections_for(rec, agg), fusion, transaction_id=rec["id"],
        gate_id="eval", captured_at="1970-01-01T00:00:00+00:00",
        duration_ms=int(rec["vision_ms"]),
        frames_used={c: 0 for c in rec["raw"]})
    obs = json.loads(json.dumps(obs))          # the boundary is JSON
    return reference_verify.verify(obs, rec["receipt"])["verdict"]


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def score(records: list[dict], agg: str, fusion, strong: float, weak: float) -> dict:
    """false-stop and miss rates. REVIEW is a stop."""
    n_pass = n_flag = stops = misses = reviews_on_pass = 0
    by_scenario = defaultdict(Counter)
    for rec in records:
        v = verdict_for(rec, agg, fusion, strong, weak)
        by_scenario[rec["scenario"]][v] += 1
        if rec["gt"] == "PASS":
            n_pass += 1
            stops += int(v != "PASS")
            reviews_on_pass += int(v == "REVIEW")
        else:
            n_flag += 1
            misses += int(v == "PASS")
    return {"agg": agg, "sim_strong": strong, "sim_weak": weak,
            "n_pass": n_pass, "n_flag": n_flag,
            "false_stop_rate": stops / max(n_pass, 1),
            "miss_rate": misses / max(n_flag, 1),
            "review_share_of_stops": reviews_on_pass / max(stops, 1),
            "by_scenario": {k: dict(v) for k, v in by_scenario.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carts", default="cart_dataset/carts.json")
    ap.add_argument("--limit", type=int, default=0, help="0 = all carts")
    ap.add_argument("--dataset", default="dataset/images")
    ap.add_argument("--weights", default="runs/detector/best.pt")
    ap.add_argument("--onnx", default="dino_arc.onnx")
    ap.add_argument("--device", default="0")
    ap.add_argument("--calib", default=pipeline.CALIB_PATH)
    ap.add_argument("--out", default="out/eval_carts.json")
    args = ap.parse_args()

    carts_path = Path(args.carts)
    carts = json.loads(carts_path.read_text())
    if args.limit:
        carts = carts[:args.limit]
    if carts and carts[0].get("format_version", 1) < 2:
        raise SystemExit("carts.json is the one-frame-per-camera v1 format; regenerate "
                         "with scripts/make_cart_dataset.py --frames 4 (min_frames needs >=2)")

    dev = 0 if args.device.isdigit() else args.device
    embedder = get_embedder(args.onnx, pad=("dino" in args.onnx or "vit" in args.onnx))
    print(f"embedder providers: {getattr(embedder, 'providers', ['classical'])}")
    gallery = build_gallery(args.dataset, embedder, "out", remove_bg=False, enrich_synth=16)
    model = YOLO(args.weights)
    # Both fusion strategies are scored from the SAME vision pass: fusion runs
    # after detection/recognition, so only the post-processing differs.
    fusions = [vision_fusion.AsymmetricFusion()]
    calibrated = pipeline.load_fusion(args.calib)
    if calibrated.name != "asymmetric":
        fusions.append(calibrated)
    print(f"fusions: {[f.name for f in fusions]}")
    print(f"carts: {len(carts)}  gt: {Counter(c['verdict'] for c in carts)}")

    records, t0 = [], time.perf_counter()
    for i, cart in enumerate(carts):
        records.append(observe_cart(cart, carts_path.parent, model, embedder, gallery, dev))
        if (i + 1) % 50 == 0:
            el = time.perf_counter() - t0
            print(f"  {i + 1}/{len(carts)} carts  ({el:.0f}s, {el/(i+1)*1000:.0f} ms/cart)", flush=True)
    vis_ms = np.array([r["vision_ms"] for r in records])
    print(f"\nvision: mean {vis_ms.mean():.0f} ms/cart, median {np.median(vis_ms):.0f} ms "
          f"(2 cams x {len(carts[0]['cameras'][0]['frames'])} frames)\n")

    # --- headline table at the contract's default thresholds ---
    print(f"{'fusion':<12}{'agg':<6}{'strong':>8}{'weak':>7}{'false-stop':>12}{'miss':>9}   scenario breakdown")
    print("-" * 110)
    rows = []
    for fusion in fusions:
        for agg in AGGS:
            r = score(records, agg, fusion, 0.55, 0.42)      # contract §5 defaults
            r["fusion"] = fusion.name
            rows.append(r)
            print(f"{fusion.name:<12}{agg:<6}{r['sim_strong']:>8.2f}{r['sim_weak']:>7.2f}"
                  f"{r['false_stop_rate']:>11.1%}{r['miss_rate']:>9.1%}   {r['by_scenario']}")

    # --- threshold sweep: where does each combination sit on the trade-off? ---
    print(f"\n{'fusion':<12}{'agg':<6}{'strong':>8}{'weak':>7}{'false-stop':>12}{'miss':>9}")
    print("-" * 56)
    sweep = []
    for fusion in fusions:
        for agg in AGGS:
            for strong, weak in GRID:
                r = score(records, agg, fusion, strong, weak)
                r["fusion"] = fusion.name
                sweep.append(r)
                print(f"{fusion.name:<12}{agg:<6}{strong:>8.2f}{weak:>7.2f}"
                      f"{r['false_stop_rate']:>11.1%}{r['miss_rate']:>9.1%}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n_carts": len(carts), "fusions": [f.name for f in fusions],
               "vision_ms_mean": float(vis_ms.mean()),
               "default_thresholds": rows, "sweep": sweep},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
