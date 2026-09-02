"""VISION SIDE end-to-end demo: capture -> detect -> embed -> per-camera tracks
-> cross-camera fusion -> VisionObservation (docs/CONTRACT_v1.1.md §3).

Boundary rule (contract §1): this script answers "what is there, how many".
It does NOT decide whether the cart matches the receipt — no verdict, no band,
no similarity threshold. The demo self-test at the bottom calls the DECISION
side's reference implementation (cartgate.verification.reference_verify) purely
to check the vision output is good enough to decide on; production wiring hands
the JSON to the teammate's VerificationService instead.

The runtime itself lives in cartgate/vision.py (resolve_camera / load_fusion);
this script is the demo harness around it — synthetic carts in, observation JSON
out, plus a self-test that runs the decision reference implementation.

Fusion strategy is chosen by calibration: gate_calib.json present ->
PlaneMatchFusion (instances are real physical objects, cross_camera_resolved
true), absent -> AsymmetricFusion (per-camera detections, the decision layer
falls back to its conservative mode).
"""
import argparse
import datetime as dt
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from cartgate import config, vision_fusion
from cartgate.embed import get_embedder
from cartgate.gallery import build_gallery
from cartgate.synth import synth_cart_views
from cartgate.vision import (CALIB_PATH, EMBED_BATCH, load_fusion, resolve_camera,
                             save_evidence)
from cartgate.verification import reference_verify   # demo self-test only

def load_cutouts(cut_dir: str) -> dict:
    cut = defaultdict(list)
    for p in sorted(Path(cut_dir).glob("*.png")):
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is not None and im.ndim == 3 and im.shape[2] == 4:
            cut[p.stem.split("__")[0]].append(im)
    return dict(cut)


def run(dataset, cutouts_dir, weights, onnx, dev, seed=7, out_path=None, n_frames=4,
        calib_path=CALIB_PATH):
    rng = np.random.default_rng(seed)
    embedder = get_embedder(onnx, pad=("dino" in onnx or "vit" in onnx))
    print(f"[1/4] embedder: {embedder.name}  (pad={embedder.pad}, "
          f"providers={getattr(embedder, 'providers', ['classical'])}, batch={EMBED_BATCH})")
    print("[2/4] building gallery (studio + synthetic enriched views)...")
    gallery = build_gallery(dataset, embedder, "out", remove_bg=False, enrich_synth=16)
    cutouts = load_cutouts(cutouts_dir)
    model = YOLO(weights)
    fusion = load_fusion(calib_path)
    print(f"[3/4] detector: {weights}  (device={dev})")
    print(f"      fusion: {fusion.name} (cross_camera_resolved={fusion.cross_camera_resolved})")

    skus = sorted(cutouts.keys())
    base = [str(s) for s in rng.choice(skus, size=3, replace=False)]
    extra = next(s for s in skus if s not in base)
    scenarios = [
        ("정상 결제 카트", list(base), {s: 1 for s in base}, "PASS"),
        ("미결제 물건 포함", base + [extra], {s: 1 for s in base}, "FLAG"),
        ("수량 초과(1결제 2적재)", base + [base[0]], {s: 1 for s in base}, "FLAG"),
        ("결제했지만 가려짐", base[:2], {s: 1 for s in base}, "PASS"),
    ]
    cams = [(-1.0, "cam_left"), (1.0, "cam_right")]      # 2 upper-diagonal cameras
    print("[4/4] gate scenarios (real detection -> fusion -> VisionObservation)\n")
    results, observations = [], []
    for i, (name, cart, receipt, expect) in enumerate(scenarios):
        # frames stand in for camera capture -> generated OUTSIDE the timer, so
        # duration_ms measures only detect+recognize+fuse (what a gate would spend).
        views = synth_cart_views(cart, cutouts, rng, cameras=[d for d, _ in cams],
                                 n_frames=n_frames, size=(640, 640))
        t0 = time.perf_counter()
        per_cam, crop_store = {}, {}
        for d, cam_id in cams:
            dets, crops = resolve_camera(model, views[float(d)], embedder, gallery,
                                         list(receipt.keys()), dev, camera_id=cam_id)
            per_cam[cam_id] = dets
            crop_store.update({tid: (cam_id, crop) for tid, crop in crops.items()})
        obs = vision_fusion.build_observation(
            per_cam, fusion,
            transaction_id=f"TX-DEMO-{i:03d}",
            gate_id=config.GATE_ID,
            captured_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            duration_ms=int((time.perf_counter() - t0) * 1000),
            frames_used={c: n_frames for c in per_cam})
        observations.append(obs)

        # --- demo self-test ONLY: the decision side is the teammate's to own ---
        verdict = reference_verify.verify(obs, receipt)
        n_saved = save_evidence(obs, verdict, crop_store)
        ok = verdict["verdict"] == expect
        results.append((name, expect, verdict, ok))
        n_inst = len(obs["instances"])
        print(f"  [{'OK ' if ok else 'MISS'}] {name:20s} expect {expect:6s} -> {verdict['verdict']:6s}"
              f" | instances={n_inst} counts={verdict['observed_counts']}"
              f" mode={verdict['decision_mode']} {obs['duration_ms']}ms"
              f"{f' crops={n_saved}' if n_saved else ''}")
        for r in verdict["reasons"]:
            print(f"           {r['severity']}: {r['code']} {r.get('sku_id') or r.get('instance_id', '')}")
    n_ok = sum(r[3] for r in results)
    print(f"\n  {n_ok}/{len(results)} scenarios as expected (decision = reference impl, demo only)")

    if out_path:
        Path(out_path).write_text(json.dumps(observations, ensure_ascii=False, indent=2))
        print(f"  wrote {len(observations)} VisionObservation(s) -> {out_path}")
    return results, observations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset/images")
    ap.add_argument("--cutouts", default="out/cut_rembg")
    ap.add_argument("--weights", default="runs/detector/best.pt")
    ap.add_argument("--onnx", default="dino_arc.onnx")  # DINOv2+ArcFace (padded, enriched gallery)
    ap.add_argument("--device", default="0")
    ap.add_argument("--frames", type=int, default=4, help="frames per camera")
    ap.add_argument("--calib", default=CALIB_PATH, help="gate_calib.json (absent -> asymmetric)")
    ap.add_argument("--out", default=None, help="write VisionObservation JSON here")
    args = ap.parse_args()
    dev = 0 if args.device.isdigit() else args.device
    run(args.dataset, args.cutouts, args.weights, args.onnx, dev,
        out_path=args.out, n_frames=args.frames, calib_path=args.calib)


if __name__ == "__main__":
    main()
