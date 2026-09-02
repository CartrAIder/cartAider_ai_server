"""VISION SIDE end-to-end demo: capture -> detect -> embed -> per-camera tracks
-> cross-camera fusion -> VisionObservation (docs/CONTRACT_v1.1.md §3).

Boundary rule (contract §1): this script answers "what is there, how many".
It does NOT decide whether the cart matches the receipt — no verdict, no band,
no similarity threshold. The demo self-test at the bottom calls the DECISION
side's reference implementation (cartgate.verification.reference_verify) purely
to check the vision output is good enough to decide on; production wiring hands
the JSON to the teammate's VerificationService instead.

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

from cartgate import calibrate_plane, config, vision_fusion
from cartgate.embed import get_embedder
from cartgate.gallery import build_gallery
from cartgate.synth import synth_cart_views
from cartgate.match import sku_similarity
from cartgate.verification import reference_verify   # demo self-test only

CALIB_PATH = "gate_calib.json"
CROP_DIR = "out/crops"   # evidence store for cited instances
EMBED_BATCH = 8          # measured sweet spot on L40S: 1.72 ms/crop (vs 3.6 looped)

# How a track's per-frame similarities collapse into ONE number per SKU.
#   "max"  — best available view. Semantically right under occlusion, but biased
#            by track length: max over 5 frames beats max over 2 for the same object.
#   "mean" — length-unbiased, but frames where the item is buried dilute the one
#            clear view that identifies it.
#   "top2" — mean of the two best frames: keeps the clear views, damps the
#            single-lucky-frame outlier that "max" rewards.
# This interacts with the decision layer's SIM_STRONG/SIM_WEAK, so the two must
# be calibrated together on real footage (contract §1: thresholds are decision-owned).
#
# Measured over 500 carts (scripts/eval_carts.py): all three sit on the SAME
# false-stop/miss curve — at a matched false-stop level their miss rates differ by
# less than two standard errors, i.e. the choice re-parameterizes the threshold
# rather than buying accuracy. "max" is kept because it is the simplest to explain
# and holds the low-false-stop end of that curve.
CAND_AGG = "max"


def load_cutouts(cut_dir: str) -> dict:
    cut = defaultdict(list)
    for p in sorted(Path(cut_dir).glob("*.png")):
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is not None and im.ndim == 3 and im.shape[2] == 4:
            cut[p.stem.split("__")[0]].append(im)
    return dict(cut)


def load_fusion(calib_path: str = CALIB_PATH):
    """PlaneMatchFusion when calibrated, AsymmetricFusion otherwise.

    The file is whatever cartgate.calibrate_plane.save() wrote: per-camera
    homographies (image px -> cart-plane cm) plus a meta block. The merge radius
    comes from meta.merge_radius_cm when calibration measured one
    (check_cross_camera reports a suggestion), else the module default.
    """
    p = Path(calib_path)
    if not p.exists():
        return vision_fusion.AsymmetricFusion()
    H = calibrate_plane.load(str(p))
    meta = json.loads(p.read_text()).get("meta", {})
    return vision_fusion.PlaneMatchFusion(
        H, merge_radius_cm=float(meta.get("merge_radius_cm",
                                          vision_fusion.MERGE_RADIUS_CM)))


def save_evidence(observation: dict, verdict: dict, crop_store: dict,
                  root: str = CROP_DIR) -> int:
    """Write the crops the decision layer actually cited and fill their crop_refs.

    Path convention (evidence store, contract §4):
        out/crops/{transaction_id}/{instance_id}_{camera_id}.jpg

    Only instances behind a FLAG or REVIEW are stored — a clean cart needs no
    evidence. The path is deterministic, so a production capture service can
    write these during capture instead of after the verdict.
    """
    reasons = verdict.get("reasons", [])
    cited = {r["instance_id"] for r in reasons if r.get("instance_id")}
    sku_only = {r["sku_id"] for r in reasons if r.get("sku_id") and not r.get("instance_id")}
    for inst in observation["instances"]:          # conservative mode cites a SKU, not an instance
        cand = inst.get("candidates") or {}
        if cand and sku_only and max(cand, key=cand.get) in sku_only:
            cited.add(inst["instance_id"])
    if not cited:
        return 0

    tx = observation["transaction_id"]
    outdir = Path(root) / tx
    outdir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for inst in observation["instances"]:
        if inst["instance_id"] not in cited:
            continue
        refs = {}
        for tid in inst["track_ids"]:
            cam, crop = crop_store.get(tid, (None, None))
            if crop is None or not crop.size:
                continue
            fn = f"{inst['instance_id']}_{cam}.jpg"
            cv2.imwrite(str(outdir / fn), crop)
            refs[cam] = f"{root}/{tx}/{fn}"
            saved += 1
        inst["crop_refs"] = refs
    return saved


def _aggregate(sims: list[float], how: str | None = None) -> float:
    """Collapse one track's per-frame similarities for a single SKU. See CAND_AGG."""
    how = how or CAND_AGG
    if not sims:
        return 0.0
    if how == "max":
        return float(max(sims))
    if how == "mean":
        return float(np.mean(sims))
    return float(np.mean(sorted(sims, reverse=True)[:2]))     # top2


def _reset_tracker(model) -> None:
    """ByteTrack state is per camera sequence: carry it across a camera's frames,
    never across cameras or carts (ids would leak between physical scenes)."""
    pred = getattr(model, "predictor", None)
    for t in getattr(pred, "trackers", None) or []:
        if hasattr(t, "reset"):
            t.reset()


def resolve_camera(model, frames, embedder, gallery, receipt_skus, dev,
                   camera_id: str = "cam0",
                   raw_out: dict | None = None) -> tuple[list[vision_fusion.Detection], dict]:
    """One camera's frame sequence -> one Detection per tracked object.

    Object identity comes from ByteTrack (ultralytics `model.track`), so nothing
    here reads the synthetic ground truth — the same code path works on real
    footage. Crops are embedded in batches (the ONNX graph has a dynamic batch
    axis), and every receipt SKU's similarity is kept: the decision layer's
    global assignment needs the full vector, not the argmax. Candidates are
    restricted to this cart's receipt by design (contract §3), never the full
    catalog.

    Returns (detections, {track_id: crop}) — the crop is kept so the evidence
    store can save it if the decision layer ends up citing that instance.
    raw_out, when given, receives {track_id: {sku: [per-frame sims]}} so an
    offline sweep can re-aggregate without re-running the models.
    """
    rc = sorted(set(str(s) for s in receipt_skus))
    prefix = "L" if camera_id.endswith("left") else ("R" if camera_id.endswith("right") else "T")
    mid = (len(frames) - 1) / 2.0        # stand-in for the QR trigger instant, see below
    tracks = defaultdict(lambda: {"cand": defaultdict(list), "n": 0,
                                  "obs": []})        # obs: (frame_idx, box, det_conf, crop)
    _reset_tracker(model)
    for fi, f in enumerate(frames):
        res = model.track(f.image, conf=config.DET_CONF, persist=True,
                          tracker="bytetrack.yaml", verbose=False, device=dev)[0]
        if res.boxes is None or res.boxes.id is None:
            continue                                 # nothing tracked in this frame
        ids = res.boxes.id.cpu().numpy().astype(int)
        hits = []                                    # (track_id, box, det_conf, crop)
        for tid, box, det_conf in zip(ids, res.boxes.xyxy.cpu().numpy(),
                                      res.boxes.conf.cpu().numpy()):
            bx = tuple(int(v) for v in box)
            crop = f.image[max(0, bx[1]):bx[3], max(0, bx[0]):bx[2]]
            if crop.size:
                hits.append((int(tid), bx, float(det_conf), crop))

        for s in range(0, len(hits), EMBED_BATCH):   # batched embedding
            chunk = hits[s:s + EMBED_BATCH]
            vecs = embedder.embed_batch([h[3] for h in chunk])
            for (tid, bx, det_conf, crop), vec in zip(chunk, vecs):
                t = tracks[tid]
                t["n"] += 1
                for sku in rc:
                    t["cand"][sku].append(sku_similarity(vec, gallery, sku))
                t["obs"].append((fi, [int(v) for v in bx], det_conf, crop))

    dets, crops = [], {}
    for tid, t in sorted(tracks.items()):
        if not t["n"]:
            continue
        # Representative observation = the frame closest to the QR trigger. The
        # homography is calibrated for where the cart stands at t=0, and the cart
        # keeps rolling, so a later frame projects to the wrong plane position.
        # TODO: frames carry no capture timestamp yet -> use the middle frame of
        # the burst as the trigger stand-in; switch to the real t=0 once the
        # ring-buffer capture stamps frames.
        fidx, box, _, crop = min(t["obs"], key=lambda o: abs(o[0] - mid))
        track_id = f"{prefix}{tid}"
        crops[track_id] = crop
        if raw_out is not None:
            raw_out[track_id] = {s: list(v) for s, v in t["cand"].items()}
        dets.append(vision_fusion.Detection(
            camera_id=camera_id,
            track_id=track_id,
            candidates={s: round(_aggregate(v), 4) for s, v in sorted(t["cand"].items())},
            n_frames=int(t["n"]),
            box=tuple(box),
            det_conf=round(float(np.mean([o[2] for o in t["obs"]])), 3),
            crop_ref=None,                           # filled by save_evidence() on demand
        ))
    return dets, crops


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
