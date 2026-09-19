"""Vision runtime — the part a service actually calls.

    frames + receipt SKUs  ->  per-camera Detections  ->  VisionObservation

Everything here is inference: detection, ByteTrack, embedding, and the choice of
fusion strategy. No verdict, no receipt reconciliation, no HTTP — those live on
the decision/backend side (docs/CONTRACT_v1.1.md §1).

Typical wiring, models loaded once per process:

    from ultralytics import YOLO
    from cartgate.embed import get_embedder
    from cartgate.gallery import load_gallery
    from cartgate.vision import resolve_camera, load_fusion
    from cartgate import vision_fusion

    detector = YOLO("runs/detector/best.pt")
    embedder = get_embedder("dino_arc.onnx", pad=True)
    gallery  = load_gallery("out/gallery.npz")
    fusion   = load_fusion("gate_calib.json")

    per_cam = {}
    for cam_id, frames in captured.items():          # frames: objects with .image (BGR)
        dets, crops = resolve_camera(detector, frames, embedder, gallery,
                                     receipt_skus, dev=0, camera_id=cam_id)
        per_cam[cam_id] = dets
    observation = vision_fusion.build_observation(per_cam, fusion, ...)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from cartgate import calibrate_plane, config, vision_fusion
from cartgate.match import sku_similarity

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
