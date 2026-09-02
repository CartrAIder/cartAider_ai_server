"""VISION SIDE — perception only. No verdict, no receipt logic, no thresholds
on similarity.

Owned by: 비전 담당
Produces: VisionObservation (plain JSON dict) -> handed to the decision layer.

Boundary rule
    이 모듈은 "무엇이 몇 개 있는가"까지만 답한다.
    "결제와 맞는가"는 decision_verify.py 의 일이다.

Consequence of that rule
  * similarity thresholds (strong/weak) live in the DECISION layer, not here.
    This module never decides whether 0.47 is "good enough" - it just reports
    the number. Recalibrating similarity after field data lands therefore
    touches the decision layer only.
  * cross-camera fusion is PURELY GEOMETRIC. Detections are merged by plane
    position, deliberately ignoring the SKU label, so that two cameras
    disagreeing about one object still yield ONE object.
  * this module DOES own capture/geometry parameters: min_frames,
    merge_radius_cm, detector confidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

MIN_FRAMES = 2          # vision-owned: observation quality
MERGE_RADIUS_CM = 12.0  # vision-owned: geometry


# --------------------------------------------------------------------------

@dataclass
class Detection:
    """One tracked object as seen by ONE camera.

    `candidates` = cosine similarity against every SKU on THIS CART'S RECEIPT.
    Never the full catalog. All of them are kept (not just the argmax) because
    the decision layer's global assignment needs the full score vector.
    """
    camera_id: str
    track_id: str
    candidates: dict[str, float]
    n_frames: int
    box: tuple[float, float, float, float]
    det_conf: float | None = None
    crop_ref: str | None = None

    @property
    def stable(self) -> bool:
        return self.n_frames >= MIN_FRAMES

    def support_point(self) -> tuple[float, float]:
        """Bottom-center: least parallax when projected onto the cart plane."""
        x0, y0, x1, y1 = self.box
        return ((x0 + x1) / 2.0, y1)


@dataclass
class Instance:
    """One physical object after cross-camera merging."""
    detections: list[Detection] = field(default_factory=list)
    plane_xy: tuple[float, float] | None = None

    @property
    def candidates(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for d in self.detections:
            for sku, sim in d.candidates.items():
                out[sku] = max(out.get(sku, 0.0), float(sim))
        return out

    @property
    def n_frames(self) -> int:
        return max(d.n_frames for d in self.detections)

    @property
    def stable(self) -> bool:
        return any(d.stable for d in self.detections)

    @property
    def label_conflict(self) -> bool:
        """Cameras disagree about which SKU this object is. Informational -
        the decision layer resolves it via global assignment."""
        tops = {max(d.candidates, key=d.candidates.get)
                for d in self.detections if d.candidates}
        return len(tops) > 1


# --------------------------------------------------------------------------
# fusion strategies
# --------------------------------------------------------------------------

class AsymmetricFusion:
    """No calibration available. Cannot resolve cross-camera duplicates, so it
    reports each camera's detections separately and tells the decision layer
    so via cross_camera_resolved=False."""
    name = "asymmetric"
    cross_camera_resolved = False

    def fuse(self, per_cam: dict[str, list[Detection]]) -> list[Instance]:
        return [Instance([d]) for dets in per_cam.values() for d in dets]


class PlaneMatchFusion:
    """Calibrated. Merges detections by cart-plane position ALONE.

    The SKU label is deliberately excluded from the matching cost: two cameras
    that disagree about one object must still merge into one object, otherwise
    a misrecognition is silently counted twice.
    """
    name = "plane_match"
    cross_camera_resolved = True

    def __init__(self, homographies: dict[str, np.ndarray],
                 merge_radius_cm: float = MERGE_RADIUS_CM):
        self.H = homographies
        self.merge_radius = merge_radius_cm

    def _project(self, d: Detection) -> tuple[float, float]:
        px, py = d.support_point()
        v = self.H[d.camera_id] @ np.array([px, py, 1.0], np.float64)
        if abs(v[2]) < 1e-9:
            return (float("inf"), float("inf"))
        return (float(v[0] / v[2]), float(v[1] / v[2]))

    def fuse(self, per_cam: dict[str, list[Detection]]) -> list[Instance]:
        cams = sorted(per_cam.keys())
        if len(cams) != 2:
            return AsymmetricFusion().fuse(per_cam)

        A = [d for d in per_cam[cams[0]] if d.stable]
        B = [d for d in per_cam[cams[1]] if d.stable]
        pa, pb = [self._project(d) for d in A], [self._project(d) for d in B]

        instances, ma, mb = [], set(), set()
        if A and B:
            cost = np.array([[float(np.hypot(p[0] - q[0], p[1] - q[1]))
                              for q in pb] for p in pa], np.float64)
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] <= self.merge_radius:
                    mid = ((pa[i][0] + pb[j][0]) / 2, (pa[i][1] + pb[j][1]) / 2)
                    instances.append(Instance([A[i], B[j]], mid))
                    ma.add(i)
                    mb.add(j)
        for i, d in enumerate(A):
            if i not in ma:
                instances.append(Instance([d], pa[i]))
        for j, d in enumerate(B):
            if j not in mb:
                instances.append(Instance([d], pb[j]))
        return instances


# --------------------------------------------------------------------------
# VisionObservation — the contract artifact
# --------------------------------------------------------------------------

def build_observation(per_cam: dict[str, list[Detection]], fusion, *,
                      transaction_id: str, gate_id: str, captured_at: str,
                      duration_ms: int, vision_status: str = "OK",
                      frames_used: dict[str, int] | None = None) -> dict:
    """Serialize perception output as a plain JSON-safe dict.

    NOTE: no verdict, no band, no threshold interpretation. Everything here is
    an observation; nothing is a judgement.
    """
    instances = fusion.fuse(per_cam)
    out_inst = []
    for k, inst in enumerate(instances):
        out_inst.append({
            "instance_id": f"I{k + 1}",
            "track_ids": [d.track_id for d in inst.detections],
            "camera_ids": sorted({d.camera_id for d in inst.detections}),
            "plane_xy": ([round(float(v), 2) for v in inst.plane_xy]
                         if inst.plane_xy else None),
            "candidates": {str(s): round(float(v), 4)
                           for s, v in inst.candidates.items()},
            "n_frames": int(inst.n_frames),
            "stable": bool(inst.stable),
            "label_conflict": bool(inst.label_conflict),
            "boxes": {d.camera_id: [float(x) for x in d.box]
                      for d in inst.detections},
            "crop_refs": {d.camera_id: d.crop_ref
                          for d in inst.detections if d.crop_ref},
        })

    return {
        "schema_version": "1.1",
        "transaction_id": str(transaction_id),
        "gate_id": str(gate_id),
        "captured_at": captured_at,
        "duration_ms": int(duration_ms),
        "vision_status": vision_status,
        "cameras": [{"camera_id": c, "status": "OK",
                     "frames_used": int((frames_used or {}).get(c, 0))}
                    for c in sorted(per_cam.keys())],
        "fusion_strategy": fusion.name,
        "cross_camera_resolved": bool(fusion.cross_camera_resolved),
        "min_frames": MIN_FRAMES,
        "instances": out_inst,
    }
