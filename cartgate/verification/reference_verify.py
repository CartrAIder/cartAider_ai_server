"""DECISION SIDE — receipt reconciliation and verdict.

Owned by: 판정 담당 (팀원)
Consumes: VisionObservation (plain dict/JSON from the vision side)
Produces: GateVerdict (plain dict/JSON) -> backend

Boundary rule
    이 모듈은 카메라·기하·모델을 전혀 모른다.
    관측 결과(dict)와 영수증만 보고 판정한다.

This is the reference implementation for VerificationService in PR #1.
It replaces the per-SKU count comparison with a GLOBAL assignment, which is
what prevents a single misrecognition from becoming a false FLAG.

Threshold ownership
    similarity thresholds live HERE, not in vision. Recalibrating after field
    data lands touches this file only; the vision pipeline is untouched.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

# decision-owned thresholds (재보정 대상)
SIM_STRONG = 0.55
SIM_WEAK = 0.42


def _slots(receipt: dict[str, int]) -> list[str]:
    """One slot per paid unit: qty 2 -> two slots."""
    return [sku for sku, qty in receipt.items() for _ in range(int(qty))]


def verify(observation: dict, receipt: dict[str, int]) -> dict:
    """VisionObservation + receipt -> GateVerdict.

    FLAG   : an object that no receipt line can explain,
             or more objects of an SKU than were paid for
    REVIEW : an object explained only weakly
    PASS   : otherwise
    Paid-but-unseen items are reported, never penalized.
    """
    status = observation.get("vision_status", "OK")
    if status == "FAILED":
        # fail-open: stopping a customer on zero evidence is pure friction.
        return _verdict_doc(observation, "PASS", [{
            "code": "VISION_UNAVAILABLE", "severity": "INFO",
            "detail": "관측 실패 - 통과 처리 후 로깅"}], {}, receipt)

    insts = [i for i in observation.get("instances", []) if i.get("stable")]

    if not observation.get("cross_camera_resolved", False):
        return _verify_conservative(observation, insts, receipt)
    return _verify_assignment(observation, insts, receipt)


# --------------------------------------------------------------------------

def _verify_assignment(observation: dict, insts: list[dict],
                       receipt: dict[str, int]) -> dict:
    """Calibrated path: instances are real physical objects, so a global
    Hungarian assignment against receipt slots is valid."""
    slots = _slots(receipt)
    assigned: dict[int, int] = {}
    if insts and slots:
        cost = np.array([[1.0 - float(i.get("candidates", {}).get(s, 0.0))
                          for s in slots] for i in insts], np.float64)
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] <= 1.0 - SIM_WEAK:
                assigned[int(i)] = int(j)

    reasons, counts = [], defaultdict(int)
    for idx, inst in enumerate(insts):
        cand = inst.get("candidates", {})
        best_sim = max(cand.values()) if cand else 0.0
        if idx in assigned:
            sku = slots[assigned[idx]]
            counts[sku] += 1
            if cand.get(sku, 0.0) < SIM_STRONG:
                reasons.append({"code": "AMBIGUOUS_ITEM", "severity": "REVIEW",
                                "instance_id": inst["instance_id"],
                                "assigned_sku": sku,
                                "sim": round(cand.get(sku, 0.0), 3)})
            continue
        if best_sim < SIM_WEAK:
            reasons.append({"code": "UNEXPLAINED_ITEM", "severity": "FLAG",
                            "instance_id": inst["instance_id"],
                            "top_sim": round(best_sim, 3),
                            "crop_refs": inst.get("crop_refs", {})})
        else:
            sku = max(cand, key=cand.get)
            counts[sku] += 1
            reasons.append({"code": "QTY_EXCEEDED", "severity": "FLAG",
                            "instance_id": inst["instance_id"], "sku_id": sku,
                            "paid_qty": int(receipt.get(sku, 0)),
                            "observed_qty": int(receipt.get(sku, 0)) + 1,
                            "sim": round(cand[sku], 3),
                            "crop_refs": inst.get("crop_refs", {})})

    return _verdict_doc(observation, _severity(reasons), reasons,
                        dict(counts), receipt)


def _verify_conservative(observation: dict, insts: list[dict],
                         receipt: dict[str, int]) -> dict:
    """Uncalibrated path: cross-camera duplicates are unresolved, so the same
    object may appear once per camera. Counting per camera and taking the
    MINIMUM for the excess test keeps a single-camera misrecognition from
    stopping an honest customer. Presence uses the maximum so that an item
    occluded from one side is still credited."""
    per_cam: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    unexplained, ambiguous = [], []
    for inst in insts:
        cand = inst.get("candidates", {})
        best_sim = max(cand.values()) if cand else 0.0
        cam = (inst.get("camera_ids") or ["?"])[0]
        if best_sim >= SIM_STRONG:
            per_cam[cam][max(cand, key=cand.get)] += 1
        elif best_sim >= SIM_WEAK:
            ambiguous.append(inst)
        else:
            unexplained.append(inst)

    cams = list(per_cam.keys()) or ["?"]
    skus = {s for c in per_cam.values() for s in c}
    counts = {s: max(per_cam[c].get(s, 0) for c in cams) for s in skus}
    excess = {s: min(per_cam[c].get(s, 0) for c in cams) for s in skus}

    reasons = []
    for sku, obs in excess.items():
        paid = int(receipt.get(sku, 0))
        if obs > paid:
            reasons.append({"code": "QTY_EXCEEDED", "severity": "FLAG",
                            "sku_id": sku, "paid_qty": paid, "observed_qty": obs})
    for inst in unexplained:
        reasons.append({"code": "UNEXPLAINED_ITEM", "severity": "FLAG",
                        "instance_id": inst["instance_id"],
                        "crop_refs": inst.get("crop_refs", {})})
    for inst in ambiguous:
        reasons.append({"code": "AMBIGUOUS_ITEM", "severity": "REVIEW",
                        "instance_id": inst["instance_id"]})

    return _verdict_doc(observation, _severity(reasons), reasons, counts, receipt)


# --------------------------------------------------------------------------

def _severity(reasons) -> str:
    if any(r["severity"] == "FLAG" for r in reasons):
        return "FLAG"
    if any(r["severity"] == "REVIEW" for r in reasons):
        return "REVIEW"
    return "PASS"


def _verdict_doc(observation, verdict, reasons, counts, receipt) -> dict:
    unseen = [{"sku_id": s, "qty": int(q) - counts.get(s, 0)}
              for s, q in receipt.items() if int(q) > counts.get(s, 0)]
    return {
        "schema_version": "1.1",
        "transaction_id": observation.get("transaction_id"),
        "gate_id": observation.get("gate_id"),
        "verdict": verdict,
        "reasons": reasons,
        "observed_counts": {str(k): int(v) for k, v in counts.items()},
        "unseen_paid_items": unseen,     # 정보용. 판정에 반영하지 않음
        "vision_status": observation.get("vision_status", "OK"),
        "decision_mode": ("assignment" if observation.get("cross_camera_resolved")
                          else "conservative"),
    }
