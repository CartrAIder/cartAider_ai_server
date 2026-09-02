"""Boundary test: vision -> JSON -> decision.

Proves the two sides are decoupled: the decision layer only ever sees a plain
dict that survived json.dumps/loads. No shared Python objects.
"""
import json

import numpy as np
import pytest

from cartgate import vision_fusion as V
from cartgate.verification import reference_verify as D

H = {"cam_left": np.eye(3),
     "cam_right": np.array([[1, 0, -200.0], [0, 1, 0], [0, 0, 1]])}


def det(cam, tid, cands, x, n_frames=4):
    return V.Detection(camera_id=cam, track_id=tid, candidates=cands,
                       n_frames=n_frames, box=(x - 25, 100, x + 25, 150),
                       crop_ref=f"{cam}/{tid}.jpg")


def case_real_bug():
    """The actual pipeline.py failure: cam_left misreads S0035 as S0032."""
    receipt = {"S0047": 1, "S0032": 1, "S0035": 1}
    return "정상 카트(한쪽 오인식)", receipt, {
        "cam_left": [
            det("cam_left", "L1", {"S0032": 0.815, "S0047": 0.21, "S0035": 0.30}, 125),
            det("cam_left", "L2", {"S0047": 0.661, "S0032": 0.19, "S0035": 0.25}, 325),
            det("cam_left", "L3", {"S0032": 0.726, "S0035": 0.710, "S0047": 0.22}, 525),
        ],
        "cam_right": [
            det("cam_right", "R1", {"S0032": 0.785, "S0047": 0.20, "S0035": 0.28}, 325),
            det("cam_right", "R2", {"S0035": 0.723, "S0032": 0.51, "S0047": 0.18}, 725),
            det("cam_right", "R3", {"S0047": 0.759, "S0032": 0.17, "S0035": 0.24}, 525),
        ]}, "PASS"


def case_excess():
    receipt = {"S0047": 1, "S0032": 1}
    return "실제 수량 초과", receipt, {
        "cam_left": [det("cam_left", "L1", {"S0047": 0.80, "S0032": 0.20}, 125),
                     det("cam_left", "L2", {"S0047": 0.78, "S0032": 0.19}, 425),
                     det("cam_left", "L3", {"S0032": 0.81, "S0047": 0.18}, 625)],
        "cam_right": [det("cam_right", "R1", {"S0047": 0.79, "S0032": 0.21}, 325),
                      det("cam_right", "R2", {"S0047": 0.77, "S0032": 0.20}, 625),
                      det("cam_right", "R3", {"S0032": 0.82, "S0047": 0.17}, 825)]}, "FLAG"


def case_unpaid():
    receipt = {"S0047": 1}
    return "미결제 물건 포함", receipt, {
        "cam_left": [det("cam_left", "L1", {"S0047": 0.80}, 125),
                     det("cam_left", "L2", {"S0047": 0.31}, 425)],
        "cam_right": [det("cam_right", "R1", {"S0047": 0.79}, 325),
                      det("cam_right", "R2", {"S0047": 0.29}, 625)]}, "FLAG"


def case_occluded():
    receipt = {"S0047": 1, "S0032": 1, "S0035": 1}
    return "결제했지만 가려짐", receipt, {
        "cam_left": [det("cam_left", "L1", {"S0047": 0.80, "S0032": 0.2, "S0035": 0.2}, 125)],
        "cam_right": [det("cam_right", "R1", {"S0047": 0.79, "S0032": 0.2, "S0035": 0.2}, 325),
                      det("cam_right", "R2", {"S0032": 0.75, "S0047": 0.2, "S0035": 0.3}, 525)]}, "PASS"


def case_one_cam_blind():
    receipt = {"S0047": 1, "S0032": 1}
    return "한쪽 카메라 가림", receipt, {
        "cam_left": [],
        "cam_right": [det("cam_right", "R1", {"S0047": 0.79, "S0032": 0.2}, 325),
                      det("cam_right", "R2", {"S0032": 0.75, "S0047": 0.2}, 525)]}, "PASS"


def case_ambiguous():
    receipt = {"S0047": 1, "S0032": 1}
    return "확신도 애매", receipt, {
        "cam_left": [det("cam_left", "L1", {"S0047": 0.80, "S0032": 0.2}, 125),
                     det("cam_left", "L2", {"S0032": 0.47, "S0047": 0.2}, 425)],
        "cam_right": [det("cam_right", "R1", {"S0047": 0.79, "S0032": 0.2}, 325),
                      det("cam_right", "R2", {"S0032": 0.46, "S0047": 0.2}, 625)]}, "REVIEW"


CASES = [case_real_bug, case_excess, case_unpaid, case_occluded,
         case_one_cam_blind, case_ambiguous]


def run(per_cam, receipt, fusion):
    obs = V.build_observation(per_cam, fusion, transaction_id="TX-TEST",
                              gate_id="GATE-03",
                              captured_at="2026-08-16T14:23:05+09:00",
                              duration_ms=1000,
                              frames_used={c: 10 for c in per_cam})
    # the boundary is JSON, not Python objects
    obs = json.loads(json.dumps(obs))
    return obs, D.verify(obs, receipt)


def _fusion(name):
    return (V.AsymmetricFusion() if name == "asymmetric"
            else V.PlaneMatchFusion(H, merge_radius_cm=30.0))


@pytest.mark.parametrize("fusion_name", ["asymmetric", "plane_match"])
@pytest.mark.parametrize("case", CASES, ids=[c.__name__ for c in CASES])
def test_boundary_case(case, fusion_name):
    """Same 6 scenarios under both fusion strategies (6/6 each)."""
    name, receipt, per_cam, expect = case()
    obs, verdict = run(per_cam, receipt, _fusion(fusion_name))
    assert obs["cross_camera_resolved"] is (fusion_name == "plane_match")
    assert verdict["verdict"] == expect, f"{name} [{fusion_name}] -> {verdict['verdict']}"


def main():
    strategies = {
        "미캘리브(asymmetric)": V.AsymmetricFusion(),
        "캘리브 후(plane_match)": V.PlaneMatchFusion(H, merge_radius_cm=30.0),
    }
    hdr = f"{'시나리오':<20}{'기대':<8}" + "".join(f"{k:<26}" for k in strategies)
    print(hdr)
    print("-" * len(hdr))
    score = {k: 0 for k in strategies}
    for case in CASES:
        name, receipt, per_cam, expect = case()
        row = f"{name:<20}{expect:<8}"
        for sname, fu in strategies.items():
            _, verdict = run(per_cam, receipt, fu)
            hit = verdict["verdict"] == expect
            score[sname] += int(hit)
            row += f"{'OK ' if hit else 'MISS'} {verdict['verdict']:<21}"
        print(row)
    print("-" * len(hdr))
    print(f"{'합계':<28}" + "".join(f"{score[k]}/{len(CASES)}{'':<22}" for k in strategies))

    print("\n=== 경계 통과 실물: VisionObservation (case1, plane_match) ===")
    name, receipt, per_cam, _ = case_real_bug()
    obs, verdict = run(per_cam, receipt, V.PlaneMatchFusion(H, merge_radius_cm=30.0))
    slim = dict(obs)
    slim["instances"] = [{k: v for k, v in i.items()
                          if k in ("instance_id", "track_ids", "camera_ids",
                                   "plane_xy", "candidates", "stable",
                                   "label_conflict")}
                         for i in obs["instances"]]
    print(json.dumps(slim, ensure_ascii=False, indent=2))
    print("\n=== 판정 결과 GateVerdict ===")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
