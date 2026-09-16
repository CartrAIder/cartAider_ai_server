from dataclasses import dataclass, field

import pytest

from cartgate.server.catalog import ProductCatalog
from cartgate.server.service import CameraUnavailableError, GateService, InspectionRequest, VisionInferenceError
from cartgate.server.spring import SpringGateError


@dataclass
class RecordingSpring:
    receipt: list[dict]
    calls: list[tuple] = field(default_factory=list)

    def start(self, gate_token, gate_id):
        self.calls.append(("start", gate_token, gate_id))
        return self.receipt

    def complete(self, gate_token, verdict):
        self.calls.append(("complete", gate_token, verdict))

    def fail(self, gate_token, reason):
        self.calls.append(("fail", gate_token, reason))


def catalog(tmp_path):
    csv_path = tmp_path / "products.csv"
    csv_path.write_text("sku_id,name,barcode\nS0001,aloe,0000289908820\n", encoding="utf-8")
    return ProductCatalog.from_csv(csv_path)


def request():
    return InspectionRequest(
        gate_token="token-1",
        gate_id="GATE-01",
        frames={"cam_left": [object(), object()], "cam_right": [object(), object()]},
    )


def test_inspection_gets_receipt_runs_vision_and_completes(tmp_path):
    """Catches bypassing Spring's receipt or omitting the terminal complete transition."""
    spring = RecordingSpring([{"barcode": "0000289908820", "qty": 1}])

    result = GateService(spring, catalog(tmp_path), lambda **_: "PASS").inspect(request())

    assert result.verdict == "PASS"
    assert spring.calls == [("start", "token-1", "GATE-01"), ("complete", "token-1", "PASS")]


def test_camera_failure_after_start_is_reported_to_spring(tmp_path):
    """Catches a captured-camera failure being reported as a normal PASS or left IN_PROGRESS."""
    spring = RecordingSpring([{"barcode": "0000289908820", "qty": 1}])

    def unavailable(**_):
        raise CameraUnavailableError("cam_right")

    with pytest.raises(CameraUnavailableError):
        GateService(spring, catalog(tmp_path), unavailable).inspect(request())

    assert spring.calls == [
        ("start", "token-1", "GATE-01"),
        ("fail", "token-1", "CAMERA_UNAVAILABLE"),
    ]


def test_inference_error_after_start_is_reported_to_spring(tmp_path):
    """Catches local inference errors leaving the Spring token in IN_PROGRESS."""
    spring = RecordingSpring([{"barcode": "0000289908820", "qty": 1}])

    def broken_model(**_):
        raise VisionInferenceError("ONNX failed")

    with pytest.raises(VisionInferenceError):
        GateService(spring, catalog(tmp_path), broken_model).inspect(request())

    assert spring.calls == [
        ("start", "token-1", "GATE-01"),
        ("fail", "token-1", "AI_INFERENCE_ERROR"),
    ]


def test_rejects_any_camera_set_other_than_the_fixed_pair(tmp_path):
    """Catches accepting a missing or third camera despite the two-camera gate contract."""
    spring = RecordingSpring([])
    invalid = InspectionRequest(
        gate_token="token-1", gate_id="GATE-01", frames={"cam_left": [object()], "cam_extra": [object()]}
    )

    with pytest.raises(ValueError, match="cam_left and cam_right"):
        GateService(spring, catalog(tmp_path), lambda **_: "PASS").inspect(invalid)

    assert spring.calls == []


def test_completion_transport_failure_does_not_replace_result_with_fail(tmp_path):
    """Catches changing a possibly persisted Spring completion into FAILED after response loss."""
    class CompletionLostSpring(RecordingSpring):
        def complete(self, gate_token, verdict):
            self.calls.append(("complete", gate_token, verdict))
            raise SpringGateError("response lost")

    spring = CompletionLostSpring([{"barcode": "0000289908820", "qty": 1}])

    with pytest.raises(SpringGateError):
        GateService(spring, catalog(tmp_path), lambda **_: "PASS").inspect(request())

    assert spring.calls == [("start", "token-1", "GATE-01"), ("complete", "token-1", "PASS")]
