"""Gate-inspection state machine built around the existing Spring contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from cartgate.server.catalog import CatalogError, ProductCatalog, ReceiptItem
from cartgate.server.spring import SpringGateError


class GateSpringClient(Protocol):
    def start(self, gate_token: str, gate_id: str) -> list[dict[str, Any]]: ...
    def complete(self, gate_token: str, verdict: str) -> None: ...
    def fail(self, gate_token: str, failure_reason: str) -> None: ...


class VisionRunner(Protocol):
    def __call__(
        self, *, receipt: dict[str, int], frames: dict[str, list[Any]], transaction_id: str, gate_id: str
    ) -> str: ...


class CameraUnavailableError(RuntimeError):
    pass


class VisionInferenceError(RuntimeError):
    pass


class InvalidImageDataError(ValueError):
    pass


@dataclass(frozen=True)
class InspectionRequest:
    gate_token: str
    gate_id: str
    frames: dict[str, list[Any]]


@dataclass(frozen=True)
class InspectionResult:
    verdict: str


class GateService:
    _CAMERAS = {"cam_left", "cam_right"}
    _VERDICTS = {"PASS", "REVIEW", "FLAG"}

    def __init__(self, spring: GateSpringClient, catalog: ProductCatalog, vision_runner: VisionRunner):
        self._spring = spring
        self._catalog = catalog
        self._vision_runner = vision_runner

    def inspect(self, request: InspectionRequest) -> InspectionResult:
        self._validate_request(request)
        items = self._spring.start(request.gate_token, request.gate_id)
        try:
            receipt = self._catalog.receipt_to_skus([
                ReceiptItem(barcode=str(item.get("barcode", "")), qty=item.get("qty")) for item in items
            ])
            verdict = self._vision_runner(
                receipt=receipt,
                frames=request.frames,
                transaction_id=request.gate_token,
                gate_id=request.gate_id,
            )
            if verdict not in self._VERDICTS:
                raise VisionInferenceError(f"unsupported vision verdict: {verdict}")
            self._spring.complete(request.gate_token, verdict)
            return InspectionResult(verdict=verdict)
        except SpringGateError:
            # Spring may have persisted the completion before its response was
            # lost. Never replace that terminal state with a conflicting fail.
            raise
        except CameraUnavailableError:
            self._spring.fail(request.gate_token, "CAMERA_UNAVAILABLE")
            raise
        except InvalidImageDataError:
            self._spring.fail(request.gate_token, "INVALID_IMAGE_DATA")
            raise
        except VisionInferenceError:
            self._spring.fail(request.gate_token, "AI_INFERENCE_ERROR")
            raise
        except CatalogError:
            self._spring.fail(request.gate_token, "INTERNAL_ERROR")
            raise
        except Exception:
            self._spring.fail(request.gate_token, "INTERNAL_ERROR")
            raise

    def _validate_request(self, request: InspectionRequest) -> None:
        if set(request.frames) != self._CAMERAS:
            raise ValueError("exactly cam_left and cam_right are required")
        if not request.gate_token or not request.gate_id:
            raise ValueError("gate_token and gate_id are required")
        for camera_id, frames in request.frames.items():
            if not frames:
                raise CameraUnavailableError(f"no frames received for {camera_id}")
