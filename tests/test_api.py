import asyncio

import httpx
import numpy as np

from cartgate.server.api import create_app
from cartgate.server.service import InspectionResult


class RecordingService:
    def __init__(self):
        self.request = None

    def inspect(self, request):
        self.request = request
        return InspectionResult(verdict="PASS")


def jpeg_bytes():
    import cv2

    ok, encoded = cv2.imencode(".jpg", np.zeros((16, 16, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def inspection_files():
    image = jpeg_bytes()
    return [
        ("cam_left", ("left-1.jpg", image, "image/jpeg")),
        ("cam_right", ("right-1.jpg", image, "image/jpeg")),
    ]


def request(app, method, path, **kwargs):
    async def send():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_health_separates_liveness_from_model_readiness():
    """Catches a live process being reported ready before models have loaded."""
    app = create_app(service=None, readiness=lambda: (False, ["CPUExecutionProvider"]))

    response = request(app, "GET", "/healthz")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready", "provider": ["CPUExecutionProvider"]}


def test_upload_decodes_both_camera_streams_and_returns_final_verdict():
    """Catches ignoring one camera's uploaded frames or returning an observation instead of a Spring verdict."""
    service = RecordingService()
    app = create_app(service=service, readiness=lambda: (True, ["CUDAExecutionProvider"]))

    response = request(
        app,
        "POST",
        "/v1/gate/inspections",
        data={"gate_token": "token-1", "gate_id": "GATE-01"},
        files=inspection_files(),
    )

    assert response.status_code == 200
    assert response.json() == {"verdict": "PASS"}
    assert set(service.request.frames) == {"cam_left", "cam_right"}
    assert all(len(stream) == 1 for stream in service.request.frames.values())


def test_upload_rejects_a_missing_fixed_camera_stream():
    """Catches accepting a one-camera upload as a valid gate inspection."""
    app = create_app(service=RecordingService(), readiness=lambda: (True, ["CUDAExecutionProvider"]))

    response = request(
        app,
        "POST",
        "/v1/gate/inspections",
        data={"gate_token": "token-1", "gate_id": "GATE-01"},
        files=[("cam_left", ("left-1.jpg", jpeg_bytes(), "image/jpeg"))],
    )

    assert response.status_code == 422
    assert "cam_right" in response.json()["detail"]
