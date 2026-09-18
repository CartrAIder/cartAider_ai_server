"""FastAPI entry point for uploaded two-camera gate inspections."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import anyio
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from typing_extensions import Annotated

from cartgate.server.catalog import ProductCatalog
from cartgate.server.runtime import CapturedFrame, VisionRuntime
from cartgate.server.service import (
    CameraUnavailableError,
    GateService,
    InspectionRequest,
    InvalidImageDataError,
    VisionInferenceError,
)
from cartgate.server.settings import Settings
from cartgate.server.spring import SpringGateClient

MAX_IMAGE_BYTES = 5 * 1024 * 1024


def create_app(
    *,
    service: Optional[Any] = None,
    readiness: Optional[Callable[[], Tuple[bool, List[str]]]] = None,
) -> FastAPI:
    app = FastAPI(title="CartGate AI Server", version="0.1.0")
    ready = readiness or (lambda: (False, []))

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        is_ready, providers = ready()
        body = {"status": "ready" if is_ready else "not_ready", "provider": providers}
        return JSONResponse(body, status_code=200 if is_ready else 503)

    @app.post("/v1/gate/inspections")
    async def inspect(
        raw_request: Request,
        gate_token: Annotated[str, Form()],
        gate_id: Annotated[str, Form()],
        cam_left: Annotated[Optional[List[UploadFile]], File()] = None,
        cam_right: Annotated[Optional[List[UploadFile]], File()] = None,
    ) -> Dict[str, str]:
        if service is None:
            raise HTTPException(status_code=503, detail="AI service is not ready")
        if not cam_left:
            raise HTTPException(status_code=422, detail="cam_left frames are required")
        if not cam_right:
            raise HTTPException(status_code=422, detail="cam_right frames are required")
        form = await raw_request.form()
        unknown_parts = set(form.keys()) - {"gate_token", "gate_id", "cam_left", "cam_right"}
        if unknown_parts:
            raise HTTPException(status_code=422, detail="unknown multipart field")
        frames = {
            "cam_left": [await _decode_upload(upload, "cam_left") for upload in cam_left],
            "cam_right": [await _decode_upload(upload, "cam_right") for upload in cam_right],
        }
        request = InspectionRequest(gate_token=gate_token, gate_id=gate_id, frames=frames)
        try:
            result = await anyio.to_thread.run_sync(service.inspect, request)
        except CameraUnavailableError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except InvalidImageDataError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except VisionInferenceError as exc:
            raise HTTPException(status_code=502, detail="AI inference failed") from exc
        return {"verdict": result.verdict}

    return app


async def _decode_upload(upload: UploadFile, camera_id: str) -> CapturedFrame:
    raw = await upload.read(MAX_IMAGE_BYTES + 1)
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=422, detail=f"{camera_id} image is empty or too large")
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=422, detail=f"{camera_id} image cannot be decoded")
    return CapturedFrame(decoded)


def create_production_app() -> FastAPI:
    settings = Settings.from_env()
    runtime = VisionRuntime(settings.model_dir, min_frames=settings.min_frames_per_camera)
    spring = SpringGateClient(
        settings.spring_base_url,
        settings.spring_service_secret,
        timeout_seconds=settings.request_timeout_seconds,
    )
    catalog = ProductCatalog.from_csv(settings.model_dir / "products.csv")
    service = GateService(spring, catalog, runtime)
    return create_app(service=service, readiness=lambda: (True, runtime.provider))


app = create_app()
