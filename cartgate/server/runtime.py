"""Model-backed two-camera CartGate runtime."""

from __future__ import annotations

import datetime as dt
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cartgate import vision_fusion
from cartgate.embed import get_embedder
from cartgate.gallery import load_gallery
from cartgate.vision import load_fusion, resolve_camera
from cartgate.verification import reference_verify
from cartgate.server.service import CameraUnavailableError, InvalidImageDataError, VisionInferenceError


CAMERA_IDS = {"cam_left", "cam_right"}


@dataclass(frozen=True)
class CapturedFrame:
    image: np.ndarray


def validate_two_camera_frames(frames: dict[str, list[CapturedFrame]], min_frames: int) -> None:
    if set(frames) != CAMERA_IDS:
        raise CameraUnavailableError("exactly cam_left and cam_right are required")
    for camera_id, stream in frames.items():
        if len(stream) < min_frames:
            raise CameraUnavailableError(f"insufficient frames for {camera_id}")
        for frame in stream:
            image = frame.image
            if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
                raise InvalidImageDataError(f"{camera_id} frame is not a BGR image")
            if image.shape[0] < 2 or image.shape[1] < 2:
                raise InvalidImageDataError(f"{camera_id} frame is too small")


class VisionRuntime:
    """Loads models once and serializes tracker-backed inference across requests."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: int | str = 0,
        min_frames: int = 2,
        detector: Any | None = None,
        embedder: Any | None = None,
        gallery: dict | None = None,
        fusion: Any | None = None,
    ):
        root = Path(model_dir)
        self._device = device
        self._min_frames = min_frames
        try:
            if detector is None:
                from ultralytics import YOLO
                detector = YOLO(str(root / "runs/detector/best.pt"))
            self._detector = detector
            self._embedder = embedder or get_embedder(str(root / "dino_arc.onnx"), pad=True)
            self._gallery = gallery or load_gallery(str(root / "out/gallery.npz"))
            self._fusion = fusion or load_fusion(str(root / "gate_calib.json"))
        except Exception as exc:
            raise VisionInferenceError("unable to initialize CartGate models") from exc
        self._lock = threading.Lock()

    @property
    def provider(self) -> list[str]:
        return list(getattr(self._embedder, "providers", ["classical"]))

    def __call__(
        self, *, receipt: dict[str, int], frames: dict[str, list[CapturedFrame]], transaction_id: str, gate_id: str
    ) -> str:
        validate_two_camera_frames(frames, self._min_frames)
        started = time.perf_counter()
        try:
            with self._lock:
                per_camera, crop_store = {}, {}
                for camera_id in ("cam_left", "cam_right"):
                    detections, crops = resolve_camera(
                        self._detector,
                        frames[camera_id],
                        self._embedder,
                        self._gallery,
                        list(receipt),
                        self._device,
                        camera_id=camera_id,
                    )
                    per_camera[camera_id] = detections
                    crop_store.update(crops)
                observation = vision_fusion.build_observation(
                    per_camera,
                    self._fusion,
                    transaction_id=transaction_id,
                    gate_id=gate_id,
                    captured_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    frames_used={camera_id: len(stream) for camera_id, stream in frames.items()},
                )
                verdict = reference_verify.verify(observation, receipt)
                return str(verdict["verdict"])
        except (CameraUnavailableError, InvalidImageDataError, VisionInferenceError):
            raise
        except Exception as exc:
            raise VisionInferenceError("CartGate inference failed") from exc
