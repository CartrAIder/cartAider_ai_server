"""Configuration read once at service startup."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    spring_base_url: str
    spring_service_secret: str
    request_timeout_seconds: float = 5.0
    min_frames_per_camera: int = 2

    @classmethod
    def from_env(cls) -> "Settings":
        model_dir = Path(os.environ.get("CARTGATE_MODEL_DIR", "."))
        spring_base_url = os.environ.get("SPRING_BASE_URL", "http://spring:8080").rstrip("/")
        return cls(
            model_dir=model_dir,
            spring_base_url=spring_base_url,
            spring_service_secret=os.environ.get("GATE_SERVICE_SECRET", ""),
            request_timeout_seconds=float(os.environ.get("SPRING_REQUEST_TIMEOUT_SECONDS", "5")),
            min_frames_per_camera=int(os.environ.get("MIN_FRAMES_PER_CAMERA", "2")),
        )
