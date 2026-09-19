"""Configuration read once at service startup."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    spring_base_url: str
    spring_service_secret: str
    gate_api_key_hashes: Dict[str, str]
    request_timeout_seconds: float = 5.0
    min_frames_per_camera: int = 2

    @classmethod
    def from_env(cls) -> "Settings":
        model_dir = Path(os.environ.get("CARTGATE_MODEL_DIR", "."))
        spring_base_url = os.environ.get("SPRING_BASE_URL", "http://spring:8080").rstrip("/")
        gate_api_key_hashes = _load_gate_api_key_hashes(os.environ.get("GATE_API_KEY_HASHES", ""))
        return cls(
            model_dir=model_dir,
            spring_base_url=spring_base_url,
            spring_service_secret=os.environ.get("GATE_SERVICE_SECRET", ""),
            gate_api_key_hashes=gate_api_key_hashes,
            request_timeout_seconds=float(os.environ.get("SPRING_REQUEST_TIMEOUT_SECONDS", "5")),
            min_frames_per_camera=int(os.environ.get("MIN_FRAMES_PER_CAMERA", "2")),
        )


def _load_gate_api_key_hashes(raw: str) -> Dict[str, str]:
    if not raw:
        raise ValueError("GATE_API_KEY_HASHES is required")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("GATE_API_KEY_HASHES must be a JSON object") from exc
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError("GATE_API_KEY_HASHES must be a non-empty JSON object")
    normalized: Dict[str, str] = {}
    for gate_id, key_hash in parsed.items():
        if not isinstance(gate_id, str) or not gate_id:
            raise ValueError("GATE_API_KEY_HASHES contains an invalid gate_id")
        if (
            not isinstance(key_hash, str)
            or len(key_hash) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in key_hash)
        ):
            raise ValueError("GATE_API_KEY_HASHES contains an invalid SHA-256 hash")
        normalized[gate_id] = key_hash.lower()
    return normalized
