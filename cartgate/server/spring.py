"""Client for the existing Spring internal gate-inspection contract."""

from __future__ import annotations

from typing import Any

import httpx


class SpringGateError(RuntimeError):
    """Spring rejected a gate state transition or returned an invalid response."""


class SpringGateClient:
    def __init__(
        self,
        base_url: str,
        service_secret: str,
        *,
        timeout_seconds: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"X-AI-Service-Secret": service_secret},
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def start(self, gate_token: str, gate_id: str) -> list[dict[str, Any]]:
        response = self._request("POST", "/api/internal/gate/inspections", {
            "gate_token": gate_token,
            "gate_id": gate_id,
        })
        try:
            payload = response.json()
            items = payload["items"]
        except (TypeError, KeyError, ValueError) as exc:
            raise SpringGateError("Spring start response does not contain items") from exc
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise SpringGateError("Spring start response items must be a list")
        return items

    def complete(self, gate_token: str, verdict: str) -> None:
        self._request("POST", "/api/internal/gate/inspections/complete", {
            "gate_token": gate_token,
            "verdict": verdict,
        })

    def fail(self, gate_token: str, failure_reason: str) -> None:
        self._request("POST", "/api/internal/gate/inspections/fail", {
            "gate_token": gate_token,
            "failure_reason": failure_reason,
        })

    def _request(self, method: str, path: str, payload: dict[str, str]) -> httpx.Response:
        try:
            response = self._client.request(method, path, json=payload)
            response.raise_for_status()
            return response
        except httpx.HTTPError as exc:
            raise SpringGateError(f"Spring gate request failed: {method} {path}") from exc
