import json

import httpx

from cartgate.server.spring import SpringGateClient


def test_start_sends_spring_contract_and_parses_receipt():
    """Catches a wrong Spring route, header, snake_case payload, or receipt shape."""
    def spring(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/internal/gate/inspections"
        assert request.headers["X-AI-Service-Secret"] == "shared-secret"
        assert json.loads(request.content) == {"gate_token": "token-1", "gate_id": "GATE-01"}
        return httpx.Response(200, json={"items": [{"barcode": "0000289908820", "qty": 2}]})

    client = SpringGateClient("http://spring:8080", "shared-secret", transport=httpx.MockTransport(spring))

    assert client.start("token-1", "GATE-01") == [{"barcode": "0000289908820", "qty": 2}]


def test_complete_sends_final_verdict_to_spring():
    """Catches completing an inspection with an observation instead of Spring's final verdict payload."""
    def spring(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/internal/gate/inspections/complete"
        assert json.loads(request.content) == {"gate_token": "token-1", "verdict": "REVIEW"}
        return httpx.Response(204)

    client = SpringGateClient("http://spring:8080", "shared-secret", transport=httpx.MockTransport(spring))

    client.complete("token-1", "REVIEW")


def test_fail_sends_only_a_spring_failure_enum():
    """Catches reporting a local error name Spring cannot deserialize."""
    def spring(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/internal/gate/inspections/fail"
        assert json.loads(request.content) == {"gate_token": "token-1", "failure_reason": "CAMERA_UNAVAILABLE"}
        return httpx.Response(204)

    client = SpringGateClient("http://spring:8080", "shared-secret", transport=httpx.MockTransport(spring))

    client.fail("token-1", "CAMERA_UNAVAILABLE")
