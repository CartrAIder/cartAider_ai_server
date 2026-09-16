import asyncio
import csv
import json

import httpx
import numpy as np

from cartgate.server.api import create_app
from cartgate.server.catalog import ProductCatalog
from cartgate.server.service import GateService
from cartgate.server.spring import SpringGateClient


def request(app, **kwargs):
    async def send():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/v1/gate/inspections", **kwargs)

    return asyncio.run(send())


def image():
    import cv2

    ok, encoded = cv2.imencode(".jpg", np.zeros((16, 16, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def files():
    return [("cam_left", ("left.jpg", image(), "image/jpeg")), ("cam_right", ("right.jpg", image(), "image/jpeg"))]


def test_uploaded_inspection_claims_receipt_then_completes_same_token(tmp_path):
    """Catches an HTTP upload that returns PASS without performing the Spring state transitions."""
    catalog_path = tmp_path / "products.csv"
    with catalog_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sku_id", "name", "barcode"])
        writer.writeheader()
        writer.writerow({"sku_id": "S0001", "name": "aloe", "barcode": "0000289908820"})
    calls = []

    def spring(request):
        calls.append((request.url.path, json.loads(request.content)))
        if request.url.path.endswith("/inspections"):
            return httpx.Response(200, json={"items": [{"barcode": "0000289908820", "qty": 1}]})
        return httpx.Response(204)

    client = SpringGateClient("http://spring:8080", "secret", transport=httpx.MockTransport(spring))
    service = GateService(client, ProductCatalog.from_csv(catalog_path), lambda **_: "PASS")
    app = create_app(service=service, readiness=lambda: (True, ["CPUExecutionProvider"]))

    response = request(app, data={"gate_token": "token-1", "gate_id": "GATE-01"}, files=files())

    assert response.status_code == 200
    assert response.json() == {"verdict": "PASS"}
    assert calls == [
        ("/api/internal/gate/inspections", {"gate_token": "token-1", "gate_id": "GATE-01"}),
        ("/api/internal/gate/inspections/complete", {"gate_token": "token-1", "verdict": "PASS"}),
    ]
