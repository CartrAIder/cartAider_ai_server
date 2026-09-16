# AI Server Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deployable AI HTTP service that receives exactly two camera streams, obtains the paid receipt from Spring, runs the CartGate pipeline, and reports a final Spring gate state.

**Architecture:** The FastAPI layer owns input validation, bounded request serialization, and HTTP responses. A service layer owns the Spring start/complete/fail state machine and delegates vision work to a runtime initialized once per process. A barcode catalog maps Spring receipt lines to CartGate SKUs; vision and the reference decision implementation remain reusable library components.

**Tech Stack:** Python 3.11, FastAPI, Uvicorn, HTTPX, Pydantic, Ultralytics, ONNX Runtime, pytest.

**Spec:** `AI_SERVER_IMPLEMENTATION_REQUIREMENTS.md` in the parent workspace; Spring contract in `../quickPass/src/main/java/com/mart/quickpass/gate/`.

## Global Constraints

- Preserve the existing Spring contract and use `X-AI-Service-Secret` only for AI-to-Spring calls.
- Accept exactly `cam_left` and `cam_right`; no third camera path exists.
- Do not treat absent frames, camera failure, invalid images, or failed inference as PASS.
- Use prebuilt `out/gallery.pkl`; do not build a gallery while serving.
- Serialize access to the shared Ultralytics tracker until per-request detector instances are proven safe.
- Read barcodes as strings so leading zeros are preserved.

---

### Task 1: Runtime configuration and catalog validation

**Files:**
- Create: `cartgate/server/settings.py`
- Create: `cartgate/server/catalog.py`
- Test: `tests/test_server_catalog.py`

**Interfaces:**
- Produces `Settings.from_env() -> Settings` and `ProductCatalog.from_csv(path) -> ProductCatalog`.
- `ProductCatalog.receipt_to_skus(items) -> dict[str, int]` raises `CatalogError` for unknown barcode or non-positive quantity.

- [ ] Write failing tests for BOM CSV parsing, leading-zero barcode preservation, and unknown barcode rejection.
- [ ] Run `pytest tests/test_server_catalog.py -q`; expect import failure before implementation.
- [ ] Implement immutable settings and catalog mapping without loading models.
- [ ] Run the catalog tests and commit the green change.

### Task 2: Spring gate client

**Files:**
- Create: `cartgate/server/spring.py`
- Test: `tests/test_spring_client.py`

**Interfaces:**
- Produces `SpringGateClient.start(token, gate_id) -> list[ReceiptItem]`, `complete(token, verdict)`, and `fail(token, reason)`.
- Sends snake_case payloads and the `X-AI-Service-Secret` header.

- [ ] Write failing transport-level tests for start response parsing and complete/fail request fields.
- [ ] Run targeted tests and confirm they fail because `SpringGateClient` is absent.
- [ ] Implement a small HTTPX client with explicit connect/read timeouts and retry only for transport errors and 5xx responses.
- [ ] Run client tests and commit the green change.

### Task 3: Two-camera request validation and orchestration state machine

**Files:**
- Create: `cartgate/server/service.py`
- Test: `tests/test_gate_service.py`

**Interfaces:**
- Consumes an injected `SpringGateClient`, `ProductCatalog`, and `VisionRunner` protocol.
- Produces `GateService.inspect(request) -> InspectionResult` and maps local failures to Spring failure enums.

- [ ] Write failing tests proving exactly two named cameras are required and invalid frames cause `INVALID_IMAGE_DATA` after a successful start.
- [ ] Write failing tests proving a camera capture failure causes `CAMERA_UNAVAILABLE`, and a local inference error causes `AI_INFERENCE_ERROR`.
- [ ] Implement the minimal state machine so only a successfully started inspection can be completed or failed.
- [ ] Run service tests and commit the green change.

### Task 4: Vision runtime adapter and safe decision behavior

**Files:**
- Create: `cartgate/server/runtime.py`
- Modify: `cartgate/verification/reference_verify.py`
- Test: `tests/test_runtime.py`
- Test: `tests/test_boundary.py`

**Interfaces:**
- Produces `VisionRunner.inspect(receipt, frames, transaction_id, gate_id) -> dict`.
- Loads detector, ONNX embedder, gallery, and two-camera calibration once.

- [ ] Write failing tests for a missing camera, zero frames, and failed vision status never producing a completion verdict.
- [ ] Write a failing regression test that keeps an empty but healthy camera in conservative count calculations.
- [ ] Implement runtime initialization checks and the per-process inference lock; separate operational failures from a normal empty detection set.
- [ ] Update conservative camera handling and run all boundary/runtime tests.
- [ ] Commit the green change.

### Task 5: FastAPI application and deployment artifacts

**Files:**
- Create: `cartgate/server/api.py`
- Create: `Dockerfile`
- Create: `.dockerignore`
- Modify: `requirements.txt`
- Modify: `pyproject.toml`
- Test: `tests/test_api.py`

**Interfaces:**
- Exposes `GET /healthz` and `POST /v1/gate/inspections` on port 8000.
- Health returns readiness separately from liveness and the actual model provider.

- [ ] Write failing API tests for readiness, request validation, and a successful final verdict response.
- [ ] Implement the FastAPI dependency wiring using the Task 3 service.
- [ ] Add runtime-only dependencies, package discovery for `cartgate.verification`, and pytest test paths.
- [ ] Add a non-root Docker image that starts Uvicorn on `0.0.0.0:8000`.
- [ ] Run API and package tests, build the Docker image when dependencies/platform allow, then commit.

### Task 6: End-to-end local contract test and deployment documentation

**Files:**
- Create: `tests/test_end_to_end_contract.py`
- Create: `DEPLOYMENT.md`
- Modify: `README.md`

**Interfaces:**
- Verifies gate start → result → complete and start → operational failure → fail against a local Spring-contract stub.

- [ ] Write failing end-to-end tests with an in-process HTTPX mock transport that records Spring requests.
- [ ] Implement only integration wiring required to pass those tests.
- [ ] Document environment variables, model bundle layout, Docker network alias `fastapi`, and Nginx/Spring health checks.
- [ ] Run the full test suite and commit.

## Verification

- `python -m pytest tests -q -p no:cacheprovider`
- Build and install the wheel in a clean Python 3.11 environment, then import `cartgate.verification`.
- Build the AI Docker image for the target Jetson architecture and run it on `cartAider-network` with the `fastapi` alias.
- Exercise a real Spring test token through start, completion, and operational failure flows before production enablement.
