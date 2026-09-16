# AI Server Deployment

The AI container joins the already-running `cartAider-network`. Nginx resolves
the container through the `fastapi` network alias and proxies the configured AI
hostname to port 8000.

## Required model directory

Mount a read-only directory as `/models`. It must contain:

```text
/models/
├── dino_arc.onnx
├── products.csv
├── gate_calib.json                 # measured calibration for this physical gate
├── out/gallery.pkl
└── runs/detector/best.pt
```

`sample_gate_calib.json` is not a production calibration. A missing
`gate_calib.json` makes the runtime use asymmetric fusion; decide whether that
mode is acceptable before deploying it at a gate.

## Required environment

```dotenv
GATE_SERVICE_SECRET=<same value as Spring GATE_SERVICE_SECRET>
SPRING_BASE_URL=http://spring:8080
CARTGATE_MODEL_DIR=/models
SPRING_REQUEST_TIMEOUT_SECONDS=5
MIN_FRAMES_PER_CAMERA=2
```

The gate device sends `POST /v1/gate/inspections` as `multipart/form-data`:

- `gate_token`: Spring payment response token
- `gate_id`: configured physical gate ID
- one or more `cam_left` image files
- one or more `cam_right` image files

The server only accepts the two fixed camera names. It returns
`{"verdict":"PASS"}`, `{"verdict":"REVIEW"}`, or `{"verdict":"FLAG"}`
after Spring confirms the corresponding terminal state. It does not expose the
Spring service secret to the gate device.

## Build and run

For a CPU local test image:

```bash
docker build -t cartgate-ai:local .
docker run --rm --name cartgate-ai \
  --network cartAider-network \
  --network-alias fastapi \
  --env-file /secure/cartgate-ai.env \
  --mount type=bind,src=/absolute/path/to/models,dst=/models,readonly \
  cartgate-ai:local
```

For Jetson, build on the target or use a Jetson-compatible Python/CUDA base
image through `--build-arg BASE_IMAGE=<jetson-image>`. Install the matching
Jetson PyTorch and ONNX Runtime GPU packages in that image. Before allowing
traffic, call `GET /healthz` through Nginx and confirm the reported provider
contains `CUDAExecutionProvider`; CPU fallback is not a production-ready GPU
deployment.

## Operational checks

1. Confirm `GET /healthz` returns HTTP 200 and the expected provider.
2. Submit a test payment token and two camera streams; confirm Spring records
   the final `USED` token state.
3. Submit an invalid image or simulate a camera/inference failure; confirm
   Spring records `FAILED` with an allowed failure reason.
4. Confirm a retry with the same terminal result is accepted by Spring and a
   conflicting terminal result is rejected.

The container does not publish a host port. Nginx and Spring communicate over
the shared Docker network.
