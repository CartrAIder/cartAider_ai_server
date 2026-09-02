# 백엔드 연동 가이드 — 무엇을 배포하고, 어떤 코드가 무슨 일을 하는가

대상: AI 모델을 서버에 올려 서빙할 담당자.
인터페이스 규격 자체는 [`CONTRACT_v1.1.md`](CONTRACT_v1.1.md)가 기준 문서이고, 이 문서는
**"그래서 무슨 파일을 올리고 무슨 코드를 부르면 되는가"**만 다룹니다.

---

## 1. 배포에 필요한 것 (모델 번들)

`cartgate_models.zip`을 리포 루트에 풀면 아래 경로가 됩니다.

| 파일 | 크기 | 역할 |
|---|---|---|
| `dino_arc.onnx` | 84 MB | **인식 임베더.** 크롭 1장 → 256-d 벡터 |
| `runs/detector/best.pt` | 5.2 MB | **검출기** (YOLO11n, 단일 클래스 `product`). ultralytics로 로드 |
| `deploy/product_det_640.onnx` | 10.4 MB | 같은 검출기의 ONNX. ultralytics 의존을 피하고 싶을 때 |
| `deploy/product_det_512.onnx` | 10.1 MB | 위와 동일, 입력 512 (엣지/저사양) |
| `out/gallery.pkl` | 2.4 MB | **상품 갤러리** — 51 SKU × 총 2,310개 임베딩 |
| `out/gallery_meta.json` | 2 KB | SKU별 벡터 개수 (점검용) |
| `products.csv` | 1.5 KB | sku_id · 상품명 · EAN-13 바코드 |
| `sample_gate_calib.json` | 1 KB | 캘리브레이션 **예시** (합성 게이트 기하). 실제 게이트 값 아님 |

`gallery.pkl`이 핵심입니다. 이게 있으면 **상품 사진 원본(`dataset/`, 수백 MB)이 서버에 없어도**
인식이 됩니다. 상품 사진이나 임베딩 모델이 바뀔 때만 다시 만들면 됩니다.

### 모델 I/O 규격

```
dino_arc.onnx      입력  "input"   float32 [b, 3, 224, 224]   ImageNet mean/std 정규화, RGB
                   출력  "emb"     float32 [b, 256]           L2 정규화됨 → 내적 = 코사인 유사도
                   opset 17 · 배치축 동적 (B=8이 가장 효율적)

product_det_*.onnx 입력  "images"  float32 [1, 3, S, S]       S=640 또는 512, 0~1 스케일
                   출력  "output0" float32 [1, 5, N]          (cx, cy, w, h, score) — NMS 미포함!
                   opset 12 · 배치 1 고정 · 클래스 1개(product)
```

ONNX 검출기를 쓰면 **NMS를 직접 구현해야 합니다.** 그게 부담이면 `best.pt` + ultralytics를
쓰는 편이 빠릅니다(`model.track()`이 ByteTrack까지 같이 해결해 줍니다).

---

## 2. FastAPI(또는 무엇이든)에 붙이는 지점

AI 쪽이 제공하는 건 **함수 4개**입니다. 서비스 형태·라우팅·인증·큐는 백엔드 설계 영역이라
아래는 "어디에 무엇을 끼우는가"만 보여주는 최소 예시입니다.

| 언제 | 호출 | 파일 |
|---|---|---|
| 프로세스 기동 시 1회 | `YOLO(...)`, `get_embedder(...)`, `load_gallery(...)`, `load_fusion(...)` | `cartgate/embed.py`, `gallery.py`, `vision.py` |
| 요청마다 · 카메라별 | `resolve_camera(...)` → `Detection` 리스트 | `cartgate/vision.py` |
| 요청마다 · 마지막 1회 | `build_observation(...)` → 응답 JSON | `cartgate/vision_fusion.py` |

```python
from ultralytics import YOLO
from cartgate import vision_fusion
from cartgate.embed import get_embedder
from cartgate.gallery import load_gallery
from cartgate.vision import load_fusion, resolve_camera

# ── 기동 시 1회 (전역 보관) ─────────────────────────────────────────
DETECTOR = YOLO("runs/detector/best.pt")            # 번들의 검출기
EMBEDDER = get_embedder("dino_arc.onnx", pad=True)  # 번들의 인식 모델. pad=True 필수(ViT)
GALLERY  = load_gallery("out/gallery.pkl")          # 번들의 갤러리 51 SKU
FUSION   = load_fusion("gate_calib.json")           # 없으면 자동으로 AsymmetricFusion
assert "CUDAExecutionProvider" in EMBEDDER.providers   # CPU 폴백이면 6배 느림

# ── 요청 처리 ──────────────────────────────────────────────────────
def observe(transaction_id, receipt_skus, cameras):
    """cameras = {"cam_left": [BGR ndarray, ...], "cam_right": [...]}  (촬영 순서대로)"""
    from types import SimpleNamespace
    per_cam = {}
    for cam_id, images in cameras.items():
        frames = [SimpleNamespace(image=img) for img in images]   # .image 만 있으면 됨
        dets, crops = resolve_camera(DETECTOR, frames, EMBEDDER, GALLERY,
                                     receipt_skus, dev=0, camera_id=cam_id)
        per_cam[cam_id] = dets
    return vision_fusion.build_observation(          # 순수 dict → 그대로 JSON 직렬화 가능
        per_cam, FUSION,
        transaction_id=transaction_id, gate_id="GATE-03",
        captured_at="2026-09-02T10:00:00+09:00",     # ISO8601
        duration_ms=123,                             # 측정값을 넣어 주세요
        frames_used={c: len(v) for c, v in cameras.items()})
```

FastAPI라면 위 `observe()`를 라우트 하나에서 부르면 끝입니다. 이미지를 multipart로 받든
base64로 받든 Redis에서 꺼내든, `cv2.imdecode`로 **BGR ndarray**만 만들어 넘기면 됩니다.

```python
# 예: 이미지 바이트 → 입력 형태
import cv2, numpy as np
img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
```

**인자 규칙 3가지**

1. `receipt_skus`에는 **그 카트 영수증의 SKU만** 넣습니다(전체 카탈로그 아님). 후보를 좁히는 게
   설계의 핵심입니다. 유효한 SKU 목록은 `sorted(GALLERY.keys())`로 확인할 수 있습니다.
2. **카메라당 프레임 2장 이상.** 1장이면 모든 인스턴스가 `stable:false`가 되어 판정이
   아무것도 세지 못합니다(`min_frames=2`). 4장 전후가 적당합니다.
3. `dev`는 GPU 인덱스(`0`) 또는 `"cpu"`. 첫 추론은 CUDA 워밍업으로 1.2~1.5초 걸리니
   기동 직후 더미 프레임으로 한 번 돌려 두면 첫 요청이 느려지지 않습니다.

반환값은 계약 §3의 `VisionObservation`이고 **verdict는 없습니다** — PASS/FLAG/REVIEW 판정은
`cartgate/verification/reference_verify.py:verify(observation, receipt)`가 담당합니다.

## 3. 입출력 규격 (2026-09-02 실측)

### 입력 — 지금은 "JSON"이 아니라 **디코딩된 이미지 배열**입니다

이 리포는 HTTP 서비스를 포함하지 않으므로 **와이어 포맷(JSON/멀티파트/바이트)은 정해져 있지
않습니다.** 백엔드가 무엇으로 받든, `resolve_camera()`에 넘길 때는 아래 형태여야 합니다.

| 인자 | 타입 | 실측/제약 |
|---|---|---|
| `frames` | `.image` 속성을 가진 객체의 리스트 | `image`는 **`numpy.ndarray (H, W, 3) uint8, BGR, 0~255`** |
| `receipt_skus` | `list[str]` | 그 카트 영수증 SKU만. 예 `["S0010","S0026","S0040"]` |
| `camera_id` | `str` | `"cam_left"` / `"cam_right"` — 관측 JSON에 그대로 실림 |
| `dev` | `int` 또는 `str` | GPU 인덱스(`0`) 또는 `"cpu"` |
| `model`,`embedder`,`gallery` | 기동 시 로드한 객체 | 요청마다 재로드 금지 |

바이트 → 배열 변환은 한 줄입니다. 프레임은 **촬영 순서대로** 넣어야 추적이 맞습니다.

```python
img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)  # BGR ndarray
frames = [SimpleNamespace(image=img) for img in images]
```

**크기 실측** (합성 640×640 JPEG 기준): 프레임 1장 **73 KB**, base64로 감싸면 **97 KB**.
카트 1대(2캠 × 4프레임) = 8장 **약 600 KB**, base64면 **약 800 KB**.
와이어 포맷은 백엔드 판단이지만, 참고로 base64는 용량이 33% 늘고 JSON 파싱 비용이 붙습니다.

**해상도**: 검출기는 내부에서 640×640으로 letterbox 하므로 원본 크기는 자유입니다. 다만
학습·측정이 640×640에서 이뤄졌으니 그 근처를 권장합니다.

### 중간 산출물 — `resolve_camera()` 반환 (참고용, 그대로 응답하지 않음)

`(list[Detection], dict[str, ndarray])` 튜플입니다.

```
Detection.camera_id   str    'cam_left'
Detection.track_id    str    'L1'
Detection.candidates  dict   {'S0010': 0.3175, 'S0026': 0.5249, 'S0040': 0.2582}   # 영수증 SKU별 유사도
Detection.n_frames    int    4
Detection.box         tuple  (161, 20, 227, 90)      # xyxy, 픽셀
Detection.det_conf    float  0.891
Detection.crop_ref    str|None
두 번째 값 crops: {track_id: BGR ndarray}   # 증거 이미지, 저장은 선택
```

### 출력 — `build_observation()` 반환: **순수 dict → 그대로 JSON**

numpy 타입이 남지 않아 `json.dumps(obs)`가 바로 됩니다(검증 완료). 카트 1대 응답 **약 1.8 KB**.

| 최상위 필드 | 타입 | 값 |
|---|---|---|
| `schema_version` | str | `"1.1"` 고정 |
| `transaction_id` | str | 요청에서 받은 값 그대로 |
| `gate_id` | str | 요청에서 받은 값 그대로 |
| `captured_at` | str | ISO8601 (호출자가 넣어줌) |
| `duration_ms` | int | 호출자가 측정해 넣음 |
| `vision_status` | str | `"OK"` / 실패 시 호출자가 `"FAILED"` |
| `cameras` | list | `[{camera_id: str, status: str, frames_used: int}]` |
| `fusion_strategy` | str | `"plane_match"`(캘리브 O) / `"asymmetric"`(X) |
| `cross_camera_resolved` | bool | 카메라 간 중복 제거 여부 — **판정 모드를 결정** |
| `min_frames` | int | `2` |
| `instances` | list | 아래 |

| `instances[]` 필드 | 타입 | 예 |
|---|---|---|
| `instance_id` | str | `"I1"` |
| `track_ids` | list[str] | `["L1","R1"]` (2캠 병합 시 2개) |
| `camera_ids` | list[str] | `["cam_left","cam_right"]` |
| `plane_xy` | list[float] \| null | `[29.85, 12.09]` cm — 미캘리브면 `null` |
| `candidates` | dict[str,float] | `{"S0010":0.3175,"S0026":0.5249,"S0040":0.2582}` 0~1 |
| `n_frames` | int | `4` |
| `stable` | bool | `n_frames >= min_frames` |
| `label_conflict` | bool | 두 카메라가 top SKU를 다르게 봄 |
| `boxes` | dict[str,list[float]] | `{"cam_left":[161,20,227,90]}` xyxy |
| `crop_refs` | dict[str,str] | 증거 이미지 경로. 저장 안 했으면 `{}` |

**응답에 없는 것**: verdict, band, best_sku. 판정은 `reference_verify.verify(obs, receipt)` 몫입니다.

<details><summary>실제 응답 전문 (2캠 × 4프레임, 인스턴스 4개)</summary>

```json
{
  "schema_version": "1.1",
  "transaction_id": "TX-IO-SPEC",
  "gate_id": "GATE-03",
  "captured_at": "2026-09-02T11:00:00+09:00",
  "duration_ms": 142,
  "vision_status": "OK",
  "cameras": [
    {
      "camera_id": "cam_left",
      "status": "OK",
      "frames_used": 4
    },
    {
      "camera_id": "cam_right",
      "status": "OK",
      "frames_used": 4
    }
  ],
  "fusion_strategy": "plane_match",
  "cross_camera_resolved": true,
  "min_frames": 2,
  "instances": [
    {
      "instance_id": "I1",
      "track_ids": [
        "L1",
        "R1"
      ],
      "camera_ids": [
        "cam_left",
        "cam_right"
      ],
      "plane_xy": [
        29.85,
        12.09
      ],
      "candidates": {
        "S0010": 0.3175,
        "S0026": 0.5249,
        "S0040": 0.2582
      },
      "n_frames": 4,
      "stable": true,
      "label_conflict": false,
      "boxes": {
        "cam_left": [
          161.0,
          20.0,
          227.0,
          90.0
        ],
        "cam_right": [
          260.0,
          21.0,
          335.0,
          88.0
        ]
      },
      "crop_refs": {}
    },
    {
      "instance_id": "I2",
      "track_ids": [
        "L2",
        "R2"
      ],
      "camera_ids": [
        "cam_left",
        "cam_right"
      ],
      "plane_xy": [
        60.64,
        43.79
      ],
      "candidates": {
        "S0010": 0.3169,
        "S0026": 0.3386,
        "S0040": 0.2246
      },
      "n_frames": 4,
      "stable": true,
      "label_conflict": true,
      "boxes": {
        "cam_left": [
          336.0,
          294.0,
          449.0,
          405.0
        ],
        "cam_right": [
          388.0,
          293.0,
          493.0,
          406.0
        ]
      },
      "crop_refs": {}
    },
    {
      "instance_id": "I3",
      "track_ids": [
        "L3",
        "R3"
      ],
      "camera_ids": [
        "cam_left",
        "cam_right"
      ],
      "plane_xy": [
        26.98,
        33.01
      ],
      "candidates": {
        "S0010": 0.2079,
        "S0026": 0.4353,
        "S0040": 0.3598
      },
      "n_frames": 4,
      "stable": true,
      "label_conflict": false,
      "boxes": {
        "cam_left": [
          146.0,
          197.0,
          217.0,
          281.0
        ],
        "cam_right": [
          213.0,
          197.0,
          295.0,
          281.0
        ]
      },
      "crop_refs": {}
    },
    {
      "instance_id": "I4",
      "track_ids": [
        "L4",
        "R4"
      ],
      "camera_ids": [
        "cam_left",
        "cam_right"
      ],
      "plane_xy": [
        46.66,
        53.54
      ],
      "candidates": {
        "S0010": 0.092,
        "S0026": 0.3227,
        "S0040": 0.2815
      },
      "n_frames": 4,
      "stable": true,
      "label_conflict": true,
      "boxes": {
        "cam_left": [
          232.0,
          204.0,
          400.0,
          537.0
        ],
        "cam_right": [
          258.0,
          202.0,
          434.0,
          540.0
        ]
      },
      "crop_refs": {}
    }
  ]
}
```
</details>

### 모델 텐서 규격 (직접 ONNX를 부를 경우)

```
dino_arc.onnx        (88.0 MB, sha256 dfcd17f7…, opset 17)
  IN  'input'   float32 ['b',3,224,224]   RGB, letterbox(회색 128 패딩), /255 후 ImageNet mean/std
  OUT 'emb'     float32 ['b',256]         L2 정규화됨 → 내적이 곧 코사인 유사도

product_det_640.onnx (10.6 MB, sha256 813e127a…, opset 12)
  IN  'images'  float32 [1,3,640,640]     RGB, letterbox, /255
  OUT 'output0' float32 [1,5,8400]        (cx,cy,w,h,score) — 클래스 1개 'product'
                                          NMS 미포함 (그래프에 NonMaxSuppression 노드 없음)
```

전처리를 직접 구현하기보다 `cartgate.embed.get_embedder()` / ultralytics `YOLO`를 쓰는 편이
안전합니다. letterbox·정규화·NMS가 이미 맞춰져 있습니다.

## 4. 리포 코드 지도 — 무엇이 서빙 코드이고 무엇이 아닌가

### 서버에 필요한 코드 (런타임 경로)

| 파일 | 하는 일 |
|---|---|
| `cartgate/embed.py` | ONNX 임베더. 크롭 전처리(레터박스 224) + 배치 추론 `embed_batch()` |
| `cartgate/gallery.py` | `load_gallery()`(서빙) / `build_gallery()`(오프라인 재생성) |
| `cartgate/match.py` | 크롭 벡터 ↔ 갤러리 코사인 유사도 |
| `cartgate/vision_fusion.py` | 2캠 인스턴스 융합 + `build_observation()` — **비전 산출물의 정의** |
| `cartgate/calibrate_plane.py` | 게이트 1회 캘리브레이션(호모그래피) + 검증 |
| `cartgate/config.py` | 임계값. 비전 소유(`DET_CONF`,`TRACK_IOU`,`MIN_FRAMES`)와 판정 소유 구분 |
| `cartgate/vision.py` | **런타임 핵심** — `resolve_camera()`(검출→ByteTrack→임베딩), `load_fusion()`, `save_evidence()` |
| `cartgate/verification/reference_verify.py` | 판정 참조 구현 (팀원 소유, 서버에선 이걸 대체) |

### 오프라인 전용 — 서버에 올릴 필요 없음

| 파일 | 언제 쓰나 |
|---|---|
| `scripts/pipeline.py` | 합성 카트 데모 하네스 (런타임 코드는 `cartgate/vision.py`로 분리됨) |
| `scripts/train_recognition.py` | 인식 임베더 학습 → `dino_arc.onnx` 생성 |
| `scripts/train_detector.py` | 검출기 학습 → `best.pt` 생성 |
| `cartgate/train_embed.py` | MobileNetV3 베이스라인 + augmentation 유틸 |
| `cartgate/synth.py` | 합성 카트 장면 생성 (학습·벤치마크 데이터) |
| `cartgate/segment.py` | 상품 사진 누끼 (학습 데이터 준비) |
| `scripts/make_cart_dataset.py` | 벤치마크 데이터셋 + `gate_calib.json` 생성 |
| `scripts/eval_carts.py` | 500카트 성능 평가 (false-stop / miss) |
| `scripts/mine_crops.py` | 벤치마크 GT 박스에서 인식 학습용 크롭 추출 |
| `scripts/ingest.py` | 원본 상품 사진 → `dataset/` 정리 |
| `scripts/make_paired.py`, `viz_recognition.py`, `stress_test.py` | 데이터 준비·시각화·강건성 점검 |
| `scripts/build_handoff.py` | 백엔드용 상품 마스터(xlsx) 생성 |
| `scripts/export_deploy.py` | 검출기 ONNX export + TensorRT 명령 안내 |
| `tests/test_boundary.py` | 비전↔판정 경계 테스트 |

---

## 5. 실행 환경

```bash
conda create -n cartgate python=3.11 -y && conda activate cartgate
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt && pip install -e .
```

- **`onnxruntime-gpu==1.22.0` 고정입니다.** 1.23+ 는 CUDA 13을 요구해 이 환경(CUDA 12.6)에서 안 뜹니다.
- **CUDA 실행에는 `torch` import가 선행돼야 합니다.** onnxruntime-gpu가 cuDNN을 안 들고 오는데
  torch 휠 안의 `libcudnn.so.9`를 빌려 쓰기 때문입니다(`cartgate/embed.py:_preload_cuda_libs`).
  torch 없이 띄우려면 `pip install nvidia-cudnn-cu12` 또는 `LD_LIBRARY_PATH` 설정이 필요합니다.
- 실제로 CUDA를 쓰는지 반드시 확인하세요. CPU로 조용히 폴백해도 동작은 합니다(6배 느림):
  ```python
  print(embedder.providers)   # ['CUDAExecutionProvider', 'CPUExecutionProvider']
  ```
- 자세한 내용은 [`RUNTIME_ENV.md`](RUNTIME_ENV.md).

## 6. 성능 (L40S 실측)

| 구간 | 시간 |
|---|---|
| 검출기 1프레임 (640) | 6.9 ms |
| 임베더 1크롭 | 3.6 ms (배치 8이면 1.72 ms/크롭) |
| **카트 1대 (2캠 × 4프레임)** | **250 ms** |
| 첫 요청 (CUDA 워밍업) | 1.2~1.5 s → 기동 시 더미 추론 1회 권장 |

메모리는 검출기 + 임베더 합쳐 GPU 1~2 GB 수준입니다.

## 7. 경계 — 어디까지가 이 리포인가

```
[게이트 카메라] ─프레임─▶ [AI 비전 서비스]  ─VisionObservation JSON─▶ [백엔드]
                            (이 리포)                                  (설계·구현: 백엔드팀)
```

이 서비스는 **상태가 없고**, 카트 한 대 분량의 프레임과 영수증 SKU를 받아
"무엇이 몇 개 보이고 각각이 영수증 SKU와 얼마나 닮았는가"만 JSON으로 답합니다.

**이 리포가 하지 않는 것** — 결제 조회, PASS/FLAG/REVIEW 판정, 게이트 개폐, 세션·큐·캐시,
재시도, 인증. 전부 백엔드 소유입니다. 판정만은 참조 구현
(`cartgate/verification/reference_verify.py`)이 있으니 그대로 쓰거나 바꿔서 쓰시면 됩니다.

**AI 쪽에서 알려드릴 수 있는 제약 사실**(설계 판단은 백엔드팀 몫):

| 사실 | 근거 |
|---|---|
| 카트 1대 응답 250~600 ms, 첫 요청은 워밍업 포함 1.2~1.5 s | L40S 실측 |
| 카메라당 프레임 2장 미만이면 모든 인스턴스가 `stable:false` | `min_frames=2` |
| GPU 1장 기준 동시 추론은 순차 처리됨 (내부 큐 없음) | 단일 프로세스 |
| CPU 폴백 시에도 200 OK지만 6배 느림 | `/healthz`의 `gpu` 필드로 확인 가능 |
| 같은 입력이라도 재추론 시 유사도가 소수점 4째 자리에서 흔들릴 수 있음 | GPU 부동소수 |
| 관측 JSON을 저장해 두면 재추론 없이 판정만 다시 돌릴 수 있음 | 판정은 순수 함수 |

## 8. 번들에 없는 것 / 주의

1. **실제 게이트 캘리브레이션이 없습니다.** `sample_gate_calib.json`은 합성 기하 값입니다.
   실제 설치 후 `cartgate/calibrate_plane.py`로 한 번 만들어야
   `PlaneMatchFusion`(카메라 간 중복 제거)이 켜집니다. 없으면 `AsymmetricFusion`으로
   동작하고, 판정 레이어가 conservative 모드로 받습니다.
2. **정확도가 아직 배포 수준이 아닙니다.** 500카트 벤치마크에서 정상 카트를 세우는 비율
   (false-stop)이 어떤 설정에서도 40.8% 아래로 안 내려갑니다. 원인은 인식 top-1 72~74%이고,
   오인식의 68%가 sim ≥ 0.55로 "자신 있게" 틀립니다. 서비스 오픈 전 인식 재학습이 필요합니다.
   숫자 근거는 `README.md` 현재 성능 절 / `out/eval_carts.json`.
3. **`productId → sku_id` 매핑이 미확정**입니다(계약 §7 D-5). 현재 참조 구현은
   `f"S{productId:04d}"`로 단순 변환하는데 보장이 없습니다. **바코드(EAN-13) 조인 권장**,
   `products.csv`에 51/51 다 있습니다.
4. **증거 크롭 저장소**: `out/crops/{transaction_id}/{instance_id}_{camera_id}.jpg` 규약으로
   FLAG/REVIEW 인스턴스만 저장합니다. 실제 저장 위치(로컬/S3)와 보존 기간은 결정 필요.
5. Jetson 배포는 `deploy/*.onnx`에서 기기 위에서 `trtexec`로 엔진을 빌드해야 합니다
   (엔진은 하드웨어 종속이라 미리 못 만듭니다).
