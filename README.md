# CartGate

셀프 계산대 매장을 위한 AI 출구 게이트. 손님이 휴대폰으로 스캔·결제하고, 출구에서 카메라가
카트를 보고 **보이는 물건이 결제 내역과 일치하는지** 확인합니다. 카트 안을 전부 인식하는 게
아니라 **영수증 조건부 이상탐지(receipt-conditioned anomaly detection)** — 미결제·수량 초과만
잡으면 되므로, 보이는 물건을 전체 카탈로그가 아니라 *그 카트의 영수증 SKU*와만 대조합니다.

## 파이프라인

```
 카메라 2대 (양쪽 위 대각선, ~30fps)
        │
        ▼
 [1] 검출(Detector)     YOLO11n, "product" 단일 클래스   → 박스 (클래스 무관)
        │
        ▼
 [2] 인식(Recognition)  DINOv2-S/14 + ArcFace 헤드 → 256-d 임베딩,
        │               영수증 SKU에만 대조
        ▼
 [3] 집계(Aggregation)  다중 프레임 추적 + 2카메라 융합 → SKU별 수량
        │
        ▼
 [4] 판정(Decision)     결제 DB와 대조 → PASS / FLAG / REVIEW
```

**[1]~[3]은 이 리포에 구현·학습되어 있습니다. [4] 판정 레이어는 팀원 담당** —
[`docs/DECISION_LAYER.md`](docs/DECISION_LAYER.md) 참고.

## 디렉토리 구조

```
cjs/
├─ cartgate/              # 라이브러리 (import 되는 모듈)
│   ├─ embed.py           #   임베딩 백엔드 (classical + ONNX, 배치 추론)
│   ├─ gallery.py         #   SKU별 임베딩 갤러리 (지문 기반 캐시)
│   ├─ match.py           #   크롭↔갤러리 코사인 유사도 (비전 몫만 남김)
│   ├─ vision.py          #   런타임: 검출→추적→임베딩 (서비스가 부르는 지점)
│   ├─ vision_fusion.py   #   2캠 인스턴스 융합 + VisionObservation  ← 비전 소유
│   ├─ calibrate_plane.py #   카트 평면 캘리브레이션 (호모그래피)   ← 비전 소유
│   ├─ config.py          #   임계값 (비전/판정 소유 구분)
│   ├─ synth.py           #   합성 카트 장면 (시간 일관 · 카메라 고정)
│   ├─ segment.py         #   누끼 매팅 (rembg / grabcut)
│   ├─ train_embed.py     #   MobileNetV3 베이스라인 + 공용 augmentation 유틸
│   └─ verification/      #   판정 레이어  ← 팀원 소유
│       └─ reference_verify.py   영수증 대조 참조 구현 (전역 할당 / conservative)
│
├─ scripts/               # 실행 진입점 (리포 루트에서 실행)
│   ├─ ingest.py                # 원본 사진 → dataset/
│   ├─ make_cart_dataset.py     # 합성 카트 데이터셋 + gate_calib.json
│   ├─ train_detector.py        # 검출기 학습 (YOLO, cart_dataset/data.yaml)
│   ├─ train_recognition.py     # 인식 임베더 학습 → dino_arc.onnx
│   ├─ pipeline.py              # 합성 카트 데모 (cartgate/vision.py 를 호출)
│   ├─ eval_carts.py            # 500카트 전수 평가 (false-stop / miss)
│   ├─ viz_recognition.py       # 인식 결과 시각화
│   ├─ stress_test.py           # 강건성 스윕 (혼잡도 × 화질)
│   ├─ make_paired.py           # 원본 + BiRefNet 누끼 pair 생성
│   ├─ build_handoff.py         # 백엔드용 상품 마스터 (xlsx/csv)
│   └─ export_deploy.py         # Jetson용 ONNX export
│
├─ tests/test_boundary.py   # 비전→JSON→판정 경계 테스트 (6 시나리오 × 2 융합)
├─ docs/
│   ├─ CONTRACT_v1.1.md     # 비전↔판정 인터페이스 계약 (기준 문서)
│   ├─ DECISION_LAYER.md    # 판정 레이어 설계 노트 (팀원용)
│   └─ RUNTIME_ENV.md       # onnxruntime-gpu / CUDA / Jetson 주의사항
├─ products.csv            # 상품 마스터 (sku_id · 상품명 · EAN-13 바코드)
├─ pyproject.toml          # 패키지 정의 (pip install -e .)
├─ requirements.txt
└─ (데이터·모델 디렉토리 — git 제외, 별도 전달)
```

> 데이터·모델·산출물은 `.gitignore` 처리되어 git에 안 올라갑니다 (**데이터** 항목 참고).

## 설치

이 환경 드라이버가 CUDA 12.6이라, 맞는 torch로 conda 환경을 씁니다:

```bash
conda create -n cartgate python=3.11 -y
conda activate cartgate
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .          # cartgate 패키지를 import 가능하게 등록
```

CPU만으로도 실행 가능(‑‑index-url 빼고 `onnxruntime` 사용). 학습은 GPU 필요.
스크립트는 **리포 루트에서 실행**하세요 (`dataset/`, `dino_arc.onnx` 등 경로가 상대 경로).

## 데이터

용량이 커서 git에 없습니다. 전달받은 번들을 리포 루트에 풀면 아래 구조가 됩니다:

```
dataset/
  images/<sku_id>__<이름>/*.jpg     상품 사진 (51 SKU) — 갤러리 소스
  barcode/<sku_id>__<이름>.png      상품 바코드
out/cut_rembg/                      인식 학습용 누끼 (RGBA)
cart_dataset/                       합성 카트 (2캠 대각선) — 판정 벤치마크 + 검출기 학습(YOLO)
runs/detect/.../weights/best.pt     학습된 검출기
dino_arc.onnx                       학습된 인식 임베더
yolo11n.pt                          검출기 베이스 가중치
```

`products.csv`(sku_id·상품명·EAN-13 바코드)는 상품 마스터로 git에 포함됩니다.

판정 레이어용 **합성 카트 벤치마크**는 `scripts/make_cart_dataset.py`로 생성합니다
(`cart_dataset/`: 카트별 영수증·실내용물·카메라별 GT·정답 PASS/FLAG). 포맷은 `cart_dataset/README.md` 참고.

## 실행

### AI HTTP 서버

게이트 장치가 카메라 2대의 프레임을 업로드하는 서버는 `cartgate.server.api`에 있다.
Spring의 검사 시작·완료·실패 API와 연동하며, Nginx는 Docker 네트워크의 `fastapi:8000`으로
전달한다. 모델 번들 배치, 환경변수, Jetson GPU 확인 및 요청 형식은
[`DEPLOYMENT.md`](DEPLOYMENT.md)를 따른다.

```bash
# 엔드투엔드 데모 (검출 → 인식 → 융합 → 판정)
python scripts/pipeline.py

# 어려운 장면(가림/저해상도/블러/저조도) 인식 시각화  → out/viz/
python scripts/viz_recognition.py

# 강건성 스윕: 혼잡도 × 화질별 인식/e2e
python scripts/stress_test.py

# 500카트 벤치마크 전수 평가 (false-stop / miss rate)  → out/eval_carts.json
python scripts/eval_carts.py --limit 500 --calib cart_dataset/gate_calib.json

# 백엔드용 상품 마스터 (xlsx + csv)  → service_products/
python scripts/build_handoff.py --no-images
```

재학습 (GPU):

```bash
python scripts/make_cart_dataset.py --num 500 --frames 4 && python scripts/train_detector.py   # 검출기
python scripts/train_recognition.py                                                  # 인식 → dino_arc.onnx
```

## 현재 성능

### 벤치마크 평가 (2026-09-02, `scripts/eval_carts.py`)

지표는 정확도가 아니라 **운영점**입니다. REVIEW는 손님 입장에서 멈추는 것과 같으므로
false-stop에 포함합니다. 아래는 학습 크롭과 시드가 겹치지 않는 **홀드아웃 300카트** 결과.

| 인식 모델 | 최저 false-stop | 같은 상한(45%)에서 miss | 비고 |
|---|---|---|---|
| v1 (스튜디오 누끼 학습) | 44.2% | 28.2% | 2026-08-17 |
| **v2 (게이트 크롭 혼합 학습)** | **27.9%** | **20.9%** | 현재 `dino_arc.onnx` |

v2는 프론티어 전 구간에서 v1보다 낫습니다(같은 false-stop 상한에서 miss −1.8~−7.3%p).
그래도 **정상 카트 4대 중 1대 이상을 세우므로 아직 서비스 투입 불가**입니다.

병목은 여전히 인식입니다:
- v1 기준 실제 트랙 top-1 72~74%, 오인식의 68%가 sim ≥ 0.55로 자신 있게 틀림
- 검출·추적은 병목이 아님: 트랙 수 = GT 객체 수의 1.04배, 미매칭 4.5%
- v2 학습은 이 격차의 원인(학습 크롭은 고립된 물체 / 실제는 파묻힌 물체)을 좁힌 것이고,
  남은 격차는 상품 사진 수(SKU당 2~3장)와 가림 자체의 난이도로 보입니다

> 과거 README의 **인식 90.8%**는 프레임마다 물체 배치가 새로 뽑히던 v1 합성 데이터 기준이라
> 폐기했습니다. 검출기 mAP50 ≈ 0.94 / mAP50-95 ≈ 0.80은 v1 학습 당시 기록이며 재검증 로그가 없습니다.

### 지연 (L40S, onnxruntime-gpu CUDA EP)

- 검출기 1프레임 6.9 ms · 임베더 1크롭 3.6 ms (배치 8일 때 1.72 ms/크롭)
- 카트 1대(2캠 × 4프레임) 실측 **250 ms**, 첫 카트만 CUDA 워밍업으로 1.2~1.5 s

전부 합성 카트 기준입니다. 가장 큰 남은 과제는 **인식 정확도**이고, 그다음이
**실제 게이트 영상으로의 검증과 임계값 보정**입니다.

## 참고

- 인식은 *영수증 제한*: 크롭을 그 카트 영수증 SKU와만 대조 → SKU당 1~3장만으로도 동작.
- 런타임엔 배경 제거 안 함. 누끼(BiRefNet)는 오프라인 학습 데이터 합성용으로만 사용
  (물체를 다양한 배경에 얹어 배경 강건성 확보).
- 배포 타깃은 Jetson(AGX/Orin) + TensorRT. `export_deploy.py`가 검출기 ONNX를 뽑고,
  `dino_arc.onnx`가 인식 임베더입니다.
