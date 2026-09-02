# 비전 ↔ 판정 인터페이스 계약서 v1.1

2026-08-16 · v1.0 대체
전제: 카메라 2대(양 상단 대각선), QR 리더기 트리거, Jetson 배포

---

## 1. 경계 — 한 줄 원칙

> **비전은 "무엇이 몇 개 있는가"까지만 답한다. "결제와 맞는가"는 판정이 답한다.**

| | 비전 (나) | 판정 (팀원) |
|---|---|---|
| 담당 | 링버퍼 캡처 · 검출 · 임베딩 · 유사도 계산 · **2캠 인스턴스 융합** | 결제 조회 · **영수증 대조** · verdict · 2차 설명 |
| 아는 것 | 카메라, 기하, 캘리브레이션, 모델 | 영수증, 결제 DB, 운영 정책 |
| 모르는 것 | 영수증 수량, verdict | 카메라, 모델, 픽셀 |
| 산출물 | `VisionObservation` (verdict 없음) | `GateVerdict` → 백엔드 |

**융합이 비전에 있는 이유:** 카메라 캘리브레이션과 평면 기하를 알아야 합니다.
**대조가 판정에 있는 이유:** 결제 DB와 운영 정책의 문제입니다.

### 임계값 소유권 (중요)

| 임계값 | 소유 | 이유 |
|---|---|---|
| `min_frames`, `merge_radius_cm`, `det_conf` | **비전** | 관측 품질·기하 |
| `SIM_STRONG(0.55)`, `SIM_WEAK(0.42)` | **판정** | "얼마나 닮아야 같은 물건인가"는 판단 |

→ **실영상으로 유사도 임계값을 재보정할 때 판정 레이어만 고치면 됩니다.** 비전은 유사도 숫자를 보고할 뿐 해석하지 않습니다.

---

## 2. 시퀀스

```
QR 태깅 (t=0)
  ├─[판정] PaymentRepository.fetch(qr_token) → 영수증
  │              │
  │        receipt.sku_list ──────────────┐
  │                                       ▼
  └─[비전] 링버퍼 [-0.3s,+0.7s] 캡처 → 검출 → 임베딩   ← 영수증 불필요(병렬)
                                              │
                                     후보 유사도 계산 → 2캠 융합
                                              │
                                     VisionObservation (JSON)
                                              ▼
           [판정] 헝가리안 할당 → PASS/REVIEW/FLAG
                          │
                          ▼ (FLAG일 때만)
           [판정] UnknownItemIdentifier — 전체 카탈로그로 "뭐였나" 조회
                          │
                          ▼
                    GateVerdict → 백엔드
```

비전 API 2개:
- `capture(trigger_id)` — QR 트리거 시 호출. 캡처·검출·임베딩까지 (영수증 불필요)
- `observe(trigger_id, receipt_sku_list) → VisionObservation`

---

## 3. VisionObservation (비전 → 판정)

```json
{
  "schema_version": "1.1",
  "transaction_id": "TX-20260816-000937",
  "gate_id": "GATE-03",
  "captured_at": "2026-08-16T14:23:05+09:00",
  "duration_ms": 1000,
  "vision_status": "OK",
  "cameras": [{"camera_id": "cam_left", "status": "OK", "frames_used": 10}],
  "fusion_strategy": "plane_match",
  "cross_camera_resolved": true,
  "min_frames": 2,
  "instances": [
    { "instance_id": "I3",
      "track_ids": ["L3", "R2"],
      "camera_ids": ["cam_left", "cam_right"],
      "plane_xy": [525.0, 150.0],
      "candidates": {"S0032": 0.726, "S0035": 0.723, "S0047": 0.22},
      "n_frames": 4,
      "stable": true,
      "label_conflict": true,
      "boxes": {"cam_left": [500,100,550,150]},
      "crop_refs": {"cam_left": "gate03/TX-.../L3.jpg"} }
  ]
}
```

| 필드 | 의미 |
|---|---|
| `instances[]` | **물리적 객체 1개 = 1엔트리.** 2캠 융합 후 |
| `candidates` | **영수증 SKU에 대한 유사도 전체.** argmax만 남기면 안 됨 — 전역 할당에 필수 |
| `cross_camera_resolved` | `true`=캘리브레이션 완료, 인스턴스가 실제 객체 수 / `false`=미캘리브, 카메라별 중복 가능 |
| `label_conflict` | 두 카메라가 SKU를 다르게 봄. 정보용 (판정이 전역 할당으로 해소) |
| `plane_xy` | 카트 평면 좌표(cm). 미캘리브 시 `null` |

**없는 것:** verdict, band, "이게 뭐다"라는 결론. 전부 관측이지 판단이 아닙니다.

---

## 4. GateVerdict (판정 → 백엔드)

```json
{ "schema_version": "1.1",
  "transaction_id": "TX-20260816-000937",
  "verdict": "FLAG",
  "reasons": [
    {"code":"QTY_EXCEEDED","severity":"FLAG","sku_id":"S0007",
     "paid_qty":1,"observed_qty":2,"sim":0.71},
    {"code":"UNEXPLAINED_ITEM","severity":"FLAG","instance_id":"I3",
     "top_sim":0.31,"crop_refs":{"cam_left":"..."}}
  ],
  "observed_counts": {"S0007": 2},
  "unseen_paid_items": [{"sku_id":"S0031","qty":1}],
  "decision_mode": "assignment" }
```

| code | severity |
|---|---|
| `QTY_EXCEEDED` | FLAG |
| `UNEXPLAINED_ITEM` | FLAG |
| `AMBIGUOUS_ITEM` | REVIEW |
| `VISION_UNAVAILABLE` | INFO (fail-open) |

---

## 5. 판정 규칙

`cross_camera_resolved` 값에 따라 두 모드로 갈립니다.

**`true` — assignment 모드 (캘리브레이션 후)**

인스턴스 × 영수증 슬롯(수량 1 = 슬롯 1) 헝가리안 전역 할당, cost = `1 − sim`.

- 슬롯 배정 + `sim ≥ 0.55` → 정상
- 슬롯 배정 + `sim < 0.55` → `AMBIGUOUS_ITEM` (REVIEW)
- 미배정 + `best_sim < 0.42` → `UNEXPLAINED_ITEM` (FLAG)
- 미배정 + `best_sim ≥ 0.42` → `QTY_EXCEEDED` (FLAG)
- 남은 슬롯 → `unseen_paid_items` (**판정 미반영**)

> **전역 할당이 핵심입니다.** 개별 물체를 따로 판단하면 애매한 물체가 이미 설명된 SKU를 또 주장하지만, 전체를 한 번에 최적화하면 그 물체는 *아직 비어 있는* 영수증 줄을 받습니다. 실제 오탐 케이스가 이걸로 해소됩니다.

**`false` — conservative 모드 (캘리브레이션 전)**

같은 물체가 카메라마다 중복될 수 있으므로 카메라별로 세고,
**존재 = max**(가림 복구) / **수량초과 = min**(오탐 방지).

---

## 6. 소유권

| 파일 | 담당 |
|---|---|
| `cartgate/vision_fusion.py` | 비전 |
| `cartgate/calibrate_plane.py` | 비전 |
| `scripts/gate_service.py` (링버퍼·트리거) | 비전 |
| `cartgate/verification/` (VerificationService) | **판정** |
| `cartgate/payments/` (PaymentRepository, json_adapter) | **판정** |
| `cartgate/catalog_recognition/` (→ `UnknownItemIdentifier`) | **판정**, ⑥단계로 이동 |

**PR #1 처리**

| 코드 | 처리 |
|---|---|
| `PaymentRepository` / `json_adapter` | ✅ 그대로 사용 |
| `VerificationService` | ✅ 사용, 비교 로직을 `decision_verify.verify()`로 교체 (2-way→3-way) |
| `CatalogRecognizer` | 🔄 FLAG 이후 2차 설명기로 이동 + 개명 |
| `recognize_tracks`의 임베딩 수행 | ❌ 제거 — 비전이 담당 |
| `productId → S%04d` | 🔄 바코드 조인으로 교체 (백엔드 회신 대기) |

---

## 7. 열린 결정

| ID | 항목 | 권고 |
|---|---|---|
| D-1 | `vision_status == FAILED` | **fail-open** (통과+로깅). 증거 0으로 고객을 세우지 않음 |
| D-2 | 카메라 1대 장애 | 남은 1대로 진행 + `SIM_STRONG` 상향 |
| D-5 | 결제 DB 상품 식별자 | 백엔드 회신 대기 (바코드 권장) |

---

## 참조 구현

- `vision_fusion.py` — 비전 측. 융합 2전략 + `build_observation()`
- `decision_verify.py` — **판정 측 참조 구현.** `VerificationService`의 출발점
- `test_boundary.py` — 비전 → JSON → 판정 6시나리오. **양 모드 6/6 통과**

경계는 **JSON**입니다. 판정 레이어는 비전의 Python 객체를 import하지 않습니다.
