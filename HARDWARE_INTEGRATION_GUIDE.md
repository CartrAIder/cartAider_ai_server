# 게이트 하드웨어 ↔ AI 서버 연동 가이드

- 작성일: 2026-09-16
- 대상: 게이트 장치, 카메라, QR/결제 토큰 전달부를 구현하는 하드웨어·임베디드 담당자
- 기준 구현: `POST /v1/gate/inspections`

## 1. 하드웨어 담당 범위

게이트 장치는 한 번의 검사마다 다음 작업을 수행한다.

1. 카트가 촬영 위치에 들어오면 사용자가 제시한 QR 코드를 읽는다.
2. QR 코드에서 결제 완료 후 발급된 `gate_token`을 얻는다.
3. 좌·우 카메라에서 같은 카트를 촬영한 연속 프레임을 준비한다.
4. `gate_token`, `gate_id`, 두 카메라의 프레임을 하나의 HTTP multipart 요청으로 AI 서버에 전송한다.
5. AI 서버가 반환한 `PASS`, `REVIEW`, `FLAG`를 게이트 제어 로직에 전달한다.

AI 서버가 직접 카메라를 제어하거나 영상을 스트리밍으로 가져가지는 않는다. 게이트 장치가 캡처를 끝낸 이미지 묶음을 업로드한다. 게이트 장치는 Spring을 직접 호출하지 않으며, 결제 목록 조회와 최종 결과 저장은 AI 서버가 담당한다.

## 2. 전체 처리 흐름

```text
카트가 촬영 위치에 진입
  → 사용자가 QR 코드를 제시
  → 게이트 장치가 QR 코드에서 gate_token 획득
  → cam_left / cam_right 연속 프레임 캡처
  → gate_token + gate_id + 두 카메라 프레임을 AI 서버에 multipart 업로드
  → AI 서버가 gate_token으로 Spring 검사 시작 API 호출
  → Spring이 결제 상태를 검증하고 결제 상품의 barcode·qty 반환
  → 상품 검출·추적·인식
  → 좌·우 카메라의 같은 물체 융합
  → Spring 결제 목록과 AI 인식 결과를 비교
  → AI 서버가 Spring에 PASS / REVIEW / FLAG 저장
  → 게이트 장치에 PASS / REVIEW / FLAG 응답
```

HTTP 200 응답을 받았을 때는 AI 서버가 Spring 완료 API까지 성공시킨 상태다.

정리하면 비즈니스 흐름은 **QR 인식 → Spring 결제 목록 조회 → AI 인식 목록과 비교 → 최종 판정**이다. 현재 HTTP 구현에서는 게이트 장치가 QR 토큰과 카메라 프레임을 함께 AI 서버에 보낸 뒤, AI 서버 내부에서 Spring 조회가 수행된다.

## 3. 카메라 구성

| 항목 | 요구사항 |
|---|---|
| 카메라 수 | 정확히 2대 |
| 왼쪽 카메라 ID | `cam_left` |
| 오른쪽 카메라 ID | `cam_right` |
| 권장 배치 | 카트 양쪽 상단의 대각선 방향 |
| 최소 프레임 수 | 카메라별 2장, 기본 설정 `MIN_FRAMES_PER_CAMERA=2` |
| 권장 프레임 수 | 카메라별 4장 |
| 프레임 순서 | 촬영 시간순 |
| 업로드 단위 | 한 카트의 두 카메라 프레임을 한 요청에 포함 |

카메라별로 2장보다 적게 보내면 검사가 실패하고 Spring에는 `CAMERA_UNAVAILABLE`로 기록된다.

현재 요청에는 프레임별 타임스탬프 필드가 없다. AI는 전달받은 배열의 가운데 프레임을 게이트 트리거 시점에 가장 가까운 프레임으로 간주한다. 따라서 두 카메라 모두 같은 시간 구간을 촬영하고, 프레임을 시간순으로 보내야 한다.

## 4. 캘리브레이션과 촬영 조건

좌·우 카메라에서 검출된 물체를 하나의 실제 물체로 합치려면 실측 `gate_calib.json`이 필요하다. 캘리브레이션은 이미지 픽셀 좌표와 카트 평면 좌표의 관계를 사용한다.

캘리브레이션 이후에는 다음 항목을 유지해야 한다.

- 카메라 장착 위치와 높이
- 카메라 각도와 렌즈
- 입력 이미지 해상도와 화면 비율
- 카메라 내부 크롭·디지털 줌 설정
- 카트가 정지하거나 통과하는 기준 위치

위 조건이 바뀌면 기존 캘리브레이션이 무효가 될 수 있으므로 다시 보정해야 한다. `sample_gate_calib.json`은 합성 환경 예시이며 실제 게이트에 사용할 수 없다.

## 5. 검사 요청 API

### 요청

```http
POST /v1/gate/inspections
Content-Type: multipart/form-data
```

| multipart 필드 | 형식 | 필수 | 설명 |
|---|---|---|---|
| `gate_token` | 문자열 | 예 | Spring 결제 완료 응답에서 받은 게이트 토큰 |
| `gate_id` | 문자열 | 예 | 설치 게이트 식별자, 예: `GATE-01` |
| `cam_left` | 이미지 파일 반복 | 예 | 왼쪽 카메라의 시간순 연속 프레임 |
| `cam_right` | 이미지 파일 반복 | 예 | 오른쪽 카메라의 시간순 연속 프레임 |

`cam_left`, `cam_right` 필드를 프레임 수만큼 반복한다. ZIP, 동영상, JSON base64, RTSP 주소는 현재 입력 형식이 아니다.

### curl 예시

```bash
curl --request POST "https://<AI_HOSTNAME>/v1/gate/inspections" \
  --form "gate_token=<발급받은 토큰>" \
  --form "gate_id=GATE-01" \
  --form "cam_left=@left_001.jpg;type=image/jpeg" \
  --form "cam_left=@left_002.jpg;type=image/jpeg" \
  --form "cam_left=@left_003.jpg;type=image/jpeg" \
  --form "cam_left=@left_004.jpg;type=image/jpeg" \
  --form "cam_right=@right_001.jpg;type=image/jpeg" \
  --form "cam_right=@right_002.jpg;type=image/jpeg" \
  --form "cam_right=@right_003.jpg;type=image/jpeg" \
  --form "cam_right=@right_004.jpg;type=image/jpeg"
```

하드웨어 장치는 Spring 내부 통신용 `GATE_SERVICE_SECRET`을 전송하지 않는다. 이 Secret은 AI 서버와 Spring 사이에서만 사용한다.

## 6. 이미지 조건과 용량

- OpenCV가 디코딩할 수 있는 이미지 파일이어야 한다. 게이트 구현에서는 JPEG 사용을 권장한다.
- 빈 파일이나 손상된 이미지는 HTTP 422로 거절된다.
- 이미지 한 장의 최대 크기는 5MB다.
- 운영 Nginx의 요청 전체 제한은 20MB다.
- 헤더와 multipart 경계 용량도 있으므로 전체 파일 합계는 20MB보다 충분히 작게 유지한다.
- 좌·우 프레임은 가능한 한 같은 노출·해상도·프레임 수를 사용한다.
- 캘리브레이션에 사용한 해상도와 실제 업로드 해상도를 일치시킨다.

권장 구성은 카메라별 JPEG 4장, 총 8장이며 전체 요청 크기를 20MB 미만으로 유지하는 것이다.

## 7. 정상 응답

### PASS

```json
{"verdict": "PASS"}
```

보이는 물체가 결제 내역으로 설명되며, 게이트 통과가 가능한 상태다. 결제됐지만 카메라에 보이지 않은 물체는 현재 판정에서 불이익을 주지 않는다.

### REVIEW

```json
{"verdict": "REVIEW"}
```

물체는 결제 상품과 어느 정도 유사하지만 확신도가 부족한 상태다. 운영 정책에 따라 직원 확인이나 재촬영 흐름으로 연결한다.

### FLAG

```json
{"verdict": "FLAG"}
```

미결제 가능성이 있는 물체 또는 결제 수량보다 많은 물체가 관측된 상태다. 운영 정책에 따라 게이트 정지와 직원 확인 흐름으로 연결한다.

AI 서버는 모터나 게이트를 직접 제어하지 않는다. 실제 개폐 정책은 게이트 제어 시스템이 최종 verdict를 받아 적용한다.

## 8. 오류 응답

| HTTP 상태 | 의미 | 하드웨어 측 처리 |
|---|---|---|
| 200 | 검사와 Spring 결과 저장 완료 | verdict에 따른 게이트 동작 |
| 413 | Nginx 요청 전체 용량 초과 가능성 | 이미지 품질·장수·용량을 줄여 재전송 |
| 422 | 카메라 누락, 프레임 부족, 빈 파일, 손상 이미지, 알 수 없는 multipart 필드 | 입력을 수정하거나 재촬영 |
| 500 | Spring 연동·상품 매핑·내부 처리 오류 | 게이트를 자동 개방하지 말고 운영 장애 흐름으로 전환 |
| 502 | 모델 추론 실패 | 재촬영 또는 운영 장애 흐름으로 전환 |
| 503 | AI 서버 또는 모델 준비 안 됨 | 잠시 후 상태 확인, 운영 장애 흐름으로 전환 |

오류 응답 예시:

```json
{"detail": "cam_right frames are required"}
```

같은 `gate_token`으로 여러 요청을 동시에 보내지 않는다. 네트워크 타임아웃 후 재시도할 때도 이전 요청이 처리 중인지 확인하고, 같은 `gate_token`과 `gate_id`를 유지한다.

## 9. 상태 확인 API

게이트 장치 또는 운영 모니터링은 검사 전에 다음 API로 AI 서버 준비 상태를 확인할 수 있다.

```http
GET /healthz
```

준비 완료:

```json
{"status": "ready", "provider": ["CUDAExecutionProvider", "CPUExecutionProvider"]}
```

- HTTP 200: 모델과 서버가 준비됨
- HTTP 503: 아직 준비되지 않음
- 운영 Jetson에서는 `provider`에 `CUDAExecutionProvider`가 포함되는지 확인

## 10. AI 내부 처리 로직

하드웨어에서 전송한 요청은 다음 순서로 처리된다.

1. 필수 필드와 두 카메라 파일을 검사한다.
2. 모든 이미지가 디코딩 가능한 BGR 이미지인지 검사한다.
3. AI 서버가 Spring에 `gate_token`, `gate_id`를 보내 결제 상품의 barcode·qty를 받는다.
4. barcode를 AI 상품 식별자 `sku_id`로 변환한다.
5. 카메라별로 YOLO 상품 검출과 ByteTrack 추적을 수행한다.
6. 추적된 물체 크롭을 DINO ONNX 임베딩으로 변환한다.
7. 전체 상품 카탈로그가 아니라 해당 영수증의 SKU만 후보로 비교한다.
8. 실측 캘리브레이션이 있으면 카트 평면 좌표로 좌·우 카메라의 같은 물체를 합친다.
9. 물체와 결제 수량을 전역 할당해 `PASS`, `REVIEW`, `FLAG`를 결정한다.
10. AI 서버가 Spring에 완료 또는 실패 상태를 저장한 후 하드웨어에 응답한다.

검출·추적 모델은 내부 상태를 사용하므로 현재 서버는 검사 요청을 순차 처리한다. 여러 게이트가 하나의 AI 서버를 공유하면 앞 요청이 끝날 때까지 다음 추론이 대기할 수 있다.

## 11. 판정 기준 요약

실측 캘리브레이션이 있을 때는 두 카메라의 관측을 물리적 객체 단위로 합친 뒤 결제 수량과 전역 비교한다.

| 조건 | 결과 |
|---|---|
| 결제 상품과 강하게 일치 | 정상 후보 |
| 결제 상품과 약하게 일치 | `REVIEW` |
| 어떤 결제 상품으로도 설명하기 어려움 | `FLAG` |
| 결제 수량보다 물체가 많음 | `FLAG` |
| 결제됐지만 화면에 보이지 않음 | 정보로만 기록, 판정 불이익 없음 |

현재 기준 유사도는 강한 일치 0.55 이상, 약한 일치 0.42 이상이다. 이 값은 실제 게이트 영상 평가 후 조정될 수 있으며, 하드웨어 프로토콜에는 영향을 주지 않는다.

## 12. 하드웨어 구현 체크리스트

- [ ] 촬영 위치에서 사용자의 QR 코드를 읽어 `gate_token`을 추출한다.
- [ ] QR에서 얻은 `gate_token`과 설치된 장치의 `gate_id`를 함께 전송한다.
- [ ] 카메라 ID를 `cam_left`, `cam_right`로 고정했다.
- [ ] 좌·우 카메라가 같은 카트를 같은 시간 구간에 촬영한다.
- [ ] 프레임을 촬영 시간순으로 전송한다.
- [ ] 카메라별 최소 2장, 권장 4장을 전송한다.
- [ ] 이미지 한 장은 5MB 미만이고 전체 요청은 20MB보다 충분히 작다.
- [ ] 캘리브레이션과 실제 촬영의 해상도·렌즈·장착 위치가 같다.
- [ ] 한 카트에 대해 동시에 두 검사 요청을 보내지 않는다.
- [ ] HTTP 200의 verdict와 4xx·5xx 오류 흐름을 각각 처리한다.
- [ ] `/healthz`가 200일 때만 검사를 시작한다.
- [ ] gate token과 내부 오류 내용을 평문 운영 로그에 과도하게 남기지 않는다.

## 13. 관련 문서

- [`DEPLOYMENT.md`](DEPLOYMENT.md): AI 서버 컨테이너와 환경변수
- [`docs/CONTRACT_v1.1.md`](docs/CONTRACT_v1.1.md): 비전·판정 내부 데이터 계약
- [`LOCAL_AND_DEPLOYMENT_READINESS_REPORT.md`](LOCAL_AND_DEPLOYMENT_READINESS_REPORT.md): 현재 실행·배포 준비 상태
