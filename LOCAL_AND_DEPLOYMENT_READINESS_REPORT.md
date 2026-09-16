# AI 서버 로컬·배포 준비 상태 점검 보고서

- 점검일: 2026-09-16
- 대상: `CartGate_AI`의 현재 `main` 브랜치
- 기준: 기존 Spring Boot 게이트 검사 API와 현재 AI 서버 구현
- 점검 범위: 소스·모델 번들·Python 실행 환경·패키징·Docker/Jenkins 설정

## 결론

현재 소스와 단위·계약 테스트는 준비됐지만, **이 Mac의 기본 로컬 환경과 실제 Jetson 배포 환경에서 바로 기동 가능한 상태는 아니다.**

로컬에서는 Python 버전과 추론 의존성이 부족해 생산 앱 초기화가 실패한다. 배포에서는 Spring 공유 Secret, 실제 게이트 보정 파일, Jetson GPU 런타임, Docker 네트워크와 Jenkins Secret 설정을 완료하고 통합 검증해야 한다.

## 확인 결과

| 항목 | 결과 | 근거·범위 |
|---|---|---|
| 전체 자동 테스트 | 통과 | `pytest tests -q -p no:cacheprovider` → **33 passed** |
| FastAPI 업로드 계약 | 통과 | 2대 카메라 필수, 제3 카메라 거절, 최종 verdict 응답 테스트 포함 |
| Spring 연동 계약 | 통과 | 시작·완료·실패 경로, Secret 헤더, snake_case 본문, 완료 재시도 테스트 포함 |
| 바코드→SKU 매핑 | 통과 | UTF-8 BOM·선행 0 바코드·미등록 바코드·잘못된 수량 테스트 포함 |
| 인식 ONNX | CPU 더미 추론 성공 | 출력 `(1, 256)`, 유한값 확인 |
| 모델 갤러리 | 확인 | 51 SKU, `products.csv`의 51 SKU와 집합 일치 |
| Python wheel | 확인 | wheel 안에 `cartgate.server`와 `cartgate.verification` 포함 |
| 기본 로컬 Python | 부적합 | `/usr/bin/python3`은 **3.9.6**, 프로젝트 요구사항은 3.10 이상 |
| 검증용 Python 3.11 | 부분 준비 | FastAPI·HTTPX·ONNX Runtime 등 존재, `torch`·`ultralytics` 없음 |
| 생산 앱 초기화 | 실패 | `create_production_app()` → `VisionInferenceError`, 원인 `ModuleNotFoundError: ultralytics` |
| Docker·공유 네트워크 확인 | 미확인 | 로컬 Docker 소켓 접근 권한 거부로 `docker version`, `docker network inspect` 실패 |
| 실제 게이트 보정 | 미준비 | `gate_calib.json` 없음, `sample_gate_calib.json`만 존재 |
| 운영 Secret | 미설정 | 현재 셸의 `GATE_SERVICE_SECRET` 없음 |
| Jetson GPU 추론 | 미검증 | CUDA provider, 메모리, 지연, TensorRT/JetPack 조합 미검증 |
| 실제 Spring 상태 전이 | 미검증 | 로컬 Spring·Redis와 실제 결제 토큰으로 시작·완료·실패 호출 미실행 |
| 실제 카메라 정확도 | 미검증 | 실제 상품·조도·가림·움직임 조건의 오탐/미탐 미측정 |

## 현재 구현된 연결 구조

```text
게이트 장치
  └─ multipart 업로드: gate_token, gate_id, cam_left[], cam_right[]
       └─ AI 서버 :8000
            ├─ Spring 검사 시작: POST /api/internal/gate/inspections
            ├─ barcode → sku_id 변환
            ├─ 검출·추적·임베딩·2카메라 융합·판정
            ├─ Spring 완료: POST /api/internal/gate/inspections/complete
            └─ 실패 시: POST /api/internal/gate/inspections/fail

Nginx ── Docker network alias `fastapi` ── AI 서버 :8000
AI 서버 ── Docker network alias `spring` ── Spring :8080
```

카메라 이름은 `cam_left`, `cam_right` 두 개로 고정돼 있다. 누락·빈 스트림·알 수 없는 카메라 필드는 요청 오류로 처리한다.

## 배포 전 필수 조치

### 1. Jetson 실행 이미지와 GPU 검증

- Jetson의 JetPack/CUDA에 맞는 Python·PyTorch·ONNX Runtime GPU 조합을 확정한다.
- `torch`, `ultralytics`, `onnxruntime-gpu`가 설치된 컨테이너에서 `create_production_app()`이 성공해야 한다.
- `/healthz` 응답의 `provider`에 `CUDAExecutionProvider`가 포함되는지 확인한다.
- CPU provider만 나오는 경우에는 운영 준비 완료로 처리하지 않는다.

### 2. 모델 번들과 게이트 보정

컨테이너의 `/models`에 다음 파일을 읽기 전용으로 마운트한다.

```text
dino_arc.onnx
products.csv
out/gallery.pkl
runs/detector/best.pt
gate_calib.json
```

- `gate_calib.json`은 실제 설치된 좌·우 카메라로 만든 파일이어야 한다.
- `sample_gate_calib.json`은 합성 예시라서 운영 보정값으로 사용할 수 없다.
- 보정 파일 없이도 코드상 비대칭 융합으로 실행되지만, 실제 운영 정책으로 허용할지 먼저 결정해야 한다.

### 3. Spring·Jenkins 설정

- AI 컨테이너의 `GATE_SERVICE_SECRET`은 Spring의 같은 이름 환경변수와 정확히 일치시킨다.
- `SPRING_BASE_URL=http://spring:8080`을 사용한다.
- 컨테이너는 `cartAider-network`에 `fastapi` alias로 연결한다.
- Jenkins에는 `cartgate-ai-production-env` 이름의 Secret file credential을 생성한다.
- 해당 환경 파일에는 `CARTGATE_MODEL_HOST_DIR`의 절대 경로도 포함한다.

`Jenkinsfile`은 테스트용 Docker stage를 빌드한 뒤, `cartgate-ai-server` 컨테이너를 재생성하고 `http://fastapi:8000/healthz`를 확인한다.

### 4. 실제 통합 검증

다음 순서로 별도 테스트 주문을 사용해 확인한다.

1. Spring 결제를 완료해 gate token을 발급한다.
2. 게이트 장치가 좌·우 카메라 프레임과 토큰을 AI 서버에 업로드한다.
3. AI가 Spring 시작 API에서 barcode·qty를 수신하는지 확인한다.
4. `PASS`, `REVIEW`, `FLAG` 각각이 Spring 완료 API에 저장되는지 확인한다.
5. 잘못된 이미지·카메라 단절·추론 실패에서 Spring 토큰이 `FAILED`가 되는지 확인한다.
6. 완료 회신 응답 유실 상황에서 동일 완료 요청을 재시도해 멱등 처리되는지 확인한다.

## 운영 전 위험과 결정 필요 사항

| 우선순위 | 사항 | 영향 |
|---|---|---|
| 높음 | AI 공개 호스트의 게이트 장치 인증 방식 미구현 | Nginx가 AI 호스트를 외부에 제공하므로 장치 인증·네트워크 제한 정책이 필요 |
| 높음 | 실제 게이트 보정 없음 | 두 카메라의 같은 물체를 정확히 합칠 수 없음 |
| 높음 | 실제 상품 정확도 미측정 | 정상 카트 오탐과 미결제 상품 미탐을 운영 기준으로 판단할 수 없음 |
| 높음 | Jetson GPU 런타임 미검증 | CPU 폴백·의존성 불일치·처리 지연 가능성 |
| 중간 | 모델 파일 배포 방식 | Jenkins 호스트에서 `CARTGATE_MODEL_HOST_DIR`가 영속적이고 읽기 가능한 경로여야 함 |
| 중간 | 증거 이미지 저장 정책 | 현재 서버는 최종 판정만 Spring에 회신하며, 운영용 증거 저장소·보존 기간은 별도 결정 필요 |

## 최종 판정

- **코드 준비 상태:** 준비됨. API·Spring 계약·2대 카메라 입력 검증·패키징·CI/CD 골격이 구현되고 자동 테스트를 통과했다.
- **현재 Mac에서의 생산 서버 기동:** 준비되지 않음. Python 3.9.6, `torch`·`ultralytics` 부재, Secret 미설정 때문에 기동할 수 없다.
- **Jetson 운영 배포:** 아직 준비 완료로 판단할 수 없음. GPU 환경, 실제 보정, Spring 통합, 실카메라 정확도 검증을 완료해야 한다.

배포 실행 절차와 환경변수는 [`DEPLOYMENT.md`](DEPLOYMENT.md)를 따른다.
