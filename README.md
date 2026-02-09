# black-duck-model

## Quick Start
### 1. Prerequisites
```bash
(작성 예정)
```
### 2. Run
- 우선, `black-duck-web`의 backend 서버가 실행되어 있어야 함.
```bash
cd model_connect
python connect_final.py \
  --job <PATH_TO_JOB_JSON> \
  [--bucket GCS_BUCKET_NAME] \
  [--expire DEFAULT_SIGNED_URL_EXPIRE_SECONDS]
```
- `--job <path>` (required) : job json 경로
- `--bucket <name>` (optional) : GCS bucket 이름
- `--expire <seconds>` (optional, default=3600) : signed URL 만료 시간

## Model Architecture


## Project Structure
```bash
model_connect/ (Root)
├── ocr/                   # 이미지 처리 및 문자 인식 패키지
│   ├── models/            # AI 모델 아키텍처 정의
│   ├── __init__.py
│   └── network_swinir.py  # SwinIR(이미지 고해상도화) 모델
├── interface_final.py     # SwinIR + OCR 통합 인터페이스
├── sudden_stop/           # 급제동 감지 모듈
│   ├── __init__.py
│   ├── brake_detector.py  # 감속 및 브레이크 감지 알고리즘
│   └── config.py          # 감지 임계값 설정
├── main.py                # 모듈 실행 메인
├── utils.py               # 유틸리티 도구
├── scripts/               # 실행 및 환경별 테스트 스크립트
│   ├── connect_final.py   # 스크립트 기반 실행 파일
│   └── test_all_videos.sh
├── connect_final_sudden_*.py
├── connect_test.py
└── connect_final.py       # 최상위 통합 실행 엔트리 포인트
```

### 주요 폴더 설명


## 만든 사람
표(web 파트 참고) - kmj