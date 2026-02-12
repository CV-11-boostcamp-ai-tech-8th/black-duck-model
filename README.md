# black-duck-model
>[블랙박스 영상 기반 교통 법규 위반 이벤트 탐지 서비스 "Black Ori"](https://github.com/boostcampaitech8/pro-cv-finalproject-cv-11/)의 model submodule입니다.

## Quick Start
### 1. Prerequisites
- versions
```bash
(작성 예정)
```
- 가상환경 설정부터 시작할 경우:
```bash
python -m venv py310
source py310/bin/activate
source requirements.sh
```
### 2. Run
- 우선, `black-duck-web`의 backend 서버가 실행되어 있어야 함.
- 기본적으로는 백엔드 서버가 해당 코드를 실행하므로, 사용자가 명시적으로 해당 코드를 실행할 일은 없음.
- 그러나 job.json이 로컬에 있다는 가정 하에, 아래 코드들을 실행해 직접 테스트할 수 있음.
## 1. memory version
- 교통위반 clip video, 번호판 crop image, 번호판 ocr result 등이 local disk에 저장되지 않고, 변수로 memory 상에서만 저장됨.
### 1.1. GCS(google cloud storage) version
- 기본 version, 원본 video를 GCS에서 다운로드받고, 결과 clip video를 GCS 업로드함.
- 이를 위해 인자로 버킷 명 전달이 필요.
```bash
cd model_connect
python connect_final.py \
  --job <PATH_TO_JOB_JSON> \
  [--bucket GCS_BUCKET_NAME] \
  [--expire DEFAULT_SIGNED_URL_EXPIRE_SECONDS]
```
- `--job <path>` (required) : job json 경로
- `[--bucket name]` (optional) : GCS bucket 이름, 명시되지 않았을 경우 job.json의 input_url으로 추론함.
- `[--expire seconds]` (optional, default=3600) : signed URL 만료 시간

### 1.2. local version
- 테스트용 version, GCS와 통신하지 않음. local에 저장된 video를 사용하고, 결과 clip video를 GCS 업로드하지 않음.
- 이를 위해 인자로 video 경로 전달이 필요
```bash
python connect_local_final.py \
  --job <PATH_TO_JOB_JSON> \
  [--video local_video_path]
```
- `[--video path]` (optional) : 테스트를 위해 사용할 local video path

## 2. disk-IO version
- clip video, ocr crop image와 같은 결과들을 실제로 local disk에 저장함.
- 개발 과정에서 실제 결과들을 바로바로 확인하기 위해 분리.
- 실행 방법은 memory version과 동일.

## Model Architecture


## Project Structure
```bash
model_connect/ (Root)
├── ocr/                   # 이미지 처리 및 문자 인식 패키지
│   ├── __init__.py
│   ├── network_swinir.py  # SwinIR(이미지 고해상도화) 모델
│   └──interface_final.py  # SwinIR + OCR 통합 인터페이스
│
├── sudden_stop/           # 급정거 감지 패키지
│   ├── __init__.py
│   ├── brake_detector.py  # 감속 및 브레이크 감지 알고리즘
│   ├── config.py          # 감지 임계값 설정
│   ├──main.py             # 패키지 실행 메인
│   └──utils.py            # 유틸리티 도구
│
├── models/                # 사용한 가중치 파일들
│   ├── car_detection/     # 차량 Detection yolo26x 가중치   
│   ├── plate_detection/   # 번호판 Detection yolo26x 가중치
│   └── plate_ocr/         # 번호판 ocr swinIR 가중치치
│
├── scripts/               # 테스트용 임시 스크립트 파일들
├── connect_local_test.py  # 로컬 테스트용 실행행 스크립트
├── connect_final.py       # 최상위 통합 실행 스크립트
│
└── workspace/             # 개별 모델/알고리즘 학습 및 실험 workspace
    ├── yolo_car_detection/                # 차량 Detection 실험 workspace
    ├── sudden_stop_detection_algorithm/   # 급정거 감지 알고리즘 실험 workspace
    └── plate_detection_and_ocr/           # 번호판 Detection 및 ocr 실험 workspace
``` 

## Team CV-11 Model-part Members
| Model & Dataset | Algorithm & Dataset | Model & Dataset |
|:---:|:---:|:---:|
| <img src="https://github.com/M1niJ.png" width="120"> | <img src="https://github.com/uss0302-cmd.png" width="120"> | <img src="https://github.com/cuffyluv.png" width="120"> |
| [김민진](https://github.com/M1niJ) | [류제윤](https://github.com/uss0302-cmd) | [주상우](https://github.com/cuffyluv) |
| minjin0313b@gmail.com | uss0302@gmail.com | cuffyluv.1@gmail.com |
| Detection & OCR | Sudden Brake Algorithm | Detection & Tracking |

