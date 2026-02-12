# 🚗 3D 급정거 감지 시스템 (Brake Detection System)

차량 영상에서 전방 차량의 급정거 이벤트를 실시간으로 감지하고 분석하는 시스템입니다.

## 📋 목차
- [개요](#개요)
- [주요 기능](#주요-기능)
- [시스템 요구사항](#시스템-요구사항)
- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
- [프로젝트 구조](#프로젝트-구조)
- [파일별 설명](#파일별-설명)

## 개요

이 시스템은 YOLO 객체 탐지 모델과 ByteTrack 추적 알고리즘을 활용하여 차량 영상에서 급정거 상황을 자동으로 감지합니다. 물리 기반 거리 추정과 로지스틱 회귀 모델을 결합하여 다음 네 가지 유형의 경고를 제공합니다:

- **NORMAL**: 정상 주행
- **HARD_BRAKE**: 일반적인 급정거 (가속도 기반)
- **CRITICAL**: 충돌 임박 상황 (TTC 기반)
- **CUT_IN**: 끼어들기 후 감속

## 주요 기능

### 1. 거리 추정
- 핀홀 카메라 모델을 사용한 3D 거리 계산
- 실시간 속도 및 가속도 추정
- TTC (Time To Collision) 계산

### 2. 급정거 감지
- AI 기반 위험도 스코어링 (로지스틱 회귀)
- 다중 필터링 (차선 내 차량, 원거리 보정, 측면 이동 보정)
- 끼어들기 차량 특수 처리

### 3. 이벤트 관리
- 프레임 단위 추적 및 기록
- 이벤트 병합 및 요약
- 자동 클립 생성 (이벤트 전후 구간 포함)

### 4. 시각화
- 실시간 bbox 표시 (정상: 초록색, 급정거: 빨간색)
- 프레임별 상태 정보 오버레이
- MP4 형식 출력 영상

## 시스템 요구사항

### 필수 패키지
```
python >= 3.8
opencv-python >= 4.5.0
torch >= 1.9.0
ultralytics >= 8.0.0
numpy >= 1.19.0
```

## 설치 방법

1. 저장소 클론
```bash
git clone <repository-url>
cd brake-detection-system
```

2. 필요한 패키지 설치
```bash
pip install opencv-python torch ultralytics numpy
```

3. FFmpeg 설치 (MP4 변환용)
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# https://ffmpeg.org/download.html 에서 다운로드
```

## 사용 방법

### 1. 기본 실행 (프레임별 로그 + 이벤트 요약)

```bash
python main.py
```

**출력:**
```
runs/final_result/
├── clip_1/
│   ├── clip_1.mp4          # 이벤트 영상 (전후 5-2초)
│   ├── thumbnail.jpg       # 이벤트 시작 프레임
│   └── crops/
│       ├── crop_00.jpg     # 최고 품질 crop
│       ├── crop_01.jpg
│       └── ... (최대 60장)
├── clip_2/
│   └── ...
```

### 전체 영상 시각화 + 로그

```bash
python main_draw_log.py
```

**출력:**
```
runs/final_result/
├── {video_name}_annotated.mp4      # 전체 영상 (bbox 표시)
├── {video_name}_frame_log.csv      # 프레임별 데이터
└── {video_name}_event_summary.csv  # 이벤트 요약
```

### 모드 3: 상세 이벤트 클립 (IoU 매칭)

```bash
python main_clip.py
```

**출력:**
```
runs/final_result/
├── event_1_tid_3.mp4       # 이벤트 1 (TID 3)
├── event_2_tid_5.mp4       # 이벤트 2 (TID 5)
└── ...
```

**클립 구성:**
- 이벤트 발생 4초 전부터 시작
- 이벤트 종료 4초 후까지 포함
- "BRAKE!" 라벨 표시 (빨간색)
- "[BRAKING]" 상태 표시
- 프레임 번호 및 이벤트 ID 표시

### 3. 설정 변경

`config.py` 파일 수정:

```python
# 비디오 경로 설정
VIDEO_PATH = "/path/to/your/video.mp4"

# 모델 및 트래커 설정
MODEL_WEIGHT = "/path/to/yolo/weights.pt"
TRACKER_YAML = "/path/to/bytetrack.yaml"

# 출력 디렉토리
SAVE_DIR = "runs/final_result"

# 탐지 임계값 조정
CONF_THRESHOLD = 0.5  # YOLO 신뢰도
IOU_THRESHOLD = 0.5   # NMS IOU

# 카메라 파라미터
FOCAL_LENGTH = 1200   # 초점거리 (픽셀)
CAR_REAL_WIDTH = 1.8  # 차량 실제 너비 (미터)
```

## 프로젝트 구조

```
.
├── config.py              # 설정 파일
├── brake_detector.py      # 급정거 감지 로직
├── utils.py               # 유틸리티 함수
├── main.py                # 메인 실행 스크립트
├── main_draw.py           # 이벤트 클립 생성 스크립트
└── README.md              # 프로젝트 문서
```

### 파일별 설명

#### `config.py`
시스템 전반의 설정값 관리:
- 경로 설정 (비디오, 모델, 출력)
- YOLO 파라미터
- 카메라 캘리브레이션 값
- 경고 타입 정의

#### `brake_detector.py`
핵심 감지 알고리즘:
- **BrakeDetector 클래스**: 차량 추적 및 급정거 판단
- 거리/속도/가속도 계산
- AI 위험도 스코어링
- 이벤트 생성 및 관리

#### `utils.py`
보조 기능:
- AVI → MP4 변환
- 이벤트 클립 저장
- Crop 이미지 추출