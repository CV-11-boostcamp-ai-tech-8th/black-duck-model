# config.py

import os

# [경로 설정]
# 테스트할 비디오 경로를 여기에 지정하세요
VIDEO_PATH = "/data/ephemeral/home/dataset/급정거_2.mp4" 
MODEL_WEIGHT = "/data/ephemeral/home/shared_files/idx20_yolo26x.pt"
TRACKER_YAML = "/data/ephemeral/home/shared_files/bytetrack.yaml"
SAVE_DIR = "runs/final_result"

# [YOLO 설정]
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# [카메라 및 거리 추정]
FOCAL_LENGTH = 1200 
CAR_REAL_WIDTH = 1.8

# [공간 필터링]
MAX_DETECT_DIST = 80.0
MIN_BBOX_SIZE = 30

WARN_NORMAL = 0
WARN_HARD_BRAKE = 1     # 일반적인 급정거 (가속도 기반)
WARN_TTC_CRITICAL = 2   # 충돌 임박 (TTC 기반)
WARN_CUT_IN = 3         # 끼어들기 감속

# [설정값 요약]
# 차선 폭: 2.5m (걸친 차 허용)
# 반응 속도: 3프레임
# 리셋 로직: TTC 안전장치 포함