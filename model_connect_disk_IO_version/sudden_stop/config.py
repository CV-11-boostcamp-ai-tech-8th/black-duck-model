# config.py

import os

# [경로 설정]
VIDEO_PATH = "/data/ephemeral/home/dataset/20260115-11h38m24s_N.avi" 
MODEL_WEIGHT = "/data/ephemeral/home/black-duck-model/models/car_detection/idx20_yolo26x.pt"
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
WARN_HARD_BRAKE = 1   
WARN_TTC_CRITICAL = 2   
WARN_CUT_IN = 3         