#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track Runner with YAML Config
- YAML 설정 파일을 읽어 track.py를 실행
"""

import argparse
import yaml
from pathlib import Path

# track 모듈 import
import track


def load_config(config_path):
    """YAML 설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Run tracking with YAML config')
    parser.add_argument('--config', type=str, required=True,
                        help='YAML 설정 파일 경로')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # track 모듈의 전역 변수 설정
    track.VIDEO_PATH = config['video_path']
    track.VERSION = config['version']
    track.MODEL_WEIGHT = config['model_weight']
    track.CONF_THRESHOLD = config['conf_threshold']
    track.IOU_THRESHOLD = config['iou_threshold']
    track.TRACKER_TYPE = config['tracker_type']
    track.CONVERT_AVI_TO_MP4 = config['convert_avi_to_mp4']
    track.TRACK_NAME_PREFIX = f"yolo26x_{config['version']}"
    
    # track 실행
    track.main()


if __name__ == '__main__':
    main()

