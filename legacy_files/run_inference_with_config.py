#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference Runner with YAML Config
- YAML 설정 파일을 읽어 inference.py를 실행
"""

import argparse
import yaml
from pathlib import Path

# inference 모듈 import
import inference


def load_config(config_path):
    """YAML 설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Run inference with YAML config')
    parser.add_argument('--config', type=str, required=True,
                        help='YAML 설정 파일 경로')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # inference 모듈의 전역 변수 설정
    inference.MODE = "VIDEO"
    inference.VIDEO_PATH = config['video_path']
    inference.VERSION = config['version']
    inference.MODEL_WEIGHT = config['model_weight']
    inference.CONF_THRESHOLD = config['conf_threshold']
    inference.IOU_THRESHOLD = config['iou_threshold']
    inference.CONVERT_AVI_TO_MP4 = config['convert_avi_to_mp4']
    inference.INFERENCE_NAME_PREFIX = f"yolo26x_{config['version']}"
    
    # inference 실행
    inference.main()


if __name__ == '__main__':
    main()

