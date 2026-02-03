#!/bin/bash

# TensorRT 변환 스크립트
# 사용법: bash export_tensorrt.sh [fp32|fp16|int8]

MODEL_PATH="/data/ephemeral/home/shared_files/idx4_yolo26x.pt"
PRECISION=${1:-fp32}  # 기본값: fp32

echo "========================================================================"
echo "YOLOv26 TensorRT 변환 스크립트"
echo "========================================================================"
echo "모델: $MODEL_PATH"
echo "정밀도: $PRECISION"
echo "========================================================================"
echo ""

# 변환 실행
python export_tensorrt.py \
    --model "$MODEL_PATH" \
    --precision "$PRECISION" \
    --imgsz 640 \
    --batch 8 \
    --workspace 4 \
    --data "configs/yolo/vehicle_dataset.yaml"

echo ""
echo "========================================================================"
echo "변환 완료!"
echo "========================================================================"
