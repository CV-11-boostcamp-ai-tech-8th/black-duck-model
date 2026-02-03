# TensorRT 변환 가이드

YOLOv26 모델을 TensorRT 형식으로 변환하는 방법입니다.

## 📋 사전 요구사항

- NVIDIA GPU (CUDA 지원)
- `ultralytics` 패키지 설치
- TensorRT 설치 (CUDA 환경에서 자동 설치됨)

## 🚀 사용법

### 1. Python 스크립트 사용

#### FP32 (기본 정밀도)
```bash
python export_tensorrt.py --precision fp32
```

#### FP16 (반정밀도, 속도↑)
```bash
python export_tensorrt.py --precision fp16
```

#### INT8 (양자화, 속도↑↑, 정확도↓)
```bash
python export_tensorrt.py --precision int8
```

### 2. 쉘 스크립트 사용

```bash
# FP32
bash export_tensorrt.sh fp32

# FP16
bash export_tensorrt.sh fp16

# INT8
bash export_tensorrt.sh int8
```

## ⚙️ 고급 옵션

```bash
python export_tensorrt.py \
    --model /path/to/model.pt \
    --precision fp16 \
    --imgsz 640 \
    --batch 8 \
    --workspace 4 \
    --data configs/yolo/vehicle_dataset.yaml
```

### 옵션 설명

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model` | `/data/ephemeral/home/shared_files/idx4_yolo26x.pt` | 변환할 모델 경로 |
| `--precision` | `fp32` | 정밀도 (`fp32`, `fp16`, `int8`) |
| `--imgsz` | `640` | 이미지 크기 |
| `--batch` | `8` | 최대 배치 크기 |
| `--workspace` | `4` | TensorRT 작업 공간 (GiB) |
| `--data` | `configs/yolo/vehicle_dataset.yaml` | INT8 캘리브레이션용 데이터셋 |

## 📊 정밀도별 비교

| 정밀도 | 속도 | 메모리 | 정확도 | 권장 사용 |
|--------|------|--------|--------|-----------|
| **FP32** | 기준 | 기준 | 기준 | 정확도 최우선 |
| **FP16** | ~2배 빠름 | ~50% 감소 | 거의 동일 | **권장 (균형)** |
| **INT8** | ~4배 빠름 | ~75% 감소 | 약간 감소 | 속도 최우선 |

## 🔍 INT8 변환 시 주의사항

1. **캘리브레이션 데이터 필수**: `--data` 옵션으로 데이터셋 지정
2. **충분한 이미지 수**: 최소 500장 권장 (NVIDIA 가이드라인)
3. **동일 장치에서 변환 및 실행**: 장치마다 캘리브레이션 결과가 다름
4. **첫 추론 시간 증가**: 첫 몇 번의 추론은 평소보다 느릴 수 있음

## 📁 출력 파일

변환 후 생성되는 파일:
- `idx4_yolo26x_fp32.engine` (FP32)
- `idx4_yolo26x_fp16.engine` (FP16)
- `idx4_yolo26x_int8.engine` (INT8)

## ⚠️ 문제 해결

### 메모리 부족 에러
```bash
# workspace 값을 줄이거나 None으로 설정
python export_tensorrt.py --precision fp16 --workspace 2
```

### 캘리브레이션 실패 (INT8)
```bash
# batch 크기를 줄이거나 imgsz를 줄임
python export_tensorrt.py --precision int8 --batch 4 --imgsz 384
```

## 📚 참고 자료

- [TensorRT 공식 문서](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Ultralytics Export 가이드](https://docs.ultralytics.com/modes/export/)
- [TensorRT INT8 캘리브레이션 가이드](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/Int8/MinMaxCalibrator.html)
