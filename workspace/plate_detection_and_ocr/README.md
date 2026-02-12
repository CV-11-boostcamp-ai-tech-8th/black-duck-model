# 🚙 License Plate Detection & OCR Pipeline

YOLO 기반 번호판 검출과 OCR을 결합한 번호판 인식 시스템입니다.  
Super-Resolution(SwinIR)을 통해 저해상도 번호판의 인식 성능을 향상시켰습니다.


## Project Overview

본 프로젝트는 차량 이미지 또는 영상 프레임을 입력으로 받아 다음 단계를 수행합니다:

1. **YOLO 기반 번호판 Detection**
2. **SwinIR 기반 Super-Resolution**
3. **PaddleOCR 기반 문자 인식**
4. **모델 통합 및 파이프라인 처리** (별도 폴더)


## Project Structure

```
workspace/plate_detection_and_ocr
├── for_model_connect            # 모델 통합 로직
|   ├── __init__.py
|   ├── interface_final.py       # Detection + OCR 통합 코드
|   └── models                   # 모델 정의
|       ├── __init__.py
|       └── network_swinir.py    # SwinIR 관련 모듈
├── README.md                    # 프로젝트 문서
├── models                       # 모델 정의
|   ├── __init__.py
|   └── network_swinir.py        # SwinIR 관련 모듈
├── datasets                     # 데이터셋 폴더
├── weights                      # 가중치 폴더
├── carplate.yaml                # YOLO 학습 설정 파일
├── train.py                     # YOLO 학습 스크립트
├── test.py                      # YOLO 테스트 스크립트
├── inference.py                 # YOLO 추론 스크립트
├── apply_sr.py                  # SwinIR Super-Resolution 적용
└── predict_plates.py            # 번호판 OCR 스크립트
```


## How to Run

### Train
```bash
python train.py
```

### Test
```bash
python test.py
```

### Apply Super-Resolution
```bash
python apply_sr.py
```

### Predict plate number
```bash
python predict_plates.py
```


## References
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [SwinIR](https://github.com/JingyunLiang/SwinIR)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

