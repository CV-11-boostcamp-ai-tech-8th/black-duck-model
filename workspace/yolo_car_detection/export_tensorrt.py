"""
YOLOv26 모델을 TensorRT 형식으로 변환하는 스크립트

사용법:
    python export_tensorrt.py --precision fp32
    python export_tensorrt.py --precision fp16
    python export_tensorrt.py --precision int8
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def export_to_tensorrt(
    model_path: str,
    precision: str = "fp32",
    imgsz: int = 640,
    batch: int = 8,
    workspace: int = 4,
    dynamic: bool = True,
    data: str = None,
):
    """
    YOLO 모델을 TensorRT 형식으로 변환
    
    Args:
        model_path: 입력 모델 경로 (.pt)
        precision: 정밀도 (fp32, fp16, int8)
        imgsz: 이미지 크기
        batch: 배치 크기
        workspace: TensorRT 작업 공간 크기 (GiB)
        dynamic: 동적 입력 크기 허용 여부
        data: INT8 캘리브레이션용 데이터셋 경로 (.yaml)
    """
    
    print("=" * 70)
    print("TensorRT 변환 시작")
    print("=" * 70)
    
    # 모델 로드
    print(f"\n📦 모델 로드 중: {model_path}")
    model = YOLO(model_path)
    
    # 출력 파일명 설정
    model_path_obj = Path(model_path)
    output_name = model_path_obj.stem + f"_{precision}.engine"
    
    print(f"\n⚙️  변환 설정:")
    print(f"   - 정밀도: {precision.upper()}")
    print(f"   - 이미지 크기: {imgsz}")
    print(f"   - 배치 크기: {batch}")
    print(f"   - 작업 공간: {workspace} GiB")
    print(f"   - 동적 크기: {dynamic}")
    
    # TensorRT 변환 인자 설정
    export_args = {
        "format": "engine",
        "imgsz": imgsz,
        "batch": batch,
        "workspace": workspace,
        "dynamic": dynamic,
        "verbose": True,
    }
    
    # 정밀도별 설정
    if precision == "fp16":
        export_args["half"] = True
        print(f"   - FP16: 활성화")
    elif precision == "int8":
        export_args["int8"] = True
        if data:
            export_args["data"] = data
            print(f"   - INT8: 활성화")
            print(f"   - 캘리브레이션 데이터: {data}")
        else:
            # INT8의 경우 데이터셋 필수
            print(f"   - INT8: 활성화")
            print(f"   - 캘리브레이션 데이터: 기본값 사용")
    
    # 변환 실행
    print(f"\n🚀 TensorRT 변환 중... (시간이 다소 걸릴 수 있습니다)")
    print("=" * 70)
    
    try:
        output_path = model.export(**export_args)
        
        print("=" * 70)
        print("✅ 변환 완료!")
        print("=" * 70)
        print(f"\n📁 출력 파일: {output_path}")
        
        # 파일 크기 확인
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
            print(f"📊 파일 크기: {file_size:.2f} MB")
        
        return output_path
        
    except Exception as e:
        print("=" * 70)
        print("❌ 변환 실패!")
        print("=" * 70)
        print(f"에러: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="YOLO 모델을 TensorRT 형식으로 변환"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="/data/ephemeral/home/shared_files/idx4_yolo26x.pt",
        help="변환할 모델 경로 (.pt)",
    )
    
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "int8"],
        help="정밀도 선택 (fp32, fp16, int8)",
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="이미지 크기 (기본: 640)",
    )
    
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="배치 크기 (기본: 8)",
    )
    
    parser.add_argument(
        "--workspace",
        type=int,
        default=4,
        help="TensorRT 작업 공간 크기 in GiB (기본: 4)",
    )
    
    parser.add_argument(
        "--dynamic",
        action="store_true",
        default=True,
        help="동적 입력 크기 허용 (기본: True)",
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="configs/yolo/vehicle_dataset.yaml",
        help="INT8 캘리브레이션용 데이터셋 (.yaml)",
    )
    
    args = parser.parse_args()
    
    # 변환 실행
    export_to_tensorrt(
        model_path=args.model,
        precision=args.precision,
        imgsz=args.imgsz,
        batch=args.batch,
        workspace=args.workspace,
        dynamic=args.dynamic,
        data=args.data if args.precision == "int8" else None,
    )


if __name__ == "__main__":
    main()
