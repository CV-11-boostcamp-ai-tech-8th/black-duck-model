#!/usr/bin/env python3
"""
connect_final.py의 run_model() 테스트용 스크립트

기능:
- /data/ephemeral/home/model_connect/temp/crops 경로의 이미지 파일들을 불러옴
- numpy array 리스트로 변환
- ocr.detect_license_plate() 호출
- 결과를 /data/ephemeral/home/model_connect/temp/output에 저장
"""

from pathlib import Path
import cv2
import numpy as np

# OCR 모듈 import
try:
    from ocr.interface_final import detect_license_plate
except ImportError:
    # ocr 폴더가 모듈 경로에 없을 경우
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ocr.interface_final import detect_license_plate


def load_crop_images(crops_dir: Path) -> list[np.ndarray]:
    """
    crops 디렉토리에서 이미지 파일들을 불러와서 numpy array 리스트로 반환
    
    Args:
        crops_dir: crops 이미지가 있는 디렉토리
        
    Returns:
        numpy array 리스트 (BGR 형식)
    """
    crop_images = []
    
    # 이미지 파일 찾기 (jpg, jpeg, png 등)
    image_files = sorted(crops_dir.glob("*.jpg")) + \
                  sorted(crops_dir.glob("*.jpeg")) + \
                  sorted(crops_dir.glob("*.png"))
    
    if not image_files:
        print(f"[warning] {crops_dir}에 이미지 파일이 없습니다.")
        return []
    
    print(f"[info] {len(image_files)}개의 이미지 파일을 찾았습니다.")
    
    for img_path in image_files:
        try:
            # OpenCV로 이미지 읽기 (BGR 형식)
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[warning] {img_path}를 읽을 수 없습니다.")
                continue
            crop_images.append(img)
            print(f"[info] {img_path.name} 로드 완료 (shape: {img.shape})")
        except Exception as e:
            print(f"[error] {img_path} 로드 실패: {e}")
            continue
    
    return crop_images


def main():
    # 경로 설정
    base_dir = Path("/data/ephemeral/home/tmp_disk")
    crops_dir = base_dir / "CarCrop_frames"
    output_dir = base_dir / "outputs"
    
    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("OCR 테스트 시작")
    print("=" * 60)
    print(f"[crops_dir] {crops_dir}")
    print(f"[output_dir] {output_dir}")
    print()
    
    # 1. crops 이미지 로드
    print("[1단계] 이미지 파일 로드 중...")
    vehicle_crops = load_crop_images(crops_dir)
    
    if not vehicle_crops:
        print("[error] 로드된 이미지가 없습니다.")
        return
    
    print(f"[info] 총 {len(vehicle_crops)}개의 이미지를 로드했습니다.")
    print()
    
    # 2. OCR 수행
    print("[2단계] OCR 수행 중...")
    try:
        plate_result = detect_license_plate(
            vehicle_crops=vehicle_crops,
            clip_output_dir=output_dir,
        )
    except Exception as e:
        print(f"[error] OCR 수행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 결과 확인
    print()
    print("[3단계] 결과 확인")
    print("=" * 60)
    
    if plate_result is None:
        print("[result] 번호판을 감지하지 못했습니다.")
        return
    
    print(f"[result] plate_image_path: {plate_result.plate_image_path}")
    print(f"[result] plate_text_path: {plate_result.plate_text_path}")
    
    # 텍스트 파일 읽기
    plate_text_path = Path(plate_result.plate_text_path)
    if plate_text_path.exists():
        plate_text = plate_text_path.read_text(encoding="utf-8").strip()
        print(f"[result] plate_text: {plate_text}")
    else:
        print(f"[warning] plate_text_path가 존재하지 않습니다: {plate_text_path}")
    
    # 이미지 파일 확인
    plate_image_path = Path(plate_result.plate_image_path)
    if plate_image_path.exists():
        print(f"[result] plate_image 저장 완료: {plate_image_path}")
        print(f"[result] 파일 크기: {plate_image_path.stat().st_size / 1024:.2f} KB")
    else:
        print(f"[warning] plate_image_path가 존재하지 않습니다: {plate_image_path}")
    
    print()
    print("=" * 60)
    print("테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

