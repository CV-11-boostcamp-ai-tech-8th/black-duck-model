from dataclasses import dataclass
from pathlib import Path

import os
import re
import cv2
import tempfile
from typing import Optional
from collections import Counter

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# YOLO 로거 비활성화
LOGGER.disabled = True


@dataclass
class LicensePlateResult:
    plate_image_path: str
    plate_text_path: str


def normalize_plate_text(text: str) -> str:
    """
    번호판 텍스트 정규화
    
    Args:
        text: OCR로 인식된 원본 텍스트
        
    Returns:
        정규화된 번호판 텍스트
    """
    text = text.replace(" ", "").upper()
    text = re.sub(r"[^0-9가-힣]", "", text)
    match = re.search(r"\d{2,3}[가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주하허호]\d{4}", text)
    text = match.group(0) if match else ""
    return text

def save_image(plate_image: np.ndarray, clip_output_dir: Path) -> str:
    """
    번호판 이미지를 PNG 파일로 저장
    
    Args:
        plate_image: 번호판 이미지 (numpy array, BGR 또는 RGB)
        clip_output_dir: 저장할 디렉토리
        
    Returns:
        저장된 파일 경로 (문자열)
    """
    # clip_output_dir 생성
    clip_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 파일명 결정
    filename = "license_plate_image.png"
    
    file_path = clip_output_dir / filename
    
    # numpy array를 PIL Image로 변환
    # BGR → RGB 변환 (OpenCV 사용 시)
    if len(plate_image.shape) == 3 and plate_image.shape[2] == 3:
        # BGR to RGB
        plate_image_rgb = plate_image[:, :, ::-1]
        img = Image.fromarray(plate_image_rgb)
    else:
        img = Image.fromarray(plate_image)
    
    # PNG로 저장
    img.save(file_path, "PNG")
    
    return str(file_path)

def save_text(plate_text: str, clip_output_dir: Path) -> str:
    """
    번호판 텍스트를 텍스트 파일로 저장
    
    Args:
        plate_text: 번호판 텍스트 (문자열)
        clip_output_dir: 저장할 디렉토리
        
    Returns:
        저장된 파일 경로 (문자열)
    """
    # clip_output_dir 생성
    clip_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 파일명 결정
    filename = "license_plate_text.txt"
    file_path = clip_output_dir / filename
    
    # 텍스트 파일로 저장
    file_path.write_text(plate_text, encoding="utf-8")
    
    return str(file_path)


def detect_license_plate(
    vehicle_crops: list[np.ndarray],
    clip_output_dir: Path
) -> LicensePlateResult | None:
    """
    차량 crop 이미지들에서 번호판 검출 및 OCR

    Returns:
        번호판 결과 LicensePlateResult 객체 반환(없으면 None)
    """

    if not vehicle_crops:
        return None

    # 1. 차량 이미지에서 번호판 검출
    model = YOLO('/data/ephemeral/home/shared_files/plate/yolo26x_best.pt')
        
    plate_crops = []
    
    # 임시 디렉토리에 이미지 저장
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        # 차량 crop 이미지들을 임시 디렉토리에 저장
        for idx, crop in enumerate(vehicle_crops):
            img_path = temp_dir_path / f"vehicle_{idx}.png"
            cv2.imwrite(str(img_path), crop)
        
        # YOLO로 번호판 검출
        results = model(
            source=str(temp_dir_path),
            imgsz=640,
            max_det=1,
            save=False,
            save_crop=False,
            verbose=False
        )
        
        # 검출된 번호판 crop
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                # 원본 이미지
                orig_img = result.orig_img
                
                # 검출된 박스 좌표 (xyxy format)
                boxes = result.boxes.xyxy.cpu().numpy()
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    # 번호판 영역 crop
                    plate_crop = orig_img[y1:y2, x1:x2]
                    plate_crops.append(plate_crop)
    
    if not plate_crops:
        return None
    
    # 2. 번호판 OCR 수행
    # plate_image, plate_text = ocr_plates(plate_crops, ocr_reader)
    # ocr_reader = PaddleOCR(lang='korean', use_textline_orientation=False, show_log=False)
    ocr_reader = PaddleOCR(lang='korean', use_textline_orientation=False)
    ocr_results = {}  # {image_index: plate_text}
    
    for idx, plate_img in enumerate(plate_crops):
        results = ocr_reader.ocr(plate_img)
        
        if not results or len(results) == 0:
            ocr_results[idx] = ""
            continue
        
        result = results[0]
        if not result or 'rec_texts' not in result or not result['rec_texts']:
            ocr_results[idx] = ""
            continue
        
        rec_texts = result['rec_texts']
        rec_scores = result['rec_scores']
        rec_polys = result['rec_polys']
        
        plate_texts = []
        for text, conf, bbox in zip(rec_texts, rec_scores, rec_polys):
            norm_text = normalize_plate_text(text)
            if norm_text == "":
                continue
            
            # bbox의 좌측 x 좌표 기준으로 정렬하기 위한 정보 저장
            x_min = min([p[0] for p in bbox])
            plate_texts.append((x_min, norm_text, conf))
        
        # x 좌표 기준으로 정렬하여 결합
        plate_texts.sort(key=lambda x: x[0])
        final_plate = "".join([t[1] for t in plate_texts])
        ocr_results[idx] = final_plate
    
    # 유효한 번호판 텍스트만 추출
    valid_plates = [v for v in ocr_results.values() if v != ""]
    
    if len(valid_plates) == 0:
        return None
    
    # 최빈값 찾기
    counter = Counter(valid_plates)
    final_text, freq = counter.most_common(1)[0]
    
    # 해당 텍스트를 가진 첫 번째 이미지 찾기
    final_image_idx = None
    for idx, text in ocr_results.items():
        if text == final_text:
            final_image_idx = idx
            break
    
    final_image = plate_crops[final_image_idx] if final_image_idx is not None else plate_crops[0]
    
    # None 체크
    if final_image is None or final_text is None or final_text == "":
        return None
    
    # 3. 이미지 저장
    plate_image_path = save_image(final_image, clip_output_dir)
    
    # 4. 텍스트 저장
    plate_text_path = save_text(final_text, clip_output_dir)
    
    # 5. 결과 반환
    license = LicensePlateResult(plate_image_path, plate_text_path)
    
    return license

