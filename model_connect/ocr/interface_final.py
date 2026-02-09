from dataclasses import dataclass
from pathlib import Path

import os
import re
import cv2
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
    """메모리 기반 번호판 결과"""
    plate_image: np.ndarray  # 메모리에 저장된 번호판 이미지
    plate_text: str  # 번호판 텍스트
    plate_image_path: str | None = None  # 디스크에 저장된 경우 경로
    plate_text_path: str | None = None  # 디스크에 저장된 경우 경로


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
    번호판 이미지를 PNG 파일로 저장 (필요시에만 사용)
    """
    clip_output_dir.mkdir(parents=True, exist_ok=True)
    filename = "license_plate_image.png"
    file_path = clip_output_dir / filename
    
    if len(plate_image.shape) == 3 and plate_image.shape[2] == 3:
        plate_image_rgb = plate_image[:, :, ::-1]
        img = Image.fromarray(plate_image_rgb)
    else:
        img = Image.fromarray(plate_image)
    
    img.save(file_path, "PNG")
    return str(file_path)

def save_text(plate_text: str, clip_output_dir: Path) -> str:
    """
    번호판 텍스트를 텍스트 파일로 저장 (필요시에만 사용)
    """
    clip_output_dir.mkdir(parents=True, exist_ok=True)
    filename = "license_plate_text.txt"
    file_path = clip_output_dir / filename
    file_path.write_text(plate_text, encoding="utf-8")
    return str(file_path)


def detect_license_plate(
    vehicle_crops: list[np.ndarray],
    clip_output_dir: Path | None = None,
    save_to_disk: bool = False
) -> LicensePlateResult | None:
    """
    차량 crop 이미지들에서 번호판 검출 및 OCR (메모리 기반)
    
    Args:
        vehicle_crops: 차량 crop 이미지 리스트 (numpy array)
        clip_output_dir: 저장할 디렉토리 (save_to_disk=True일 때만 사용)
        save_to_disk: 디스크에 저장할지 여부 (기본값: False)
    
    Returns:
        LicensePlateResult 객체 (메모리에 저장된 이미지와 텍스트 포함)
    """
    if not vehicle_crops:
        print("No Vehicle Crop images!!!!")
        return None

    # 1. 차량 이미지에서 번호판 검출
    model = YOLO('/data/ephemeral/home/shared_files/plate/yolo26x_best.pt')
        
    plate_crops = []
    
    # 각 차량 crop 이미지를 직접 YOLO에 전달 (임시 파일 저장 없이)
    for crop in vehicle_crops:
        # YOLO로 번호판 검출 (numpy array 직접 전달)
        results = model(
            source=crop,
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
        print("No plate crops images!!!!")
        return None
    
    # 2. 번호판 OCR 수행
    ocr_reader = PaddleOCR(lang='korean', use_textline_orientation=False)
    ocr_results = {}  # {image_index: plate_text}
    original_texts = {}  # 원본 텍스트 저장 (정규화 실패 시 사용)
    
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
        original_candidates = []
        
        for text, conf, bbox in zip(rec_texts, rec_scores, rec_polys):
            norm_text = normalize_plate_text(text)
            original_candidates.append((text.strip(), conf))
            
            if norm_text == "":
                continue
            
            # bbox의 좌측 x 좌표 기준으로 정렬하기 위한 정보 저장
            x_min = min([p[0] for p in bbox])
            plate_texts.append((x_min, norm_text, conf))
        
        # x 좌표 기준으로 정렬하여 결합
        if plate_texts:
            plate_texts.sort(key=lambda x: x[0])
            final_plate = "".join([t[1] for t in plate_texts])
            ocr_results[idx] = final_plate
        else:
            # 정규화 실패 시 원본 텍스트 중 confidence가 가장 높은 것 사용
            if original_candidates:
                original_candidates.sort(key=lambda x: x[1], reverse=True)
                ocr_results[idx] = original_candidates[0][0]
            else:
                ocr_results[idx] = ""
    
    # 유효한 번호판 텍스트만 추출
    valid_plates = [v for v in ocr_results.values() if v != ""]
    
    if len(valid_plates) == 0:
        print("[warning] 정규화된 번호판이 없지만, 원본 OCR 결과를 사용합니다.")
        # ocr_results에서 빈 문자열이 아닌 것 찾기
        non_empty_results = {k: v for k, v in ocr_results.items() if v != ""}
        if len(non_empty_results) > 0:
            # 첫 번째 비어있지 않은 결과 사용
            final_text = list(non_empty_results.values())[0]
            final_image_idx = list(non_empty_results.keys())[0]
            final_image = plate_crops[final_image_idx]
        else:
            print("No valid plates images!!!!")
            return None
    else:
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
        print("No final plate check!!!!")
        return None
    
    # 디스크에 저장 (필요시에만)
    plate_image_path = None
    plate_text_path = None
    
    if save_to_disk and clip_output_dir:
        plate_image_path = save_image(final_image, clip_output_dir)
        plate_text_path = save_text(final_text, clip_output_dir)
    
    # 메모리 기반 결과 반환
    license = LicensePlateResult(
        plate_image=final_image,  # 메모리에 저장
        plate_text=final_text,  # 메모리에 저장
        plate_image_path=plate_image_path,  # 디스크 경로 (save_to_disk=True일 때만)
        plate_text_path=plate_text_path  # 디스크 경로 (save_to_disk=True일 때만)
    )
    
    return license
