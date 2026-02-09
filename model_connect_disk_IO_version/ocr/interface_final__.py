from dataclasses import dataclass
from pathlib import Path

import re
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Optional
from collections import Counter

from paddleocr import PaddleOCR
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# YOLO 로거 비활성화
LOGGER.disabled = True


@dataclass
class LicensePlateResult:
    """번호판 검출 결과"""
    plate_image_path: str
    plate_text_path: str

class ModelManager:
    """전역 모델 인스턴스 관리"""
    _yolo_model = None
    _swinir_model = None
    _swinir_device = None
    _ocr_reader = None
    
    @classmethod
    def get_yolo_model(cls, model_path: str):
        """전역 YOLO 모델 반환"""
        if cls._yolo_model is None:
            cls._yolo_model = YOLO(model_path)
            print("YOLO model loaded successfully")
        return cls._yolo_model

    @classmethod
    def get_swinir_model(cls, model_path: str):
        """전역 SwinIR 모델 반환"""
        if cls._swinir_model is None:
            cls._swinir_model, cls._swinir_device = cls._load_swinir_model(model_path)
        return cls._swinir_model, cls._swinir_device
    
    # @classmethod
    # def get_ocr_reader(cls):
    #     """전역 OCR reader 반환"""
    #     if cls._ocr_reader is None:
    #         cls._ocr_reader = PaddleOCR(lang='korean', use_textline_orientation=False)
    #         print("PaddleOCR reader loaded successfully")
    #     return cls._ocr_reader
    
    # [Change]
    @classmethod
    def get_ocr_reader(cls):
        if cls._ocr_reader is None:
            import os

            # GPU 강제
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get(
                "CUDA_VISIBLE_DEVICES", "0"
            )

            # OneDNN 강제 차단 (CPU fallback 방지)
            os.environ["FLAGS_use_mkldnn"] = "0"

            cls._ocr_reader = PaddleOCR(
                lang="korean",
                use_textline_orientation=False,
            )
            print("PaddleOCR reader loaded successfully")

        return cls._ocr_reader
    
    @classmethod
    def _load_swinir_model(cls, model_path: str, device: str = 'cuda') -> tuple:
        """SwinIR 모델 로드"""
        try:
            from .models.network_swinir import SwinIR
            
            device = torch.device(device if torch.cuda.is_available() else 'cpu')
            
            # Classical SR x4 모델 설정
            model = SwinIR(
                upscale=4,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
                embed_dim=240,
                num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                mlp_ratio=2,
                upsampler="nearest+conv",
                resi_connection="3conv",
            )
            
            # 가중치 로드
            param_key_g = 'params_ema'
            pretrained_model = torch.load(model_path, map_location=device)
            model.load_state_dict(
                pretrained_model.get(param_key_g, pretrained_model),
                strict=True
            )
            
            model.eval()
            model = model.to(device)
            
            print(f"SwinIR model loaded successfully on {device}")
            return model, device
            
        except Exception as e:
            print(f"Failed to load SwinIR model: {e}")
            return None, None


class ImageEnhancer:
    """이미지 해상도 향상 클래스"""
    
    def __init__(self, window_size: int = 8):
        self.window_size = window_size
    
    def enhance_with_swinir(
        self, 
        image_bgr: np.ndarray, 
        model, 
        device
    ) -> np.ndarray:
        """SwinIR을 사용하여 번호판 해상도 향상"""
        try:
            # BGR to RGB
            img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            
            # numpy to tensor (HWC to CHW)
            img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            # Padding to window_size multiple
            img_tensor = self._pad_image(img_tensor)
            h_old, w_old = image_bgr.shape[:2]
            
            # Inference
            with torch.no_grad():
                output = model(img_tensor)
            
            # Remove padding and convert back
            output = output[..., :h_old * 4, :w_old * 4]
            output_bgr = self._tensor_to_bgr(output)
            
            return output_bgr
            
        except Exception as e:
            print(f"SwinIR enhancement failed: {e}")
            print("Using original image without enhancement")
            return image_bgr
    
    def _pad_image(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """이미지를 window_size의 배수로 패딩"""
        h_old, w_old = img_tensor.shape[2], img_tensor.shape[3]
        h_pad = (self.window_size - h_old % self.window_size) % self.window_size
        w_pad = (self.window_size - w_old % self.window_size) % self.window_size
        
        img_tensor = torch.cat([img_tensor, torch.flip(img_tensor, [2])], 2)[:, :, :h_old + h_pad, :]
        img_tensor = torch.cat([img_tensor, torch.flip(img_tensor, [3])], 3)[:, :, :, :w_old + w_pad]
        
        return img_tensor
    
    def _tensor_to_bgr(self, output: torch.Tensor) -> np.ndarray:
        """텐서를 BGR numpy array로 변환"""
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output_bgr


class FileManager:
    """파일 저장 관리 클래스"""
    
    @staticmethod
    def save_image(plate_image: np.ndarray, output_dir: Path) -> str:
        """번호판 이미지를 PNG 파일로 저장"""
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / "license_plate_image.png"
        
        # BGR to RGB 변환 후 저장
        if len(plate_image.shape) == 3 and plate_image.shape[2] == 3:
            plate_image_rgb = plate_image[:, :, ::-1]
            img = Image.fromarray(plate_image_rgb)
        else:
            img = Image.fromarray(plate_image)
        
        img.save(file_path, "PNG")
        return str(file_path)
    
    @staticmethod
    def save_text(plate_text: str, output_dir: Path) -> str:
        """번호판 텍스트를 텍스트 파일로 저장"""
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / "license_plate_text.txt"
        file_path.write_text(plate_text, encoding="utf-8")
        return str(file_path)


class LicensePlateDetector:
    """번호판 검출 및 OCR 클래스"""
    
    def __init__(
        self,
        yolo_model_path: str,
        swinir_model_path: str,
        use_enhancement: bool = True
    ):
        self.yolo_model_path = yolo_model_path
        self.swinir_model_path = swinir_model_path
        self.use_enhancement = use_enhancement
        self.enhancer = ImageEnhancer()
        self.file_manager = FileManager()
    
    def detect_and_recognize(
        self,
        vehicle_crops: list[np.ndarray],
        output_dir: Path
    ) -> Optional[LicensePlateResult]:
        """차량 crop 이미지들에서 번호판 검출 및 OCR"""
        
        if not vehicle_crops:
            print("No Vehicle Crop images!")
            return None
        
        # 1. 번호판 검출
        plate_crops = self._detect_plates(vehicle_crops)
        if not plate_crops:
            print("No plate crops images!")
            return None
        
        # 2. 이미지 해상도 향상 (선택적)
        if self.use_enhancement:
            plate_crops = self._enhance_plates(plate_crops)
        
        # 3. OCR 수행
        ocr_results = self._perform_ocr(plate_crops)
        if not ocr_results:
            print("No valid plates detected!")
            return None
        
        # 4. 최종 결과 선택 (최빈값)
        final_image, final_text = self._select_best_result(plate_crops, ocr_results)
        if final_image is None or not final_text:
            print("No final plate result!")
            return None
        
        # 5. 결과 저장
        plate_image_path = self.file_manager.save_image(final_image, output_dir)
        plate_text_path = self.file_manager.save_text(final_text, output_dir)
        
        return LicensePlateResult(plate_image_path, plate_text_path)
    
    def _detect_plates(self, vehicle_crops: list[np.ndarray]) -> list[np.ndarray]:
        """차량 이미지에서 번호판 영역 검출"""
        yolo_model = ModelManager.get_yolo_model(self.yolo_model_path)
        plate_crops = []
        
        for crop in vehicle_crops:
            results = yolo_model(
                source=crop,
                imgsz=640,
                max_det=1,
                save=False,
                save_crop=False,
                verbose=False
            )
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    orig_img = result.orig_img
                    boxes = result.boxes.xyxy.cpu().numpy()
                    
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        plate_crop = orig_img[y1:y2, x1:x2]
                        plate_crops.append(plate_crop)
        
        return plate_crops
    
    def _enhance_plates(self, plate_crops: list[np.ndarray]) -> list[np.ndarray]:
        """번호판 이미지 해상도 향상"""
        swinir_model, swinir_device = ModelManager.get_swinir_model(self.swinir_model_path)
        
        if swinir_model is None:
            print("SwinIR model not available, skipping enhancement")
            return plate_crops
        
        print(f"Enhancing {len(plate_crops)} plate images with SwinIR...")
        enhanced_crops = [
            self.enhancer.enhance_with_swinir(plate_img, swinir_model, swinir_device)
            for plate_img in plate_crops
        ]
        print("SwinIR enhancement completed")
        
        return enhanced_crops
    
    def _perform_ocr(self, plate_crops: list[np.ndarray]) -> dict[int, str]:
        """번호판 OCR 수행"""
        ocr_reader = ModelManager.get_ocr_reader()
        ocr_results = {}
        
        for idx, plate_img in enumerate(plate_crops):
            results = ocr_reader.ocr(plate_img)
            
            if not results or len(results) == 0:
                ocr_results[idx] = ""
                continue
            
            result = results[0]
            if not result or 'rec_texts' not in result or not result['rec_texts']:
                ocr_results[idx] = ""
                continue
            
            # 텍스트를 x 좌표 기준으로 정렬하여 결합
            plate_text = self._combine_texts(
                result['rec_texts'],
                result['rec_scores'],
                result['rec_polys']
            )
            ocr_results[idx] = plate_text
        
        return ocr_results
    
    def _combine_texts(
        self,
        rec_texts: list,
        rec_scores: list,
        rec_polys: list
    ) -> str:
        """OCR 결과 텍스트를 좌표 기준으로 정렬하여 결합"""
        plate_texts = []
        
        for text, conf, bbox in zip(rec_texts, rec_scores, rec_polys):
            # 텍스트 정규화
            text = text.replace(" ", "").upper()
            norm_text = re.sub(r"[^0-9가-힣]", "", text)
            if not norm_text:
                continue
            
            x_min = min([p[0] for p in bbox])
            plate_texts.append((x_min, norm_text, conf))
        
        # x 좌표 기준으로 정렬하여 결합
        plate_texts.sort(key=lambda x: x[0])
        return "".join([t[1] for t in plate_texts])
    
    def _select_best_result(
        self,
        plate_crops: list[np.ndarray],
        ocr_results: dict[int, str]
    ) -> tuple[Optional[np.ndarray], str]:
        """최빈값을 기준으로 최종 결과 선택"""
        valid_plates = [v for v in ocr_results.values() if v]
        
        if not valid_plates:
            return None, ""
        
        # 최빈값 찾기
        counter = Counter(valid_plates)
        final_text, _ = counter.most_common(1)[0]
        
        # 해당 텍스트를 가진 첫 번째 이미지 찾기
        final_image_idx = next(
            (idx for idx, text in ocr_results.items() if text == final_text),
            0
        )
        
        final_image = plate_crops[final_image_idx]
        return final_image, final_text


# 하위 호환성을 위한 레거시 함수
def detect_license_plate(
    vehicle_crops: list[np.ndarray],
    clip_output_dir: Path
) -> Optional[LicensePlateResult]:
    """
    차량 crop 이미지들에서 번호판 검출 및 OCR (레거시 함수)
    
    Args:
        vehicle_crops: 차량 crop 이미지 리스트 (numpy arrays, BGR format)
        clip_output_dir: 저장할 디렉토리
    
    Returns:
        번호판 결과 LicensePlateResult 객체 반환(없으면 None)
    """
    detector = LicensePlateDetector(
        yolo_model_path='/data/ephemeral/home/shared_files/plate/yolo26x_best.pt',
        swinir_model_path='/data/ephemeral/home/shared_files/plate/SwinIR-L.pth',
        use_enhancement=True
    )
    
    return detector.detect_and_recognize(vehicle_crops, clip_output_dir)
