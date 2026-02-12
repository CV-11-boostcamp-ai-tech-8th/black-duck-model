import os
import re
import cv2
import json
import warnings

from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
from typing import Dict

warnings.filterwarnings("ignore")


def load_images(folder):
    """Load image file paths from folder"""
    imgs = []
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            imgs.append(os.path.join(folder, f))
    return imgs

def normalize_remove_special_chars(text):
    """
    Remove special characters: keep only numbers and Korean characters
    """
    text = text.replace(" ", "").upper()
    text = re.sub(r"[^0-9가-힣]", "", text)
    return text

def run_ocr_inference(image_dir: str, 
                       output_dir: str = "./ocr_results",
                       visualize: bool = True,
                       save_txt: bool = True):
    """Run PaddleOCR inference on images"""
    os.makedirs(output_dir, exist_ok=True)
    
    if visualize:
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
    
    if save_txt:
        txt_dir = os.path.join(output_dir, "predictions")
        os.makedirs(txt_dir, exist_ok=True)
    
    # Load images
    print("Loading images...")
    image_paths = sorted(load_images(image_dir))
    print(f"Found {len(image_paths)} images")
    
    # Initialize PaddleOCR
    print("Initializing PaddleOCR...")
    reader = PaddleOCR(lang='korean', use_textline_orientation=False)
    
    # Load Korean font
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
            size=24
        )
    except:
        print("Warning: Korean font not found. Using default font.")
        font = ImageFont.load_default()
    
    # Run OCR inference
    print("\nRunning OCR inference...")
    predictions = {}
    
    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        img_cv = cv2.imread(img_path)
        
        if img_cv is None:
            predictions[img_name] = ""
            continue
        
        results = reader.ocr(img_cv)
        
        # Handle no detection case
        if not results or len(results) == 0:
            predictions[img_name] = ""
            
            # Visualize: No detection
            if visualize:
                img_pil = Image.open(img_path).convert("RGB")
                draw = ImageDraw.Draw(img_pil)
                draw.text((10, 10), "No Detection", font=font, fill=(255, 0, 0))
                
                save_path = os.path.join(vis_dir, img_name)
                img_pil.save(save_path)
            
            continue
        
        result = results[0]
        if 'rec_texts' not in result or not result['rec_texts']:
            predictions[img_name] = ""
            continue
        
        # Visualize preparation
        if visualize:
            img_pil = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img_pil)
        
        plate_texts = []
        confidences = []

        for line in result:
            bbox = line[0]
            text_info = line[1]
            text = text_info[0]
            confidence = text_info[1]

            norm_text = normalize_remove_special_chars(text)
            if norm_text == "":
                continue
            
            x_min = min([p[0] for p in bbox])
            plate_texts.append((x_min, norm_text))
            confidences.append(confidence)
            
            # Visualize: bounding box and text
            if visualize:
                bbox_pts = [(int(x), int(y)) for x, y in bbox]
                draw.line(bbox_pts + [bbox_pts[0]], fill=(0, 255, 0), width=2)
                
                # Display individual detection text
                x, y = bbox_pts[0][0], max(0, bbox_pts[0][1] - 30)
                draw.text((x, y), text, font=font, fill=(0, 255, 0))
        
        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Sort by x-coordinate and combine
        plate_texts.sort(key=lambda x: x[0])
        final_plate = "".join([t[1] for t in plate_texts])
        predictions[img_name] = final_plate
        
        # Visualize: final result
        if visualize:
            # Display result on image
            result_text = f"Pred: {final_plate} | Conf: {avg_confidence:.3f}"            
            draw.text((10, 10), result_text, font=font, fill=(0, 0, 255))
            
            # Save visualization
            save_path = os.path.join(vis_dir, img_name)
            img_pil.save(save_path)
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(image_paths)} images")
    
    print("OCR completed")
    
    # Save predictions to text file
    if save_txt:
        txt_path = os.path.join(txt_dir, "predictions.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            for img_name in sorted(predictions.keys()):
                pred_text = predictions[img_name]
                f.write(f"{img_name}\t{pred_text}\n")
        
        print(f"\nPredictions saved to: {txt_path}")

    # Print summary
    print("\n" + "="*70)
    print("INFERENCE SUMMARY")
    print("="*70)
    print(f"Total images: {len(image_paths)}")
    print(f"Successful predictions: {sum(1 for v in predictions.values() if v != '')}")
    print(f"No detection: {sum(1 for v in predictions.values() if v == '')}")
    print(f"Results saved to: {output_dir}")
    
    if visualize:
        print(f"Visualizations: {vis_dir}")
    if save_txt:
        print(f"Predictions:    {txt_path}")
    print("="*70)
    
    # Print sample predictions
    print("\nSample predictions (first 10):")
    print("-" * 70)
    for idx, (img_name, pred_text) in enumerate(sorted(predictions.items())[:10]):
        print(f"{img_name}: {pred_text if pred_text else '(No detection)'}")
    print()
    
    return predictions


def main():
    # Configuration
    DATA_ROOT = "./datasets/plate_data"
    IMAGE_DIR = os.path.join(DATA_ROOT, "test_sr")
    OUTPUT_DIR = "./final_results"
    
    # Validate path
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Image directory not found: {IMAGE_DIR}")
        return
    
    # Run OCR inference
    predictions = run_ocr_inference(
        IMAGE_DIR,
        OUTPUT_DIR,
        visualize=True,
        save_txt=True
    )


if __name__ == "__main__":
    main()

