#utils.py
import json
import os
import cv2
import numpy as np
import subprocess
from pathlib import Path

def convert_avi_to_mp4(avi_path):
    avi_path = Path(avi_path)
    if not avi_path.exists(): return None
    
    mp4_path = avi_path.with_suffix('.mp4')

    try:
        ffmpeg_cmd = "ffmpeg"
        #home_ffmpeg = Path.home() / "bin"
        home_ffmpeg = Path("/data/ephemeral/home/bin")
        if home_ffmpeg.exists():
            #ffmpeg_dirs = list(home_ffmpeg.glob("ffmpeg-*"))
            ffmpeg_dirs = list[Path](home_ffmpeg.glob("ffmpeg-*"))
            if ffmpeg_dirs: ffmpeg_cmd = str(ffmpeg_dirs[0] / "ffmpeg")
            
        cmd = [
            ffmpeg_cmd, "-i", str(avi_path), "-c:v", "libx264",
            "-preset", "fast", "-crf", "23", "-an", "-y", str(mp4_path)
        ]

        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if result.returncode == 0 and mp4_path.exists():
            avi_path.unlink()
            return mp4_path
    except:
        return None
    
    return None

def score_crop(x1, y1, x2, y2, img_w, dist):
    width, height = x2 - x1, y2 - y1
    area = width * height
    cx = (x1 + x2) / 2
    center_bias = 1.0 - (abs(cx - img_w/2) / (img_w/2)) 
    dist_factor = 1.0 / (dist + 1.0) * 10.0
    return area * center_bias * dist_factor

def save_event_package(event, video_path, fps, save_dir, clip_idx, save_crops=True):
    """
    이벤트 영상을 저장하고, 중요 Crop 이미지를 추출함.
    clip_idx: 클립 순번 (예: 1 -> clip_1.avi)
    """
    event_dir = os.path.join(save_dir, f"clip_{clip_idx}")
    os.makedirs(event_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not event.get("frame_boxes"):
        cap.release()
        return None
        
    all_indices = sorted(event["frame_boxes"].keys())
    start_f = max(0, event["start_frame"] - int(fps * 5))
    end_f = event["end_frame"] + int(fps * 5.0)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    
    temp_avi = os.path.join(event_dir, f"clip_{clip_idx}.avi")
    writer = cv2.VideoWriter(temp_avi, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

    temp_candidates = []
    fidx = start_f

    while fidx <= end_f:
        ret, frame = cap.read()
        if not ret: break
        writer.write(frame)

        if fidx == event["start_frame"]:
            cv2.imwrite(os.path.join(event_dir, "thumbnail.jpg"), frame)

        if "frame_boxes" in event and fidx in event["frame_boxes"]:
            for info in event["frame_boxes"][fidx]:
                bbox = info.get("bbox")
                if not bbox: continue
                
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue
                
                score = score_crop(x1, y1, x2, y2, w, info.get("dist", 0))
                temp_candidates.append({"score": score, "img": crop})
        
        fidx += 1

    writer.release()
    cap.release()

    final_mp4 = convert_avi_to_mp4(temp_avi)
    final_path = str(final_mp4) if final_mp4 else temp_avi
    
    temp_candidates.sort(key=lambda x: x['score'], reverse=True)
    best_crops = [item['img'] for item in temp_candidates[:60]]
    
    if save_crops and best_crops:
        crop_dir = os.path.join(event_dir, "crops")
        os.makedirs(crop_dir, exist_ok=True)
        
        for i, crop_img in enumerate(best_crops):
            cv2.imwrite(os.path.join(crop_dir, f"crop_{i:02d}.jpg"), crop_img)

    return {
        'clip_path': final_path,
        'vehicle_crops': best_crops
    }