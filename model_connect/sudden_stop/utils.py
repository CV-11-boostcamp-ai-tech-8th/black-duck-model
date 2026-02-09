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
        # [Change] 하드코딩된 경로 사용 (model_connect와 동일)
        home_ffmpeg = Path("/data/ephemeral/home/bin")
        if home_ffmpeg.exists():
            ffmpeg_dirs = list(home_ffmpeg.glob("ffmpeg-*"))
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

def extract_event_package_memory(event, video_path, fps, clip_idx, save_to_disk=False, output_dir=None):
    """
    이벤트 영상과 crop 이미지를 메모리에서 추출
    
    Args:
        event: 이벤트 정보
        video_path: 원본 비디오 경로
        fps: 프레임레이트
        clip_idx: 클립 순번
        save_to_disk: 디스크에 저장할지 여부
        output_dir: 저장할 디렉토리 (save_to_disk=True일 때만 사용)
    
    Returns:
        dict: {
            'video_frames': list[np.ndarray],  # 메모리에 저장된 프레임 리스트
            'thumbnail': np.ndarray,  # 썸네일 이미지
            'vehicle_crops': list[np.ndarray],  # 차량 crop 이미지 리스트
            'clip_path': str | None  # 디스크에 저장된 경우 경로
        }
    """
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not event.get("frame_boxes"):
        cap.release()
        return None
        
    start_f = max(0, event["start_frame"] - int(fps * 5))
    end_f = event["end_frame"] + int(fps * 5.0)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    
    video_frames = []
    thumbnail = None
    temp_candidates = []
    fidx = start_f
    
    # 디스크 저장용
    writer = None
    clip_path = None
    
    if save_to_disk and output_dir:
        event_dir = os.path.join(output_dir, f"clip_{clip_idx}")
        os.makedirs(event_dir, exist_ok=True)
        temp_avi = os.path.join(event_dir, f"clip_{clip_idx}.avi")
        writer = cv2.VideoWriter(temp_avi, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

    while fidx <= end_f:
        ret, frame = cap.read()
        if not ret: 
            break
        
        # 메모리에 프레임 저장
        video_frames.append(frame.copy())
        
        # 디스크에 저장 (필요시)
        if writer:
            writer.write(frame)

        # 썸네일 추출 (start_frame에서)
        if fidx == event["start_frame"]:
            thumbnail = frame.copy()
            if save_to_disk and output_dir:
                event_dir = os.path.join(output_dir, f"clip_{clip_idx}")
                cv2.imwrite(os.path.join(event_dir, "thumbnail.jpg"), frame)

        # 차량 crop 추출
        if "frame_boxes" in event and fidx in event["frame_boxes"]:
            for info in event["frame_boxes"][fidx]:
                bbox = info.get("bbox")
                if not bbox: 
                    continue
                
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: 
                    continue
                
                score = score_crop(x1, y1, x2, y2, w, info.get("dist", 0))
                temp_candidates.append({"score": score, "img": crop})
        
        fidx += 1

    if writer:
        writer.release()
        # AVI를 MP4로 변환
        temp_avi = os.path.join(output_dir, f"clip_{clip_idx}", f"clip_{clip_idx}.avi")
        final_mp4 = convert_avi_to_mp4(temp_avi)
        clip_path = str(final_mp4) if final_mp4 else temp_avi

    cap.release()

    # 점수 기준으로 정렬하여 상위 60개 이미지 추출
    temp_candidates.sort(key=lambda x: x['score'], reverse=True)
    best_crops = [item['img'] for item in temp_candidates[:60]]
    
    # 디스크에 crop 저장 (필요시)
    if save_to_disk and output_dir and best_crops:
        event_dir = os.path.join(output_dir, f"clip_{clip_idx}")
        crop_dir = os.path.join(event_dir, "crops")
        os.makedirs(crop_dir, exist_ok=True)
        
        for i, crop_img in enumerate(best_crops):
            cv2.imwrite(os.path.join(crop_dir, f"crop_{i:02d}.jpg"), crop_img)

    return {
        'video_frames': video_frames,  # 메모리에 저장된 프레임 리스트
        'thumbnail': thumbnail,  # 썸네일 이미지 (numpy array)
        'vehicle_crops': best_crops,  # 차량 crop 이미지 리스트
        'clip_path': clip_path  # 디스크에 저장된 경우 경로
    }
