import cv2
import os
import torch
from pathlib import Path
from ultralytics import YOLO
# from brake_detector import BrakeDetector
from .brake_detector import BrakeDetector
# import utils
from . import utils
# import config as cfg
from . import config as cfg

class ClipResult:
    def __init__(self, video_path, crops):
        self.clip_video_path = Path(video_path)
        self.vehicle_crops = crops

def detect_and_generate_clips(input_video_path, output_dir):
    video_path = str(input_video_path)
    model_path = cfg.MODEL_WEIGHT
    tracker_config = cfg.TRACKER_YAML 
    
    save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"Processing: {video_path}")

    # 이벤트 종료 후 영상을 얼마나 더 포함할지 (초)
    POST_EVENT_WAIT_SEC = 1.0 

    try:
        model = YOLO(model_path)
        print(f"Model loaded: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Detector 초기화
    detector = BrakeDetector(fps=fps, img_w=width, img_h=height)
    
    generated_clips = []
    frame_cnt = 0
    clip_count = 0
    wait_frames = int(fps * POST_EVENT_WAIT_SEC)

    print("Starting main loop...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_cnt += 1

        if frame_cnt % int(fps * 5) == 0:
            if total_frames > 0:
                percent = (frame_cnt / total_frames) * 100
                remaining_frames = total_frames - frame_cnt
                remaining_sec = remaining_frames / fps
                print(
                    f"processed {frame_cnt}/{total_frames} frames "
                    f"({percent:.1f}%), "
                    f"ETA ~ {remaining_sec:.1f}s",
                    flush=True,
                )
            else:
                print(f"processed {frame_cnt} frames", flush=True)
        
        results = model.track(frame, persist=True, verbose=False, tracker=tracker_config)
        
        detector.update(results[0].boxes, width, height)
        
        for eid, ev in list(detector.events.items()):
            
            if ev.get("saved", False) or ev["status"] == "ACTIVE":
                continue

            if ev["status"] == "FINISHED":
                if frame_cnt < ev["end_frame"] + wait_frames:
                    continue

                ev["saved"] = True
                clip_count += 1
                
                target_tid = ev["track_id"]
                print(f"  [DETECTED] Event {eid} (TID: {target_tid}) -> Saving clip_{clip_count}...")

                ev["frame_boxes"] = {}
                start_search = ev["start_frame"]
                end_search = ev["end_frame"] + wait_frames
                
                for f in range(start_search, end_search + 1):
                    if f in detector.frame_history:
                        matched_boxes = [
                            box for box in detector.frame_history[f] 
                            if box.get("tid") == target_tid
                        ]
                        if matched_boxes:
                            ev["frame_boxes"][f] = matched_boxes

                pkg = utils.save_event_package(
                    ev, video_path, fps, save_dir, clip_count, save_crops=True
                )
                
                if pkg:
                    count = len(pkg['vehicle_crops'])
                    print(f"    -> Saved: {os.path.basename(pkg['clip_path'])} (Crops: {count})")
                    clip_res = ClipResult(pkg['clip_path'], pkg['vehicle_crops'])
                    generated_clips.append(clip_res)

    for eid, ev in detector.events.items():
        if not ev.get("saved", False):
            ev["status"] = "FINISHED"
            if ev["end_frame"] is None:
                ev["end_frame"] = frame_cnt
            
            clip_count += 1
            print(f"  [FINAL FLUSH] Event {eid} -> Saving clip_{clip_count}...")
            
            ev["frame_boxes"] = {}
            start_search = ev["start_frame"]
            end_search = min(frame_cnt, ev["end_frame"] + wait_frames)

            for f in range(start_search, end_search + 1):
                if f in detector.frame_history:
                    matched_boxes = [
                        box for box in detector.frame_history[f] 
                        if box.get("tid") == ev["track_id"]
                    ]
                    if matched_boxes:
                        ev["frame_boxes"][f] = matched_boxes

            pkg = utils.save_event_package(
                ev, video_path, fps, save_dir, clip_count, save_crops=True
            )
            
            if pkg:
                generated_clips.append(ClipResult(pkg['clip_path'], pkg['vehicle_crops']))

    cap.release()
    print("Finished processing.")
    return generated_clips

"""if __name__ == "__main__":
    input_video = cfg.VIDEO_PATH
    output_path = cfg.SAVE_DIR 
    
    if os.path.exists(input_video):
        detect_and_generate_clips(input_video, output_path)
    else:
        print(f"Video not found: {input_video}")"""