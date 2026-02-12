import cv2
import os
import torch
from pathlib import Path
from ultralytics import YOLO
from .brake_detector import BrakeDetector
from . import utils
from . import config as cfg

class ClipResult:
    """
    메모리 기반 Clip 결과 클래스
    """
    def __init__(self, clip_id, video_frames=None, thumbnail=None, vehicle_crops=None, clip_path=None, fps=None):
        self.clip_id = clip_id
        self.video_frames = video_frames  # list[np.ndarray] - 메모리에 저장
        self.thumbnail = thumbnail  # np.ndarray - 메모리에 저장
        self.vehicle_crops = vehicle_crops  # list[np.ndarray] - 메모리에 저장
        self.clip_video_path = Path(clip_path) if clip_path else None  # 디스크에 저장된 경우만
        self.fps = fps  # 프레임레이트

def detect_and_generate_clips(input_video_path, output_dir, save_to_disk=False):
    """
    급정거 감지 및 clip 생성 (메모리 기반)
    
    Args:
        input_video_path: 입력 비디오 경로
        output_dir: 출력 디렉토리 (save_to_disk=True일 때만 사용)
        save_to_disk: 디스크에 저장할지 여부 (기본값: False)
    
    Returns:
        list[ClipResult]: clip 결과 리스트
    """
    video_path = str(input_video_path)
    model_path = cfg.MODEL_WEIGHT
    tracker_config = cfg.TRACKER_YAML 
    
    save_dir = output_dir if save_to_disk else None
    if save_to_disk:
        os.makedirs(save_dir, exist_ok=True)
    
    print(f"Processing: {video_path}")

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

                pkg = utils.extract_event_package_memory(
                    ev, video_path, fps, clip_count, 
                    save_to_disk=save_to_disk, 
                    output_dir=save_dir
                )
                
                if pkg:
                    count = len(pkg['vehicle_crops'])
                    print(f"    -> Saved: clip_{clip_count} (Crops: {count})")
                    clip_res = ClipResult(
                        clip_id=clip_count,
                        video_frames=pkg['video_frames'],
                        thumbnail=pkg['thumbnail'],
                        vehicle_crops=pkg['vehicle_crops'],
                        clip_path=pkg['clip_path'],
                        fps=fps
                    )
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

            pkg = utils.extract_event_package_memory(
                ev, video_path, fps, clip_count,
                save_to_disk=save_to_disk,
                output_dir=save_dir
            )
            
            if pkg:
                count = len(pkg['vehicle_crops'])
                print(f"    -> Saved: clip_{clip_count} (Crops: {count})")
                clip_res = ClipResult(
                    clip_id=clip_count,
                    video_frames=pkg['video_frames'],
                    thumbnail=pkg['thumbnail'],
                    vehicle_crops=pkg['vehicle_crops'],
                    clip_path=pkg['clip_path'],
                    fps=fps
                )
                generated_clips.append(clip_res)

    cap.release()
    print("Finished processing.")
    return generated_clips
