import cv2
import os
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from brake_detector import BrakeDetector
import config as cfg

def calculate_iou(box1, box2):
    """
    두 bbox의 IoU(Intersection over Union) 계산
    box: (x1, y1, x2, y2)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 교집합 영역
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    
    # 합집합 영역
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def visualize_brake_events(input_video_path, output_dir):
    """
    급정거 이벤트를 시각화하는 함수
    - 전방 차량: 초록색 bbox
    - 급정거 차량: 빨간색 bbox
    - 이벤트 발생 1초 전 ~ 종료 1초 후 클립 저장
    """
    video_path = str(input_video_path)
    model_path = cfg.MODEL_WEIGHT
    tracker_config = cfg.TRACKER_YAML 
    
    save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"Processing: {video_path}")

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
    
    frame_cnt = 0
    clip_count = 0

    print("Starting detection phase...")

    # ========== 1단계: 전체 영상 처리 & 이벤트 탐지 ==========
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
                    f"Detecting: {frame_cnt}/{total_frames} frames "
                    f"({percent:.1f}%), "
                    f"ETA ~ {remaining_sec:.1f}s",
                    flush=True,
                )
            else:
                print(f"Detecting: {frame_cnt} frames", flush=True)
        
        results = model.track(frame, persist=True, verbose=False, tracker=tracker_config)
        
        detector.update(results[0].boxes, width, height)

    cap.release()
    print("Detection phase completed. Now generating event clips...")

    # ========== 2단계: 이벤트 정보 정리 ==========
    for eid, ev in detector.events.items():
        if ev["status"] == "ACTIVE":
            ev["status"] = "FINISHED"
            if ev["end_frame"] is None:
                ev["end_frame"] = frame_cnt
    
    if len(detector.events) == 0:
        print("No brake events detected.")
        return 0
    
    # ========== 3단계: 클립 생성 ==========
    for eid, ev in detector.events.items():
        if ev["status"] != "FINISHED":
            continue
        
        clip_count += 1
        target_tid = ev["track_id"]
        
        start_frame = max(1, ev["start_frame"] - int(fps * 4.0))
        end_frame = min(frame_cnt, ev["end_frame"] + int(fps * 4.0))
        
        print(f"\n[EVENT {eid}] Original TID:{target_tid} | Frames: {start_frame} ~ {end_frame}")
        
        event_bboxes = {}
        
        for f_idx in range(ev["start_frame"], ev["end_frame"] + 1):
            if f_idx in detector.frame_history:
                matched = [box for box in detector.frame_history[f_idx] 
                          if box.get("tid") == target_tid]
                if matched:
                    bbox = matched[0]["bbox"]
                    event_bboxes[f_idx] = tuple(bbox)
        
        if not event_bboxes:
            print(f"  -> Warning: No bbox data found for TID {target_tid}")
            continue
        
        print(f"  -> Found {len(event_bboxes)} bbox records in frame_history")
        
        clip_filename = f"event_{clip_count}_tid_{target_tid}.avi"
        clip_path = os.path.join(save_dir, clip_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
        
        if not writer.isOpened():
            print(f"  -> Error: Failed to create video writer")
            continue
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
        
        for f_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                print(f"  -> Warning: Failed to read frame {f_idx}")
                break
            
            annotated_frame = frame.copy()
            
            results = model.track(frame, persist=True, verbose=False, tracker=tracker_config)
            
            current_detections = []
            if results[0].boxes is not None and results[0].boxes.id is not None:
                for box in results[0].boxes:
                    if box.id is None or int(box.cls.item()) != 0:
                        continue
                    
                    new_tid = int(box.id.item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    current_detections.append({
                        'tid': new_tid,
                        'bbox': (x1, y1, x2, y2)
                    })
            
            is_event_active = (f_idx >= ev["start_frame"] and f_idx <= ev["end_frame"])
            
            event_bbox_matched = None
            
            if is_event_active and f_idx in event_bboxes:
                original_bbox = event_bboxes[f_idx]
                best_iou = 0.0
                best_detection = None
                
                for det in current_detections:
                    iou = calculate_iou(original_bbox, det['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_detection = det
                
                if best_iou > 0.3:
                    event_bbox_matched = best_detection
                    print(f"  Frame {f_idx}: Matched TID {best_detection['tid']} (IoU: {best_iou:.2f})", end='\r')
            
            for det in current_detections:
                x1, y1, x2, y2 = det['bbox']
                
                if event_bbox_matched and det['tid'] == event_bbox_matched['tid']:
                    color = (0, 0, 255)
                    thickness = 3
                    label = f"BRAKE!"
                else:
                    color = (0, 255, 0)
                    thickness = 2
                    label = f"ID:{det['tid']}"
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_y = max(y1 - 10, label_size[1])
                cv2.rectangle(annotated_frame, 
                             (x1, label_y - label_size[1] - 5), 
                             (x1 + label_size[0], label_y + 5), 
                             color, -1)
                cv2.putText(annotated_frame, label, (x1, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            info_text = f"Frame: {f_idx} | Event {eid}"
            if is_event_active:
                info_text += " [BRAKING]"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            writer.write(annotated_frame)
        
        writer.release()
        cap.release()
        print(f"\n  -> Saved AVI: {clip_filename}")
        
        # MP4 변환
        mp4_filename = f"event_{clip_count}_tid_{target_tid}.mp4"
        mp4_path = os.path.join(save_dir, mp4_filename)
        
        try:
            import subprocess
            
            ffmpeg_cmd = "ffmpeg"
            
            cmd = [
                ffmpeg_cmd, '-y', '-i', clip_path, 
                '-c:v', 'libx264',
                '-preset', 'fast', 
                '-crf', '23', 
                '-an',
                mp4_path
            ]
            
            print(f"  -> Converting to MP4...")
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode == 0 and os.path.exists(mp4_path):
                os.remove(clip_path)
                print(f"  -> Converted to MP4: {mp4_filename}")
            else:
                print(f"  -> MP4 conversion failed (keeping AVI)")
                print(f"     Error: {result.stderr[:200]}")
                
        except FileNotFoundError:
            print(f"  -> FFmpeg not found (keeping AVI)")
        except Exception as e:
            print(f"  -> MP4 conversion error: {str(e)[:100]}")
    
    print(f"\n✅ Finished! Generated {clip_count} event clips in {save_dir}")
    return clip_count


if __name__ == "__main__":
    input_video = cfg.VIDEO_PATH
    output_path = cfg.SAVE_DIR 
    
    if os.path.exists(input_video):
        visualize_brake_events(input_video, output_path)
    else:
        print(f"Video not found: {input_video}")