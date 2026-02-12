# main.py

import os
import cv2
import csv
from ultralytics import YOLO
import config as cfg
from brake_detector import BrakeDetector
from utils import convert_avi_to_mp4


def get_warning_text(w_type):
    """경고 타입을 텍스트로 변환"""
    if w_type == cfg.WARN_HARD_BRAKE:
        return "HARD BRAKE!"
    elif w_type == cfg.WARN_TTC_CRITICAL:
        return "CRITICAL!"
    elif w_type == cfg.WARN_CUT_IN:
        return "CUT-IN BRAKE!"
    else:
        return "NORMAL"

def draw_brake_info(frame, boxes, detector, braking_events, img_w, img_h):
    """
    프레임에 차량 정보 및 경고 시각화
    
    Args:
        frame: 현재 프레임
        boxes: YOLO detection boxes
        detector: BrakeDetector 인스턴스
        braking_events: detector.update()가 반환한 이벤트 리스트
        img_w, img_h: 이미지 크기
    """
    cv2.line(frame, (img_w//2, 0), (img_w//2, img_h), (0, 255, 0), 1)
    
    braking_tids = set()
    braking_info_map = {}
    
    for event in braking_events:
        tid = event['track_id']
        braking_tids.add(tid)
        if tid in detector.tracks:
            track = detector.tracks[tid]
            braking_info_map[tid] = {
                'type': track['warning_type'],
                'acc': track['acc'],
                'ttc': track['ttc'],
                'dist': track['dist'],
                'speed': track['speed']
            }
    
    if boxes is not None and boxes.id is not None:
        for box in boxes:
            if box.id is None or int(box.cls.item()) != 0:
                continue
                
            tid = int(box.id.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            color = (0, 255, 0)
            thick = 2
            
            if tid in braking_tids and tid in braking_info_map:
                info = braking_info_map[tid]
                w_type = info['type']
                
                if w_type == cfg.WARN_TTC_CRITICAL:
                    color = (0, 0, 255)
                elif w_type == cfg.WARN_CUT_IN:
                    color = (0, 165, 255)
                else:  # HARD_BRAKE
                    color = (0, 0, 255)
                    
                thick = 4
                
                warn_text = get_warning_text(w_type)
                cv2.putText(frame, warn_text, (x1, y1 - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                stats = f"Acc:{info['acc']:.1f} TTC:{info['ttc']:.1f}s"
                cv2.putText(frame, stats, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
            
            if tid in detector.tracks:
                track = detector.tracks[tid]
                label = f"ID:{tid} {track['dist']:.1f}m"
                speed_kmh = track['speed'] * 3.6
                label2 = f"{speed_kmh:.1f}km/h"
                
                cv2.putText(frame, label, (x1, y2 + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, label2, (x1, y2 + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    return frame


def main():
    print("=" * 70)
    print("🚗 3D 급정거 감지 System v10 [Full Video Visualization]")
    print("=" * 70)

    if not os.path.exists(cfg.VIDEO_PATH):
        print(f"[Error] 비디오 없음: {cfg.VIDEO_PATH}")
        return

    print(f"▶ 모델 로드 중: {cfg.MODEL_WEIGHT}...")
    model = YOLO(cfg.MODEL_WEIGHT)

    cap = cv2.VideoCapture(cfg.VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"▶ 비디오 정보: {width}x{height} @ {fps:.2f}fps, {total_frames} frames")
    
    detector = BrakeDetector(fps=fps, img_w=width, img_h=height)

    video_name = os.path.splitext(os.path.basename(cfg.VIDEO_PATH))[0]
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    
    temp_avi_path = os.path.join(cfg.SAVE_DIR, f"{video_name}_annotated.avi")
    out = cv2.VideoWriter(temp_avi_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    
    csv_path = os.path.join(cfg.SAVE_DIR, f"{video_name}_frame_log.csv")
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    writer.writerow(["Frame", "ID", "Dist(m)", "PosX(m)", "Speed(km/h)", 
                     "Acc(m/s2)", "RawAcc", "TTC(s)", "WarningType"])

    active_events = {}
    finished_events = []

    frame_idx = 0
    print(f"▶ 처리 시작... (저장: {temp_avi_path})")
    print(f"▶ CSV 로그: {csv_path}")
    print("-" * 70)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, verbose=False, tracker=cfg.TRACKER_YAML)
        
        braking_events = detector.update(results[0].boxes, width, height)
        
        current_braking_tids = set()
        
        for event in braking_events:
            tid = event['track_id']
            current_braking_tids.add(tid)
            
            if tid not in detector.tracks:
                continue
                
            track = detector.tracks[tid]
            acc = track['acc']
            ttc = track['ttc']
            w_type = track['warning_type']
            
            if tid not in active_events:
                active_events[tid] = {
                    "id": tid,
                    "start_frame": frame_idx,
                    "end_frame": frame_idx,
                    "min_acc": acc,
                    "min_ttc": ttc,
                    "warning_types": {w_type},
                    "frames": 1
                }
                print(f"[🚨 NEW EVENT] ID {tid} started at frame {frame_idx}")
            else:
                evt = active_events[tid]
                evt["end_frame"] = frame_idx
                evt["min_acc"] = min(evt["min_acc"], acc)
                evt["min_ttc"] = min(evt["min_ttc"], ttc)
                evt["warning_types"].add(w_type)
                evt["frames"] += 1
            
            type_labels = ["NORMAL", "HARD_BRAKE", "CRITICAL", "CUT_IN"]
            type_str = type_labels[w_type] if 0 <= w_type < len(type_labels) else "UNKNOWN"
            
            if frame_idx % 5 == 0:
                print(f"[{type_str:12s}] Frame {frame_idx:5d} | ID: {tid:3d} | "
                      f"Acc: {acc:6.2f} | TTC: {ttc:5.2f}s | Dist: {track['dist']:5.1f}m")
        
        ended_ids = [tid for tid in active_events if tid not in current_braking_tids]
        for tid in ended_ids:
            evt = active_events.pop(tid)
            finished_events.append(evt)
            print(f"[✓ EVENT END] ID {tid} finished at frame {frame_idx} "
                  f"(duration: {evt['frames']} frames)")
        
        annotated_frame = draw_brake_info(frame, results[0].boxes, detector, 
                                          braking_events, width, height)
        
        info_text = f"Frame: {frame_idx}/{total_frames}"
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if len(braking_events) > 0:
            warning_text = f"WARNINGS: {len(braking_events)}"
            cv2.putText(annotated_frame, warning_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        for tid, track in detector.tracks.items():
            w_type = track['warning_type']
            raw_acc = track.get('raw_acc', 0.0)
            
            row = [
                frame_idx, tid,
                f"{track['dist']:.2f}", 
                f"{track['pos_x']:.2f}",
                f"{track['speed']*3.6:.1f}", 
                f"{track['acc']:.2f}",
                f"{raw_acc:.2f}", 
                f"{track['ttc']:.2f}",
                w_type
            ]
            writer.writerow(row)
        
        out.write(annotated_frame)
        
        if frame_idx % 100 == 0:
            percent = (frame_idx / total_frames * 100) if total_frames > 0 else 0
            print(f"[Progress] {frame_idx}/{total_frames} frames ({percent:.1f}%)")
        
        frame_idx += 1

    for tid, evt in active_events.items():
        finished_events.append(evt)
        print(f"[✓ EVENT END] ID {tid} finished at frame {frame_idx} "
              f"(duration: {evt['frames']} frames)")

    cap.release()
    out.release()
    csv_file.close()
    
    print("\n" + "=" * 70)
    print("📊 Event Summary")
    print("=" * 70)
    
    event_csv_path = os.path.join(cfg.SAVE_DIR, f"{video_name}_event_summary.csv")
    with open(event_csv_path, 'w', newline='', encoding='utf-8') as f:
        ev_writer = csv.writer(f)
        ev_writer.writerow(["ID", "StartFrame", "EndFrame", "Duration(F)", 
                           "MaxDecel(m/s2)", "MinTTC(s)", "Types"])
        
        if len(finished_events) == 0:
            print("  No brake events detected.")
        else:
            for i, evt in enumerate(finished_events, 1):
                types_str = "+".join([str(t) for t in sorted(evt["warning_types"])])
                ev_writer.writerow([
                    evt["id"], 
                    evt["start_frame"], 
                    evt["end_frame"], 
                    evt["frames"],
                    f"{evt['min_acc']:.2f}", 
                    f"{evt['min_ttc']:.2f}", 
                    types_str
                ])
                
                type_names = []
                for t in sorted(evt["warning_types"]):
                    if t == cfg.WARN_HARD_BRAKE:
                        type_names.append("HARD_BRAKE")
                    elif t == cfg.WARN_TTC_CRITICAL:
                        type_names.append("CRITICAL")
                    elif t == cfg.WARN_CUT_IN:
                        type_names.append("CUT_IN")
                
                print(f"  [{i:2d}] ID {evt['id']:3d}: "
                      f"Frame {evt['start_frame']:5d}~{evt['end_frame']:5d} "
                      f"({evt['frames']:3d}F) | "
                      f"MaxDecel: {evt['min_acc']:6.2f} m/s² | "
                      f"MinTTC: {evt['min_ttc']:5.2f}s | "
                      f"Types: {', '.join(type_names)}")
    
    print("=" * 70)
    print(f"\n✅ Total Events Detected: {len(finished_events)}")
    print(f"✅ Frame Log saved: {csv_path}")
    print(f"✅ Event Summary saved: {event_csv_path}")
    print(f"✅ Video saved (AVI): {temp_avi_path}")
    
    # MP4 변환
    print("\n▶ Converting to MP4...")
    mp4_path = convert_avi_to_mp4(temp_avi_path)
    if mp4_path:
        print(f"✅ MP4 conversion completed: {mp4_path}")
    else:
        print("⚠️  MP4 conversion failed (keeping AVI)")
    
    print("\n" + "=" * 70)
    print("🎉 Processing completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()