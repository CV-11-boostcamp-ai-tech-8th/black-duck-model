# main.py

import os
import cv2
import csv
from ultralytics import YOLO
import config as cfg
from brake_detector import BrakeDetector
from utils import draw_brake_info, convert_avi_to_mp4

def main():
    print("=" * 70)
    print("🚗 3D 급정거 감지 System v9 [Type Analysis + Event Summary]")
    print("=" * 70)

    if not os.path.exists(cfg.VIDEO_PATH):
        print(f"[Error] 비디오 없음: {cfg.VIDEO_PATH}")
        return

    # 모델 로드
    print(f"▶ 모델 로드 중: {cfg.MODEL_WEIGHT}...")
    model = YOLO(cfg.MODEL_WEIGHT)

    cap = cv2.VideoCapture(cfg.VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    detector = BrakeDetector(fps=fps, img_w=width, img_h=height)

    # 저장 설정
    video_name = os.path.splitext(os.path.basename(cfg.VIDEO_PATH))[0]
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    
    temp_avi_path = os.path.join(cfg.SAVE_DIR, f"{video_name}_temp.avi")
    out = cv2.VideoWriter(temp_avi_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    
    # CSV Writer (Frame 단위)
    csv_path = os.path.join(cfg.SAVE_DIR, f"{video_name}_frame_log.csv")
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["Frame", "ID", "Dist(m)", "PosX(m)", "Speed(km/h)", "Acc(m/s2)", "RawAcc", "TTC(s)", "WarningType"])

    # --- [Event Merging Variables] ---
    active_events = {} 
    finished_events = []

    frame_idx = 0
    print(f"▶ 처리 시작... (저장: {temp_avi_path})")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # YOLO Inference
        results = model.track(frame, persist=True, verbose=False, tracker=cfg.TRACKER_YAML)
        
        # Detector Update (여기서 반환되는 리스트의 요소는 'event' 딕셔너리입니다)
        braking_cars_events = detector.update(results[0].boxes, width, height)
        
        # 현재 프레임에서 감지된 이벤트의 Track ID 목록 추출 (종료 처리용)
        current_frame_event_tids = set()

        # --- [Event Logic] ---
        # 1. 현재 경고 중인 차량 처리
        for event_info in braking_cars_events:
            # [FIX 1] Key 변경: 'id' -> 'track_id'
            tid = event_info['track_id']
            current_frame_event_tids.add(tid)
            
            # [FIX 2] 필요한 변수(acc, ttc, w_type) 가져오기
            # detector.tracks에서 실시간 수치를 가져와야 합니다.
            if tid in detector.tracks:
                track = detector.tracks[tid]
                acc = track['acc']
                ttc = track['ttc']
                w_type = track['warning_type']
            else:
                continue

            if tid not in active_events:
                # 이벤트 시작
                active_events[tid] = {
                    "id": tid,
                    "start_frame": frame_idx,
                    "end_frame": frame_idx,
                    "min_acc": acc, 
                    "min_ttc": ttc,
                    "warning_types": {w_type}, 
                    "frames": 1
                }
            else:
                # 이벤트 업데이트
                evt = active_events[tid]
                evt["end_frame"] = frame_idx
                evt["min_acc"] = min(evt["min_acc"], acc)
                evt["min_ttc"] = min(evt["min_ttc"], ttc)
                evt["warning_types"].add(w_type)
                evt["frames"] += 1
            
            # 콘솔 출력 (Type 표시)
            # w_type이 유효한 범위인지 확인
            type_labels = ["NORMAL", "HARD_BRAKE", "CRITICAL", "CUT_IN"]
            type_str = type_labels[w_type] if 0 <= w_type < len(type_labels) else "UNKNOWN"
            
            print(f"[🚨 {type_str}] Frame {frame_idx} | ID: {tid} | "
                  f"Acc: {acc:.2f} | TTC: {ttc:.2f}")

        # 2. 경고가 끝난 차량 처리 (이벤트 종료)
        # active_events에는 있는데, 이번 프레임의 braking_cars_events에는 ID가 없는 경우
        ended_ids = []
        for tid in active_events:
            # [FIX 3] 리스트 직접 비교가 아니라 ID 집합(set)과 비교해야 함
            if tid not in current_frame_event_tids:
                ended_ids.append(tid)
        
        for tid in ended_ids:
            evt = active_events.pop(tid)
            finished_events.append(evt)

        # 시각화 & 저장
        # draw_brake_info에는 detector.tracks 정보를 넘기는 것이 시각화에 더 유리할 수 있으나,
        # 기존 코드 호환성을 위해 braking_cars_events(이벤트 리스트)를 넘깁니다.
        frame = draw_brake_info(frame, results[0].boxes, detector, braking_cars_events, width, height)
        
        # CSV 기록
        for tid, track in detector.tracks.items():
            w_type = track['warning_type']
            raw_acc = track.get('raw_acc', 0.0)
            
            row = [
                frame_idx, tid,
                f"{track['dist']:.2f}", f"{track['pos_x']:.2f}",
                f"{track['speed']*3.6:.1f}", f"{track['acc']:.2f}",
                f"{raw_acc:.2f}", f"{track['ttc']:.2f}",
                w_type 
            ]
            writer.writerow(row)

        out.write(frame)
        if frame_idx % 50 == 0: print(f"[Frame {frame_idx}] Processing...")
        frame_idx += 1

    # 루프 종료 후 남은 이벤트 처리
    for tid, evt in active_events.items():
        finished_events.append(evt)

    cap.release()
    out.release()
    csv_file.close()
    
    # --- [Event Summary CSV 저장] ---
    event_csv_path = os.path.join(cfg.SAVE_DIR, f"{video_name}_event_summary.csv")
    with open(event_csv_path, 'w', newline='') as f:
        ev_writer = csv.writer(f)
        ev_writer.writerow(["ID", "StartFrame", "EndFrame", "Duration(F)", "MaxDecel(m/s2)", "MinTTC(s)", "Types"])
        
        print("\n" + "="*50)
        print("📊 Event Summary")
        for evt in finished_events:
            types_str = "+".join([str(t) for t in evt["warning_types"]])
            ev_writer.writerow([
                evt["id"], evt["start_frame"], evt["end_frame"], evt["frames"],
                f"{evt['min_acc']:.2f}", f"{evt['min_ttc']:.2f}", types_str
            ])
            print(f"ID {evt['id']}: Frame {evt['start_frame']}~{evt['end_frame']} | "
                  f"MaxDecel: {evt['min_acc']:.2f} | Types: {types_str}")
    print("="*50 + "\n")

    # MP4 변환
    convert_avi_to_mp4(temp_avi_path)

if __name__ == "__main__":
    main()
