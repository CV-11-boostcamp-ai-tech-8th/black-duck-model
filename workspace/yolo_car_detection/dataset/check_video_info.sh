#!/bin/bash
TARGET_DIR="/data/ephemeral/home/dataset"

EXTS=("mp4" "avi")

for ext in "${EXTS[@]}"; do
  for f in "$TARGET_DIR"/*."$ext"; do
    [ -f "$f" ] || continue

    # ffprobe로 width, height, r_frame_rate, avg_frame_rate 추출 (각 줄로)
    readarray -t info <<< $(ffprobe -v error -select_streams v:0 \
      -show_entries stream=width,height,r_frame_rate,avg_frame_rate \
      -of default=nokey=1:noprint_wrappers=1 "$f")

    width="${info[0]}"
    height="${info[1]}"
    r_fps="${info[2]}"
    avg_fps="${info[3]}"

    # fps 분수 -> 소수
    r_fps_val=$(awk -F/ '{printf "%.3f", $1/$2}' <<< "$r_fps")
    avg_fps_val=$(awk -F/ '{printf "%.3f", $1/$2}' <<< "$avg_fps")

    # 총 프레임 수 계산 (VFR도 안전)
    nb_frames=$(ffprobe -v error -count_frames -select_streams v:0 \
      -show_entries stream=nb_read_frames \
      -of default=nokey=1:noprint_wrappers=1 "$f")

    # 출력
    echo "FILE: $f"
    echo "imgsz: (${width}, ${height})"
    printf "r_frame_rate: %s fps\n" "$r_fps_val"
    printf "avg_frame_rate: %s fps\n" "$avg_fps_val"
    echo "nb_frames: $nb_frames"
    echo "----"
  done
done
