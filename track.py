#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv26s Vehicle Tracking (Video)
- Fine-tuned vehicle detection 모델로 비디오 파일에서 tracking 수행
- 연속된 프레임으로 더 정확한 tracking 가능
"""

import os
import subprocess
from pathlib import Path
from ultralytics import YOLO
# from huggingface_hub import hf_hub_download

# ========== 설정 변수 ==========
# 모델 설정
MODEL_WEIGHT = "./runs/detect/cv-11-final/yolo26l_v2-car-road-version_e40_b40/weights/best.pt"
# MODEL_WEIGHT = hf_hub_download(
#     # repo_id="rujutashashikanjoshi/yolo12-vehicles-detection-3941-100m",
#     repo_id="wuhp/yolocar",
#     filename="car-75e-11n.pt"
# )

# 비디오 파일 경로
# VIDEO_PATH = "/data/ephemeral/home/dataset/20260115-11h37m24s_N.avi"
# VIDEO_PATH = "/data/ephemeral/home/dataset/20260115-11h38m24s_N.avi"
# VIDEO_PATH = "/data/ephemeral/home/dataset/급정거1_cropped.mp4"
# VIDEO_PATH = "/data/ephemeral/home/dataset/daytime.mp4"
# VIDEO_PATH = "/data/ephemeral/home/dataset/nighttime.mp4"
VIDEO_PATH = "/data/ephemeral/home/dataset/급정거_2.mp4"

# Tracking 설정
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
TRACKER_TYPE = "configs/yolo/bytetrack.yaml"  # "botsort.yaml" or "bytetrack.yaml"

# 저장 경로
VERSION = "v2"
TRACK_PROJECT = f"track"
TRACK_NAME_PREFIX = f"yolo26l_{VERSION}"
# ================================


def convert_avi_to_mp4(avi_path):
    """
    AVI 파일을 MP4 (H264)로 변환하고 원본 AVI 삭제
    
    Args:
        avi_path: AVI 파일 경로 (str or Path)
    
    Returns:
        mp4_path: 변환된 MP4 파일 경로 (성공시), None (실패시)
    """
    avi_path = Path(avi_path)
    
    if not avi_path.exists():
        print(f"[Warning] AVI 파일이 존재하지 않습니다: {avi_path}")
        return None
    
    # MP4 출력 경로
    mp4_path = avi_path.with_suffix('.mp4')
    
    print()
    print("=" * 70)
    print("AVI → MP4 변환 시작")
    print("=" * 70)
    print(f"입력: {avi_path}")
    print(f"출력: {mp4_path}")
    print()
    
    try:
        # ffmpeg 명령 구성
        # -i: 입력 파일
        # -c:v libx264: H.264 코덱 사용
        # -preset slow: 압축 품질 (slow = 더 좋은 압축률)
        # -crf 23: 품질 설정 (0=무손실, 23=기본값, 51=최악)
        # -an: 오디오 스트림 제거 (tracking 영상에는 오디오 불필요)
        # -y: 기존 파일 덮어쓰기
        
        # ffmpeg 경로 찾기 (사용자 스크립트 기준)
        ffmpeg_cmd = None
        
        # 1. ~/bin/ffmpeg-* 경로에서 찾기
        home_ffmpeg = Path.home() / "bin"
        if home_ffmpeg.exists():
            ffmpeg_dirs = list(home_ffmpeg.glob("ffmpeg-*"))
            if ffmpeg_dirs:
                ffmpeg_cmd = str(ffmpeg_dirs[0] / "ffmpeg")
        
        # 2. 시스템 ffmpeg 사용
        if ffmpeg_cmd is None or not Path(ffmpeg_cmd).exists():
            ffmpeg_cmd = "ffmpeg"
        
        cmd = [
            ffmpeg_cmd,
            "-i", str(avi_path),
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "23",
            "-an",  # 오디오 제거
            "-y",  # 덮어쓰기
            str(mp4_path)
        ]
        
        print("ffmpeg 실행 중...")
        print(f"명령: {' '.join(cmd)}")
        print()
        
        # ffmpeg 실행
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            print(f"[Error] ffmpeg 변환 실패!")
            print(f"에러 메시지:\n{result.stderr}")
            return None
        
        # 변환 성공 확인
        if not mp4_path.exists():
            print(f"[Error] MP4 파일이 생성되지 않았습니다.")
            return None
        
        # 파일 크기 비교
        avi_size = avi_path.stat().st_size / (1024 * 1024)  # MB
        mp4_size = mp4_path.stat().st_size / (1024 * 1024)  # MB
        
        print("✓ 변환 완료!")
        print(f"   AVI 크기: {avi_size:.2f} MB")
        print(f"   MP4 크기: {mp4_size:.2f} MB")
        print(f"   절약된 용량: {avi_size - mp4_size:.2f} MB ({(1 - mp4_size/avi_size)*100:.1f}%)")
        print()
        
        # 원본 AVI 파일 삭제
        print("원본 AVI 파일 삭제 중...")
        avi_path.unlink()
        print("✓ AVI 파일 삭제 완료")
        print("=" * 70)
        
        return mp4_path
        
    except FileNotFoundError:
        print(f"[Error] ffmpeg를 찾을 수 없습니다.")
        print(f"ffmpeg를 설치하거나 경로를 확인해주세요.")
        return None
    except Exception as e:
        print(f"[Error] 변환 중 오류 발생: {e}")
        return None


def run_tracking_on_video(model, video_path):
    """
    비디오 파일로 tracking 수행
    
    Args:
        model: YOLO 모델
        video_path: 비디오 파일 경로
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        print(f"[Error] 비디오 파일이 존재하지 않습니다: {video_path}")
        return
    
    # 비디오 파일명 (확장자 제외)
    video_name = video_path.stem
    
    print("\n" + "=" * 70)
    print(f"Tracking 시작: {video_name}")
    print("=" * 70)
    print(f"비디오 파일: {video_path}")
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    print(f"IoU threshold: {IOU_THRESHOLD}")
    print(f"Tracker: {TRACKER_TYPE}")
    print()
    
    # 저장 경로 구성: track/yolo26s_{VERSION}/{video_name}
    project_path = f"{TRACK_PROJECT}/{TRACK_NAME_PREFIX}"
    
    # Tracking 수행
    results = model.track(
        source=str(video_path),
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        persist=True,  # 프레임 간 track 유지
        tracker=TRACKER_TYPE,
        save=True,  # 결과 이미지 저장
        save_txt=True,  # 결과 텍스트 저장

            # 시각화 커스터마이징 파라미터
        show_labels=True,    # 클래스 이름 표시 (기본: True)
        show_conf=True,      # Confidence 값 표시 (기본: True)
        show_boxes=True,     # BBox 표시 (기본: True)
        line_width=1,        # BBox 선 두께 (기본: None=자동)

        project=project_path,
        name=video_name,
        exist_ok=True,
        stream=True,  # 메모리 효율적 처리
    )
    
    # 결과 처리 (stream=True이므로 iteration 필요)
    track_ids_seen = set()
    frame_count = 0
    
    print("처리 중...")
    for result in results:
        frame_count += 1
        
        # Track ID 수집
        if result.boxes and hasattr(result.boxes, 'id') and result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().tolist()
            track_ids_seen.update(ids)
        
        # 진행 상황 출력 (100 프레임마다)
        if frame_count % 100 == 0:
            print(f"  {frame_count} 프레임 처리 완료...")
    
    print()
    print("=" * 70)
    print("Tracking 완료!")
    print("=" * 70)
    print(f"총 처리 프레임: {frame_count}")
    print(f"고유 Track ID 수: {len(track_ids_seen)}")
    print()
    print(f"📁 결과 저장 위치:")
    print(f"   - 이미지: runs/detect/{TRACK_PROJECT}/{TRACK_NAME_PREFIX}/{video_name}/")
    print(f"   - 텍스트: runs/detect/{TRACK_PROJECT}/{TRACK_NAME_PREFIX}/{video_name}/labels/")
    print("=" * 70)
    
    ### 시작
    # AVI 파일을 MP4로 변환
    output_dir = Path(f"runs/detect/{TRACK_PROJECT}/{TRACK_NAME_PREFIX}/{video_name}")
    avi_file = output_dir / f"{video_name}.avi"
    
    if avi_file.exists():
        mp4_file = convert_avi_to_mp4(avi_file)
        if mp4_file:
            print()
            print(f"✓ 최종 출력 파일: {mp4_file}")
    else:
        print()
        print(f"[Info] AVI 파일을 찾을 수 없습니다: {avi_file}")
        print(f"       (YOLO가 이미 MP4로 저장했거나, 다른 형식으로 저장되었을 수 있습니다)")
    ### 끝

def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("YOLOv26s Vehicle Tracking (Video)")
    print("=" * 70)
    print()
    
    # Step 1: 모델 로드
    print("[Step 1] 모델 로드")
    print("-" * 70)
    print(f"모델: {MODEL_WEIGHT}")
    
    if not os.path.exists(MODEL_WEIGHT):
        print(f"[Error] 모델 파일을 찾을 수 없습니다: {MODEL_WEIGHT}")
        return
    
    model = YOLO(MODEL_WEIGHT)
    print("✓ 모델 로드 완료")
    print()
    
    # Step 2: Tracking 수행
    print("[Step 2] 비디오 Tracking")
    print("-" * 70)
    
    run_tracking_on_video(model, VIDEO_PATH)
    
    print("\n✓ 모든 작업 완료!")


if __name__ == '__main__':
    main()

