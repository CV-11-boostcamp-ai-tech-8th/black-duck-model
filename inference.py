#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv26s Vehicle Detection - Inference
- Fine-tuned vehicle detection 모델로 추론 수행
- 비디오 파일 또는 이미지 시퀀스 지원
- 모든 설정은 YAML 파일을 통해 전달됨
"""

import os
import re
import subprocess
import time
import argparse
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download


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
        # -an: 오디오 스트림 제거 (inference 영상에는 오디오 불필요)
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


def find_sequences(image_dir):
    """
    이미지 디렉토리에서 시퀀스별로 그룹화
    
    Args:
        image_dir: 이미지 디렉토리 경로
        
    Returns:
        dict: {timestamp: [image_paths]}
    """
    image_dir = Path(image_dir)
    
    if not image_dir.exists():
        print(f"[Error] 이미지 디렉토리가 존재하지 않습니다: {image_dir}")
        return {}
    
    # 모든 jpg 파일 찾기
    images = sorted(image_dir.glob("*.jpg"))
    
    if not images:
        print(f"[Error] 이미지 파일을 찾을 수 없습니다: {image_dir}")
        return {}
    
    # 시퀀스별로 그룹화
    sequences = defaultdict(list)
    pattern = re.compile(r'.*_(\d{8}_\d{6})_\d+\.jpg')
    
    for img in images:
        match = pattern.match(img.name)
        if match:
            timestamp = match.group(1)
            sequences[timestamp].append(str(img))
    
    return dict(sequences)


def display_sequences(sequences):
    """
    사용 가능한 시퀀스 목록 출력
    
    Args:
        sequences: 시퀀스 딕셔너리
    """
    print("\n" + "=" * 70)
    print("사용 가능한 비디오 시퀀스")
    print("=" * 70)
    
    sorted_keys = sorted(sequences.keys())
    
    for i, timestamp in enumerate(sorted_keys, 1):
        date = timestamp[:8]  # YYYYMMDD
        time = timestamp[9:]   # HHMMSS
        frame_count = len(sequences[timestamp])
        
        # 날짜와 시간 포맷팅
        formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        formatted_time = f"{time[:2]}:{time[2:4]}:{time[4:6]}"
        
        print(f"  [{i:2d}] {timestamp}")
        print(f"       날짜: {formatted_date}, 시간: {formatted_time}")
        print(f"       프레임 수: {frame_count}개")
        print()
    
    print("=" * 70)
    return sorted_keys


def select_sequence(sequences):
    """
    사용자에게 시퀀스 선택 요청
    
    Args:
        sequences: 시퀀스 딕셔너리
        
    Returns:
        tuple: (timestamp, image_paths)
    """
    sorted_keys = display_sequences(sequences)
    
    while True:
        try:
            choice = input(f"\n추론할 시퀀스 번호를 입력하세요 (1-{len(sorted_keys)}): ")
            idx = int(choice) - 1
            
            if 0 <= idx < len(sorted_keys):
                selected_timestamp = sorted_keys[idx]
                print(f"\n✓ 선택된 시퀀스: {selected_timestamp}")
                return selected_timestamp, sequences[selected_timestamp]
            else:
                print(f"[Error] 1에서 {len(sorted_keys)} 사이의 숫자를 입력하세요.")
        except ValueError:
            print("[Error] 올바른 숫자를 입력하세요.")
        except KeyboardInterrupt:
            print("\n\n프로그램을 종료합니다.")
            exit(0)


def run_video_inference(model, video_path, cfg):
    """
    비디오 파일로 추론 수행
    
    Args:
        model: YOLO 모델
        video_path: 비디오 파일 경로
        cfg: OmegaConf 설정 객체
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        print(f"[Error] 비디오 파일을 찾을 수 없습니다: {video_path}")
        return
    
    video_name = video_path.stem
    inference_name_prefix = f"yolo26x_{cfg.version}"
    inference_project = cfg.get('inference_project', 'inference')
    
    print("\n" + "=" * 70)
    print(f"추론 시작: {video_name}")
    print("=" * 70)
    print(f"비디오: {video_path}")
    print(f"Confidence threshold: {cfg.conf_threshold}")
    print(f"IoU threshold: {cfg.iou_threshold}")
    print()
    
    # 저장 경로 구성: inference/yolo26x_{VERSION}/{video_name}
    project_path = f"{inference_project}/{inference_name_prefix}"
    
    # 시간 측정 시작
    start_time = time.time()
    
    # 추론 수행
    results = model.predict(
        source=str(video_path),
        conf=cfg.conf_threshold,
        iou=cfg.iou_threshold,
        save=True,  # 결과 비디오 저장
        save_txt=True,  # 결과 텍스트 저장
        project=project_path,
        name=video_name,
        exist_ok=True,
        stream=True,  # 메모리 효율적 처리
    )
    
    # 결과 처리 및 통계
    frame_count = 0
    total_detections = 0
    detections_per_frame = []
    
    # YOLO speed 정보 수집
    total_preprocess = 0
    total_inference = 0
    total_postprocess = 0
    
    for result in results:
        frame_count += 1
        num_boxes = len(result.boxes)
        total_detections += num_boxes
        detections_per_frame.append(num_boxes)
        
        # YOLO speed 정보 수집
        if hasattr(result, 'speed') and result.speed:
            total_preprocess += result.speed.get('preprocess', 0)
            total_inference += result.speed.get('inference', 0)
            total_postprocess += result.speed.get('postprocess', 0)
        
        # 진행 상황 출력 (100 프레임마다)
        if frame_count % 100 == 0:
            print(f"  처리 중... {frame_count} 프레임")
    
    # 시간 측정 종료
    end_time = time.time()
    total_time = end_time - start_time
    time_per_frame = total_time / frame_count if frame_count > 0 else 0
    
    # YOLO speed 평균 계산
    avg_preprocess = total_preprocess / frame_count if frame_count > 0 else 0
    avg_inference = total_inference / frame_count if frame_count > 0 else 0
    avg_postprocess = total_postprocess / frame_count if frame_count > 0 else 0
    yolo_total_time = avg_preprocess + avg_inference + avg_postprocess
    yolo_fps = 1000 / yolo_total_time if yolo_total_time > 0 else 0
    
    # 통계 계산
    avg_detections = total_detections / frame_count if frame_count > 0 else 0
    max_detections = max(detections_per_frame) if detections_per_frame else 0
    min_detections = min(detections_per_frame) if detections_per_frame else 0
    
    print()
    print("=" * 70)
    print("추론 완료!")
    print("=" * 70)
    print(f"총 처리 프레임: {frame_count}")
    print(f"총 검출된 vehicle 수: {total_detections}")
    print(f"프레임당 평균 검출 수: {avg_detections:.2f}")
    print(f"최대 검출 수 (1 프레임): {max_detections}")
    print(f"최소 검출 수 (1 프레임): {min_detections}")
    print()
    print("⏱️  처리 시간:")
    if yolo_total_time > 0:
        print(f"   - YOLO: {avg_preprocess:.1f} + {avg_inference:.1f} + {avg_postprocess:.1f} = {yolo_total_time:.1f}ms ({yolo_fps:.1f}FPS)")
    print(f"   - Python: {time_per_frame*1000:.1f}ms ({1/time_per_frame:.1f}FPS)")
    print(f"   - 총 시간: {total_time:.2f}s")
    print()
    print(f"📁 결과 저장 위치:")
    print(f"   - 비디오: {inference_project}/{inference_name_prefix}/{video_name}/")
    print(f"   - 텍스트: {inference_project}/{inference_name_prefix}/{video_name}/labels/")
    print("=" * 70)
    
    # AVI 파일을 MP4로 변환 (설정에 따라)
    convert_avi = cfg.get('convert_avi_to_mp4', True)
    if convert_avi:
        output_dir = Path(f"runs/detect/{project_path}/{video_name}")
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
    else:
        print()
        print(f"[Info] AVI → MP4 변환 스킵 (convert_avi_to_mp4 = False)")


def run_inference(model, images, sequence_name, cfg):
    """
    선택한 시퀀스로 추론 수행
    
    Args:
        model: YOLO 모델
        images: 이미지 경로 리스트
        sequence_name: 시퀀스 이름 (timestamp)
        cfg: OmegaConf 설정 객체
    """
    inference_name_prefix = f"yolo26x_{cfg.version}"
    inference_project = cfg.get('inference_project', 'inference')
    
    print("\n" + "=" * 70)
    print(f"추론 시작: {sequence_name}")
    print("=" * 70)
    print(f"프레임 수: {len(images)}")
    print(f"Confidence threshold: {cfg.conf_threshold}")
    print(f"IoU threshold: {cfg.iou_threshold}")
    print()
    
    # 저장 경로 구성: inference/yolo26s_{VERSION}/{seq_name}
    project_path = f"{inference_project}/{inference_name_prefix}"
    
    # 시간 측정 시작
    start_time = time.time()
    
    # 추론 수행
    results = model.predict(
        source=images,
        conf=cfg.conf_threshold,
        iou=cfg.iou_threshold,
        save=True,  # 결과 이미지 저장
        save_txt=True,  # 결과 텍스트 저장
        project=project_path,
        name=sequence_name,
        exist_ok=True,
        stream=True,  # 메모리 효율적 처리
    )
    
    # 결과 처리 및 통계
    frame_count = 0
    total_detections = 0
    detections_per_frame = []
    
    # YOLO speed 정보 수집
    total_preprocess = 0
    total_inference = 0
    total_postprocess = 0
    
    for result in results:
        frame_count += 1
        num_boxes = len(result.boxes)
        total_detections += num_boxes
        detections_per_frame.append(num_boxes)
        
        # YOLO speed 정보 수집
        if hasattr(result, 'speed') and result.speed:
            total_preprocess += result.speed.get('preprocess', 0)
            total_inference += result.speed.get('inference', 0)
            total_postprocess += result.speed.get('postprocess', 0)
        
        # 진행 상황 출력 (100 프레임마다)
        if frame_count % 100 == 0:
            print(f"  처리 중... {frame_count}/{len(images)} 프레임")
    
    # 시간 측정 종료
    end_time = time.time()
    total_time = end_time - start_time
    time_per_frame = total_time / frame_count if frame_count > 0 else 0
    
    # YOLO speed 평균 계산
    avg_preprocess = total_preprocess / frame_count if frame_count > 0 else 0
    avg_inference = total_inference / frame_count if frame_count > 0 else 0
    avg_postprocess = total_postprocess / frame_count if frame_count > 0 else 0
    yolo_total_time = avg_preprocess + avg_inference + avg_postprocess
    yolo_fps = 1000 / yolo_total_time if yolo_total_time > 0 else 0
    
    # 통계 계산
    avg_detections = total_detections / frame_count if frame_count > 0 else 0
    max_detections = max(detections_per_frame) if detections_per_frame else 0
    min_detections = min(detections_per_frame) if detections_per_frame else 0
    
    print()
    print("=" * 70)
    print("추론 완료!")
    print("=" * 70)
    print(f"총 처리 프레임: {frame_count}")
    print(f"총 검출된 vehicle 수: {total_detections}")
    print(f"프레임당 평균 검출 수: {avg_detections:.2f}")
    print(f"최대 검출 수 (1 프레임): {max_detections}")
    print(f"최소 검출 수 (1 프레임): {min_detections}")
    print()
    print("⏱️  처리 시간:")
    if yolo_total_time > 0:
        print(f"   - YOLO: {avg_preprocess:.1f} + {avg_inference:.1f} + {avg_postprocess:.1f} = {yolo_total_time:.1f}ms ({yolo_fps:.1f}FPS)")
    print(f"   - Python: {time_per_frame*1000:.1f}ms ({1/time_per_frame:.1f}FPS)")
    print(f"   - 총 시간: {total_time:.2f}s")
    print()
    print(f"📁 결과 저장 위치:")
    print(f"   - 이미지: {inference_project}/{inference_name_prefix}/{sequence_name}/")
    print(f"   - 텍스트: {inference_project}/{inference_name_prefix}/{sequence_name}/labels/")
    print("=" * 70)


def main(cfg):
    """메인 실행 함수"""
    mode = cfg.get('mode', 'VIDEO')  # 기본값: VIDEO
    
    print("=" * 70)
    print("YOLOv26s Vehicle Detection - Inference")
    print("=" * 70)
    print(f"모드: {mode}")
    print(f"버전: {cfg.version}")
    print()
    
    # Step 1: 모델 로드
    print("[Step 1] Fine-tuned 모델 로드")
    print("-" * 70)
    print(f"모델: {cfg.model_weight}")
    
    model = YOLO(cfg.model_weight)
    print("✓ 모델 로드 완료")
    print()
    
    if mode == "VIDEO":
        # ========== VIDEO 모드 ==========
        print("[Step 2] 비디오 파일 확인")
        print("-" * 70)
        print(f"비디오: {cfg.video_path}")
        
        if not Path(cfg.video_path).exists():
            print(f"[Error] 비디오 파일을 찾을 수 없습니다: {cfg.video_path}")
            return
        
        print("✓ 비디오 파일 확인 완료")
        
        # 추론 수행
        run_video_inference(model, cfg.video_path, cfg)
        
    elif mode == "IMAGE":
        # ========== IMAGE 모드 ==========
        print("[Step 2] 이미지 시퀀스 탐색")
        print("-" * 70)
        
        image_dir = cfg.get('image_dir', '/data/ephemeral/home/dataset/flatten_road_dataset_bb/test/images')
        sequences = find_sequences(image_dir)
        
        if not sequences:
            print("[Error] 시퀀스를 찾을 수 없습니다.")
            return
        
        print(f"✓ {len(sequences)}개의 시퀀스를 찾았습니다.")
        
        # Step 3: 시퀀스 선택
        print("\n[Step 3] 시퀀스 선택")
        print("-" * 70)
        
        selected_sequence = cfg.get('selected_sequence', None)
        if selected_sequence:
            # 설정 변수로 지정된 시퀀스 사용
            if selected_sequence in sequences:
                timestamp = selected_sequence
                images = sequences[timestamp]
                print(f"✓ 사전 선택된 시퀀스: {timestamp} ({len(images)} 프레임)")
            else:
                print(f"[Error] 지정된 시퀀스를 찾을 수 없습니다: {selected_sequence}")
                print(f"사용 가능한 시퀀스: {list(sequences.keys())[:5]}...")
                return
        else:
            # 대화형으로 선택
            timestamp, images = select_sequence(sequences)
        
        # Step 4: 추론 수행
        run_inference(model, images, timestamp, cfg)
    
    else:
        print(f"[Error] 잘못된 MODE 값: {mode}")
        print("MODE는 'VIDEO' 또는 'IMAGE'여야 합니다.")
        return
    
    print("\n✓ 모든 작업 완료!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO Inference with YAML config')
    parser.add_argument('config', type=str, help='YAML 설정 파일 경로')
    
    args = parser.parse_args()
    
    # YAML 설정 로드
    cfg = OmegaConf.load(args.config)
    
    main(cfg)

