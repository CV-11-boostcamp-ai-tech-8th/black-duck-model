#!/usr/bin/env python3
"""
connect_final.py 실제 테스트 스크립트 (sudden_stop + OCR)

기능:
- ssh_input의 기존 job.json 사용
- input_url에서 비디오 다운로드 (실제로)
- GCS 업로드 없이 로컬에만 저장 (--bucket 옵션 없이 실행)
- ssh_output에 결과 저장
- result.json 검증
"""

import json
import subprocess
import sys
from pathlib import Path


def run_connect_final(job_json_path: Path):
    """
    connect_final.py 실행 (GCS 업로드 없이)
    """
    connect_script = Path(__file__).parent / "connect_final.py"
    
    if not connect_script.exists():
        raise FileNotFoundError(f"connect_final.py를 찾을 수 없습니다: {connect_script}")
    
    # --bucket을 지정하지 않으면 GCS 업로드 안 함 (로컬 경로만 사용)
    cmd = [
        sys.executable,
        str(connect_script),
        "--job", str(job_json_path),
        # --bucket을 지정하지 않음 → bucket_name이 None → 로컬 경로만 사용
    ]
    
    print("=" * 60)
    print("connect_final.py 실행 중...")
    print("=" * 60)
    print(f"[command] {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        raise RuntimeError(f"connect_final.py 실행 실패 (exit code: {result.returncode})")
    
    print()
    print("[ok] connect_final.py 실행 완료")


def verify_results(job_json_path: Path):
    """
    결과 검증
    """
    # job.json 읽기
    job = json.loads(job_json_path.read_text(encoding="utf-8"))
    job_id = job["job_id"]
    out_root = Path(job["out_root"])
    output_task_dir = out_root / job_id
    
    print("=" * 60)
    print("결과 검증")
    print("=" * 60)
    print(f"[job_id] {job_id}")
    print(f"[output_dir] {output_task_dir}")
    print()
    
    # 1. result.json 확인
    result_json_path = output_task_dir / "result.json"
    if not result_json_path.exists():
        raise FileNotFoundError(f"result.json이 생성되지 않았습니다: {result_json_path}")
    
    print(f"[ok] result.json 존재: {result_json_path}")
    
    result = json.loads(result_json_path.read_text(encoding="utf-8"))
    
    # 2. result.json 구조 확인
    required_keys = ["job_id", "task_id", "video_id", "results"]
    for key in required_keys:
        if key not in result:
            raise ValueError(f"result.json에 '{key}' 키가 없습니다")
    
    print(f"[ok] result.json 구조 검증 통과")
    print(f"[info] job_id: {result['job_id']}")
    print(f"[info] task_id: {result.get('task_id')}")
    print(f"[info] video_id: {result.get('video_id')}")
    print(f"[info] results 개수: {len(result['results'])}")
    print()
    
    # 3. 각 clip 결과 확인
    for i, clip_result in enumerate(result['results'], start=1):
        print(f"[clip {i}]")
        print(f"  event_type_id: {clip_result.get('event_type_id')}")
        
        # clip_path 확인 (로컬 경로여야 함)
        clip_path = clip_result.get('clip_path', '')
        print(f"  clip_path: {clip_path}")
        if clip_path.startswith('http'):
            print(f"    [warning] clip_path가 URL입니다 (로컬 경로여야 함)")
        else:
            clip_path_obj = Path(clip_path)
            if clip_path_obj.exists():
                size_kb = clip_path_obj.stat().st_size / 1024
                print(f"    [ok] 파일 존재 ({size_kb:.2f} KB)")
            else:
                print(f"    [error] 파일 없음!")
        
        # thumbnail_img 확인
        thumb_path = clip_result.get('thumbnail_img', '')
        print(f"  thumbnail_img: {thumb_path}")
        if thumb_path and not thumb_path.startswith('http'):
            thumb_path_obj = Path(thumb_path)
            if thumb_path_obj.exists():
                size_kb = thumb_path_obj.stat().st_size / 1024
                print(f"    [ok] 파일 존재 ({size_kb:.2f} KB)")
            else:
                print(f"    [error] 파일 없음!")
        
        # license_plate_img 확인
        plate_path = clip_result.get('license_plate_img', '')
        print(f"  license_plate_img: {plate_path}")
        if plate_path and not plate_path.startswith('http'):
            plate_path_obj = Path(plate_path)
            if plate_path_obj.exists():
                size_kb = plate_path_obj.stat().st_size / 1024
                print(f"    [ok] 파일 존재 ({size_kb:.2f} KB)")
            else:
                print(f"    [error] 파일 없음!")
        
        # license_plate_text 확인
        plate_text = clip_result.get('license_plate_text', '')
        print(f"  license_plate_text: {plate_text}")
        if plate_text:
            print(f"    [ok] 텍스트 인식됨")
        else:
            print(f"    [warning] 텍스트 없음")
        
        print()
    
    # 4. clip 디렉토리 확인
    clip_dirs = sorted([d for d in output_task_dir.iterdir() if d.is_dir() and d.name.startswith("clip_")])
    print(f"[info] clip 디렉토리 개수: {len(clip_dirs)}")
    for clip_dir in clip_dirs:
        print(f"  - {clip_dir.name}")
        # clip 디렉토리 내 파일 확인
        files = sorted(clip_dir.iterdir())
        print(f"    파일 개수: {len(files)}")
        for f in files:
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                print(f"      - {f.name} ({size_kb:.2f} KB)")
        
        # license_plate_text.txt 확인
        plate_text_file = clip_dir / "license_plate_text.txt"
        if plate_text_file.exists():
            plate_text = plate_text_file.read_text(encoding="utf-8").strip()
            print(f"      - license_plate_text.txt: '{plate_text}'")
        else:
            print(f"      - license_plate_text.txt: 없음")
        
        # license_plate_image.png 확인
        plate_image_files = [
            clip_dir / "license_plate_image.png",
            clip_dir / "license_plate_image.jpg",
            clip_dir / "license_plate_image.jpeg",
        ]
        plate_image_found = False
        for plate_img in plate_image_files:
            if plate_img.exists():
                size_kb = plate_img.stat().st_size / 1024
                print(f"      - {plate_img.name} ({size_kb:.2f} KB)")
                plate_image_found = True
                break
        if not plate_image_found:
            print(f"      - license_plate_image.*: 없음")


def main():
    import argparse
    
    ap = argparse.ArgumentParser(description="connect_final.py 실제 테스트 (sudden_stop + OCR)")
    ap.add_argument(
        "--job",
        required=True,
        help="job.json 경로 (예: ssh_input/task1_1770046870_fd24a5fc/job.json)"
    )
    args = ap.parse_args()
    
    job_json_path = Path(args.job)
    if not job_json_path.exists():
        raise FileNotFoundError(f"job.json을 찾을 수 없습니다: {job_json_path}")
    
    print("=" * 60)
    print("connect_final.py 실제 테스트 (sudden_stop + OCR)")
    print("=" * 60)
    print(f"[job.json] {job_json_path}")
    print()
    
    # job.json 내용 확인
    job = json.loads(job_json_path.read_text(encoding="utf-8"))
    print(f"[job_id] {job.get('job_id')}")
    print(f"[input_url] {job.get('input_url', 'N/A')}")
    print(f"[input_path] {job.get('input_path', 'N/A')}")
    print(f"[out_root] {job.get('out_root', 'N/A')}")
    print()
    
    # 1. connect_final.py 실행
    print("[1단계] connect_final.py 실행")
    run_connect_final(job_json_path)
    print()
    
    # 2. 결과 검증
    print("[2단계] 결과 검증")
    verify_results(job_json_path)
    print()
    
    print("=" * 60)
    print("테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

