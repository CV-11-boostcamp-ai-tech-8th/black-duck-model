#!/usr/bin/env python3
"""
데이터셋 분할 비율 확인 스크립트
"""

import os
from pathlib import Path
from collections import defaultdict

DATASET_ROOT = "/data/ephemeral/home/dataset/flatten_car_road_dataset_bb"

def extract_scene_id(filename):
    """파일명에서 장면 ID 추출"""
    stem = Path(filename).stem
    
    # 1단계: XML 접미사 제거 (_v001_1 같은 패턴)
    if '_v' in stem:
        parts = stem.rsplit('_v', 1)
        if len(parts) == 2:
            stem = parts[0]
    
    # 2단계: _I숫자 패턴
    if '_I' in stem:
        return stem.rsplit('_I', 1)[0]
    
    # 3단계: 마지막 _숫자 패턴 제거
    parts = stem.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    
    return stem

def count_scenes(split_dir):
    """해당 split의 장면 수 카운트"""
    voc_dir = os.path.join(split_dir, 'voc_annotations')
    
    if not os.path.exists(voc_dir):
        return 0, 0
    
    scenes = set()
    file_count = 0
    
    for filename in os.listdir(voc_dir):
        if filename.endswith('.xml'):
            scene_id = extract_scene_id(filename)
            scenes.add(scene_id)
            file_count += 1
    
    return len(scenes), file_count

def main():
    print("\n" + "=" * 70)
    print("📊 데이터셋 분할 비율")
    print("=" * 70)
    print(f"데이터셋 경로: {DATASET_ROOT}")
    print()
    
    results = {}
    
    for split_name in ['train', 'val', 'test']:
        split_dir = os.path.join(DATASET_ROOT, split_name)
        scene_count, file_count = count_scenes(split_dir)
        results[split_name] = {
            'scenes': scene_count,
            'files': file_count,
        }
        
        if scene_count > 0:
            print(f"📁 {split_name.upper():<6}")
            print(f"   - 장면: {scene_count}개")
            print(f"   - 파일: {file_count}개")
        else:
            print(f"📁 {split_name.upper():<6}: 존재하지 않음")
        print()
    
    # 총계
    total_scenes = sum(r['scenes'] for r in results.values())
    total_files = sum(r['files'] for r in results.values())
    
    if total_scenes == 0:
        print("[Error] 장면이 없습니다!")
        return
    
    print("-" * 70)
    print(f"📈 전체")
    print(f"   - 총 장면: {total_scenes}개")
    print(f"   - 총 파일: {total_files}개")
    print()
    
    print("-" * 70)
    print("📊 장면 비율")
    print("-" * 70)
    for split_name in ['train', 'val', 'test']:
        scene_count = results[split_name]['scenes']
        if scene_count > 0:
            ratio = (scene_count / total_scenes) * 100
            print(f"  {split_name.upper():<6}: {ratio:>6.2f}%  ({scene_count}개)")
    
    print()
    
    print("-" * 70)
    print("📊 파일 비율")
    print("-" * 70)
    for split_name in ['train', 'val', 'test']:
        file_count = results[split_name]['files']
        if file_count > 0:
            ratio = (file_count / total_files) * 100
            print(f"  {split_name.upper():<6}: {ratio:>6.2f}%  ({file_count}개)")
    
    print()
    print("=" * 70)
    print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[Error] 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

