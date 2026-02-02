#!/usr/bin/env python3
"""
데이터셋 분할 검증 스크립트
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

def get_scene_groups(split_dir):
    """해당 split의 장면 그룹 가져오기"""
    voc_dir = os.path.join(split_dir, 'voc_annotations')
    
    if not os.path.exists(voc_dir):
        return {}
    
    scenes = defaultdict(lambda: {'images': 0, 'labels': 0, 'voc': 0})
    
    # voc_annotations 카운트
    for filename in os.listdir(voc_dir):
        if filename.endswith('.xml'):
            scene_id = extract_scene_id(filename)
            scenes[scene_id]['voc'] += 1
    
    # images 카운트
    image_dir = os.path.join(split_dir, 'images')
    if os.path.exists(image_dir):
        for filename in os.listdir(image_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                scene_id = extract_scene_id(filename)
                scenes[scene_id]['images'] += 1
    
    # labels 카운트
    label_dir = os.path.join(split_dir, 'labels')
    if os.path.exists(label_dir):
        for filename in os.listdir(label_dir):
            if filename.endswith('.txt'):
                scene_id = extract_scene_id(filename)
                scenes[scene_id]['labels'] += 1
    
    return dict(scenes)

def main():
    print("\n" + "=" * 70)
    print("🔍 데이터셋 분할 검증")
    print("=" * 70)
    print()
    
    splits_info = {}
    
    for split_name in ['train', 'val', 'test']:
        split_dir = os.path.join(DATASET_ROOT, split_name)
        scenes = get_scene_groups(split_dir)
        
        if not scenes:
            print(f"❌ {split_name.upper()}: 존재하지 않음")
            splits_info[split_name] = None
            continue
        
        total_images = sum(s['images'] for s in scenes.values())
        total_voc = sum(s['voc'] for s in scenes.values())
        total_labels = sum(s['labels'] for s in scenes.values())
        
        splits_info[split_name] = {
            'scenes': scenes,
            'scene_count': len(scenes),
            'image_count': total_images,
            'voc_count': total_voc,
            'label_count': total_labels,
        }
        
        print(f"✅ {split_name.upper():<6}")
        print(f"   - 장면: {len(scenes)}개")
        print(f"   - 이미지: {total_images}개")
        print(f"   - VOC: {total_voc}개")
        print(f"   - Labels: {total_labels}개")
        print()
    
    # 중복 검사
    print("-" * 70)
    print("🔍 장면 중복 검사")
    print("-" * 70)
    
    all_splits = [(name, info) for name, info in splits_info.items() if info]
    
    overlaps = []
    for i in range(len(all_splits)):
        for j in range(i+1, len(all_splits)):
            name1, info1 = all_splits[i]
            name2, info2 = all_splits[j]
            
            scenes1 = set(info1['scenes'].keys())
            scenes2 = set(info2['scenes'].keys())
            overlap = scenes1 & scenes2
            
            if overlap:
                overlaps.append((name1, name2, overlap))
    
    if overlaps:
        print("⚠️  장면 중복 발견!")
        for name1, name2, overlap in overlaps:
            print(f"  {name1.upper()} ↔ {name2.upper()}: {len(overlap)}개 중복")
            for scene_id in list(overlap)[:3]:
                print(f"    - {scene_id}")
            if len(overlap) > 3:
                print(f"    ... 외 {len(overlap)-3}개")
    else:
        print("✅ 장면 중복 없음 (Train/Val/Test 완전 분리)")
    
    print()
    
    # 일관성 검사
    print("-" * 70)
    print("🔍 파일 일관성 검사")
    print("-" * 70)
    
    for split_name, info in splits_info.items():
        if not info:
            continue
        
        inconsistent = []
        for scene_id, counts in info['scenes'].items():
            if counts['images'] != counts['voc']:
                inconsistent.append(scene_id)
        
        if inconsistent:
            print(f"⚠️  {split_name.upper()}: {len(inconsistent)}개 장면에서 images/voc 불일치")
            for scene_id in inconsistent[:3]:
                c = info['scenes'][scene_id]
                print(f"    - {scene_id}: images={c['images']}, voc={c['voc']}")
            if len(inconsistent) > 3:
                print(f"    ... 외 {len(inconsistent)-3}개")
        else:
            print(f"✅ {split_name.upper()}: 모든 장면 일관성 OK")
    
    print()
    print("=" * 70)
    print("✅ 검증 완료")
    print("=" * 70)
    print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[Error] 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

