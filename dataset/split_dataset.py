#!/usr/bin/env python3
"""
Train 데이터셋을 장면(폴더) 단위로 85:5:10 비율로 Train:Val:Test 분할
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# ========== 설정 ==========
DATASET_ROOT = "/data/ephemeral/home/dataset/flatten_car_road_dataset_bb"
SEED = 42
random.seed(SEED)

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
    
    scenes = defaultdict(lambda: {'images': [], 'labels': [], 'voc': []})
    
    # voc_annotations 파일 수집
    for filename in sorted(os.listdir(voc_dir)):
        if filename.endswith('.xml'):
            scene_id = extract_scene_id(filename)
            scenes[scene_id]['voc'].append(filename)
    
    # images 파일 수집
    image_dir = os.path.join(split_dir, 'images')
    if os.path.exists(image_dir):
        for filename in sorted(os.listdir(image_dir)):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                scene_id = extract_scene_id(filename)
                scenes[scene_id]['images'].append(filename)
    
    # labels 파일 수집 (있다면)
    label_dir = os.path.join(split_dir, 'labels')
    if os.path.exists(label_dir):
        for filename in sorted(os.listdir(label_dir)):
            if filename.endswith('.txt'):
                scene_id = extract_scene_id(filename)
                scenes[scene_id]['labels'].append(filename)
    
    return dict(sorted(scenes.items()))

def move_scene_files(scene_id, files_dict, src_split, dst_split):
    """한 장면의 모든 파일을 이동"""
    src_base = os.path.join(DATASET_ROOT, src_split)
    dst_base = os.path.join(DATASET_ROOT, dst_split)
    
    # 디렉토리 생성
    for subdir in ['images', 'labels', 'voc_annotations']:
        os.makedirs(os.path.join(dst_base, subdir), exist_ok=True)
    
    moved = defaultdict(int)
    
    # voc_annotations 이동
    for filename in files_dict.get('voc', []):
        src = os.path.join(src_base, 'voc_annotations', filename)
        dst = os.path.join(dst_base, 'voc_annotations', filename)
        if os.path.exists(src):
            shutil.move(src, dst)
            moved['voc'] += 1
    
    # images 이동
    for filename in files_dict.get('images', []):
        src = os.path.join(src_base, 'images', filename)
        dst = os.path.join(dst_base, 'images', filename)
        if os.path.exists(src):
            shutil.move(src, dst)
            moved['images'] += 1
    
    # labels 이동 (있다면)
    for filename in files_dict.get('labels', []):
        src = os.path.join(src_base, 'labels', filename)
        dst = os.path.join(dst_base, 'labels', filename)
        if os.path.exists(src):
            shutil.move(src, dst)
            moved['labels'] += 1
    
    return moved

def main():
    print("\n" + "=" * 70)
    print("📦 데이터셋 분할 (Train → Test)")
    print("=" * 70)
    print(f"현재: Train 95%, Val 5%")
    print(f"목표: Train 90%, Val 5%, Test 5%")
    print(f"Random Seed: {SEED}")
    print()
    
    # Step 1: 현재 train의 장면 그룹 가져오기
    train_dir = os.path.join(DATASET_ROOT, 'train')
    print("📁 Step 1: Train 장면 분석...")
    
    train_scenes = get_scene_groups(train_dir)
    train_scene_count = len(train_scenes)
    train_image_count = sum(len(s['images']) for s in train_scenes.values())
    
    print(f"  - Train: {train_scene_count}개 장면, {train_image_count}개 이미지")
    print()
    
    # Step 2: Val 확인
    val_dir = os.path.join(DATASET_ROOT, 'val')
    val_scenes = get_scene_groups(val_dir)
    val_scene_count = len(val_scenes)
    val_image_count = sum(len(s['images']) for s in val_scenes.values()) if val_scenes else 0
    
    print(f"  - Val:   {val_scene_count}개 장면, {val_image_count}개 이미지 (유지)")
    print()
    
    # Step 3: 분할 계산 (이미지 개수 기준으로 5%에 가장 가까운 장면 조합 찾기)
    print("🎯 Step 2: Test 분할 계산...")
    
    # 전체 장면 수
    total_scenes = train_scene_count + val_scene_count
    total_images = train_image_count + val_image_count
    
    # 목표: 전체의 5%를 Test로
    target_test_images = int(total_images * 0.05)
    
    print(f"  - 전체: {total_scenes}개 장면, {total_images}개 이미지")
    print(f"  - 목표 Test 이미지: {target_test_images}개 (5%)")
    print()
    
    # Step 4: 랜덤 선택 (이미지 개수 기준으로 5%에 가깝게)
    print("🎲 Step 3: 장면 선택 (이미지 개수 기준)...")
    
    scene_ids = list(train_scenes.keys())
    random.shuffle(scene_ids)
    
    # 탐욕적으로 5%에 가까운 장면 조합 찾기
    test_scene_ids = []
    current_test_images = 0
    
    for scene_id in scene_ids:
        scene_image_count = len(train_scenes[scene_id]['images'])
        
        # 현재 장면을 추가했을 때 목표에 더 가까워지면 추가
        if current_test_images + scene_image_count <= target_test_images * 1.15:  # 15% 여유
            test_scene_ids.append(scene_id)
            current_test_images += scene_image_count
            
            # 목표에 충분히 가까우면 중단
            if current_test_images >= target_test_images * 0.95:  # 95% 이상이면 OK
                break
    
    test_image_count = current_test_images
    
    print(f"  - Test용: {len(test_scene_ids)}개 장면 선택 (약 {test_image_count}개 이미지)")
    print(f"  - Train 잔여: {train_scene_count - len(test_scene_ids)}개 장면")
    print()
    
    # 최종 비율 예상 (이미지 기준)
    print("📊 예상 최종 비율 (이미지 기준):")
    final_train_images = train_image_count - test_image_count
    final_val_images = val_image_count
    final_test_images = test_image_count
    final_total_images = final_train_images + final_val_images + final_test_images
    
    print(f"  - Train: {final_train_images:>6,}개 이미지 ({final_train_images/final_total_images*100:>5.1f}%)")
    print(f"  - Val:   {final_val_images:>6,}개 이미지 ({final_val_images/final_total_images*100:>5.1f}%)")
    print(f"  - Test:  {final_test_images:>6,}개 이미지 ({final_test_images/final_total_images*100:>5.1f}%)")
    print()
    
    # Step 5: 확인
    print("⚠️  경고: Train → Test로 파일을 이동(move)합니다!")
    response = input("계속 진행하시겠습니까? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\n❌ 작업이 취소되었습니다.")
        return
    
    print()
    
    # Step 6: Test로 이동
    print("🚚 Step 4: Train → Test 이동 중...")
    test_total = defaultdict(int)
    for i, scene_id in enumerate(test_scene_ids, 1):
        moved = move_scene_files(scene_id, train_scenes[scene_id], 'train', 'test')
        for k, v in moved.items():
            test_total[k] += v
        if i % 10 == 0 or i == len(test_scene_ids):
            print(f"  [{i}/{len(test_scene_ids)}] 처리 중...")
    
    print(f"  ✓ Test 이동 완료: images {test_total['images']}, voc {test_total['voc']}, labels {test_total['labels']}")
    print()
    
    # 완료
    print("=" * 70)
    print("✅ Train → Test 이동 완료!")
    print("=" * 70)
    print(f"📊 이동 통계:")
    print(f"  - 장면: {len(test_scene_ids)}개")
    print(f"  - Images: {test_total['images']}개")
    print(f"  - VOC: {test_total['voc']}개")
    print(f"  - Labels: {test_total['labels']}개")
    print()
    print("📝 다음 단계:")
    print("  1. python verify_split.py - 검증")
    print("  2. python check_split_ratio.py - 비율 확인")
    print("  3. python custom_voc2yolo.py - VOC → YOLO 변환 (test 포함)")
    print("=" * 70)
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ 작업이 중단되었습니다.")
    except Exception as e:
        print(f"\n[Error] 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

