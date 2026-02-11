#!/usr/bin/env python3
"""
Val과 Test 폴더명 교체 스크립트

현재: Train 84.5%, Val 4.9%, Test 10.5%
교체 후: Train 84.5%, Val 10.5%, Test 4.9%
"""

import os
import shutil
from pathlib import Path

# ========== 설정 ==========
DATASET_ROOT = "/data/ephemeral/home/dataset/flatten_car_road_dataset_bb"

def main():
    print("\n" + "=" * 70)
    print("🔄 Val ↔ Test 폴더명 교체")
    print("=" * 70)
    print(f"데이터셋 경로: {DATASET_ROOT}")
    print()
    
    val_dir = os.path.join(DATASET_ROOT, "val")
    test_dir = os.path.join(DATASET_ROOT, "test")
    temp_dir = os.path.join(DATASET_ROOT, "_temp_swap")
    
    # Step 1: 폴더 존재 확인
    print("📁 Step 1: 폴더 존재 확인...")
    
    if not os.path.exists(val_dir):
        print(f"  ❌ Val 폴더가 없습니다: {val_dir}")
        return
    
    if not os.path.exists(test_dir):
        print(f"  ❌ Test 폴더가 없습니다: {test_dir}")
        return
    
    if os.path.exists(temp_dir):
        print(f"  ❌ 임시 폴더가 이미 존재합니다: {temp_dir}")
        print(f"     삭제 후 다시 실행하세요.")
        return
    
    print(f"  ✓ Val 폴더 존재: {val_dir}")
    print(f"  ✓ Test 폴더 존재: {test_dir}")
    print()
    
    # Step 2: 현재 상태 출력
    print("📊 Step 2: 현재 상태...")
    
    val_images = len([f for f in os.listdir(os.path.join(val_dir, "images")) 
                      if f.endswith(('.jpg', '.png'))]) if os.path.exists(os.path.join(val_dir, "images")) else 0
    test_images = len([f for f in os.listdir(os.path.join(test_dir, "images")) 
                       if f.endswith(('.jpg', '.png'))]) if os.path.exists(os.path.join(test_dir, "images")) else 0
    
    print(f"  - Val:  {val_images}개 이미지")
    print(f"  - Test: {test_images}개 이미지")
    print()
    
    print("📊 교체 후 예상:")
    print(f"  - Val:  {test_images}개 이미지 (현재 Test)")
    print(f"  - Test: {val_images}개 이미지 (현재 Val)")
    print()
    
    # Step 3: 확인
    print("⚠️  경고: 폴더명을 교체합니다!")
    response = input("계속 진행하시겠습니까? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\n❌ 작업이 취소되었습니다.")
        return
    
    print()
    
    # Step 4: 폴더명 교체
    print("🔄 Step 3: 폴더명 교체 중...")
    
    try:
        # val → _temp_swap
        print(f"  1/3: val → _temp_swap")
        shutil.move(val_dir, temp_dir)
        
        # test → val
        print(f"  2/3: test → val")
        shutil.move(test_dir, val_dir)
        
        # _temp_swap → test
        print(f"  3/3: _temp_swap → test")
        shutil.move(temp_dir, test_dir)
        
        print()
        print("=" * 70)
        print("✅ 교체 완료!")
        print("=" * 70)
        print(f"📊 최종 상태:")
        print(f"  - Val:  {test_images}개 이미지")
        print(f"  - Test: {val_images}개 이미지")
        print()
        print("📝 다음 단계:")
        print("  1. python verify_split.py - 검증")
        print("  2. python check_split_ratio.py - 비율 확인")
        print("=" * 70)
        print()
        
    except Exception as e:
        print(f"\n[Error] 폴더 교체 중 오류 발생: {e}")
        print()
        print("복구를 시도합니다...")
        
        # 복구 시도
        if os.path.exists(temp_dir):
            if not os.path.exists(val_dir):
                shutil.move(temp_dir, val_dir)
                print("  ✓ Val 복구 완료")
        
        print()
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ 작업이 중단되었습니다.")
    except Exception as e:
        print(f"\n[Error] 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

