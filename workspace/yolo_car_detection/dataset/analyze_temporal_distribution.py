#!/usr/bin/env python3
"""
데이터셋의 시간적 분포 분석 (월별/계절별)
VOC annotations 파일명에서 날짜 정보를 추출하여 분석
"""

import os
import re
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

# ========== 설정 ==========
DATASET_ROOT = "/data/ephemeral/home/dataset/flatten_car_road_dataset_bb"

# 계절 정의 (월 기준)
SEASONS = {
    "봄 (3-5월)": [3, 4, 5],
    "여름 (6-8월)": [6, 7, 8],
    "가을 (9-11월)": [9, 10, 11],
    "겨울 (12-2월)": [12, 1, 2],
}

def extract_date_from_filename(filename):
    """
    파일명에서 날짜 정보 추출
    예: 3_20201016_125106_003240.txt -> 2020-10-16
    """
    # YYYYMMDD 패턴 찾기
    date_pattern = r'(\d{8})'
    match = re.search(date_pattern, filename)
    
    if match:
        date_str = match.group(1)
        try:
            date = datetime.strptime(date_str, '%Y%m%d')
            return date
        except ValueError:
            return None
    return None

def get_season(month):
    """월로부터 계절 반환"""
    for season, months in SEASONS.items():
        if month in months:
            return season
    return "알 수 없음"

def analyze_split(split_name):
    """특정 split의 시간적 분포 분석"""
    voc_dir = os.path.join(DATASET_ROOT, split_name, 'voc_annotations')
    
    if not os.path.exists(voc_dir):
        return None
    
    # 통계 수집
    dates = []
    year_month_counter = Counter()
    year_counter = Counter()
    month_counter = Counter()
    season_counter = Counter()
    
    # 모든 annotation 파일 읽기
    xml_files = [f for f in os.listdir(voc_dir) if f.endswith('.xml')]
    
    for filename in xml_files:
        date = extract_date_from_filename(filename)
        if date:
            dates.append(date)
            year = date.year
            month = date.month
            year_month = f"{year}-{month:02d}"
            season = get_season(month)
            
            year_month_counter[year_month] += 1
            year_counter[year] += 1
            month_counter[month] += 1
            season_counter[season] += 1
    
    if not dates:
        return None
    
    # 날짜 범위
    min_date = min(dates)
    max_date = max(dates)
    
    return {
        'total_files': len(xml_files),
        'files_with_date': len(dates),
        'date_range': (min_date, max_date),
        'year_month': dict(year_month_counter),
        'year': dict(year_counter),
        'month': dict(month_counter),
        'season': dict(season_counter),
    }

def print_analysis(split_name, stats):
    """분석 결과 출력"""
    if stats is None:
        print(f"⚠️  {split_name.upper()} set의 voc_annotations 폴더를 찾을 수 없습니다.")
        return
    
    total = stats['total_files']
    with_date = stats['files_with_date']
    min_date, max_date = stats['date_range']
    
    print("=" * 70)
    print(f"📊 {split_name.upper()} Set - 시간적 분포 분석")
    print("=" * 70)
    print(f"총 파일 수: {total:,}개")
    print(f"날짜 추출 성공: {with_date:,}개 ({with_date/total*100:.1f}%)")
    print(f"기간: {min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}")
    print()
    
    # 연도별
    print("📅 연도별 분포:")
    print("-" * 70)
    for year in sorted(stats['year'].keys()):
        count = stats['year'][year]
        percentage = count / with_date * 100
        bar = "█" * int(percentage / 2)
        print(f"  {year}년: {count:>6,}개 ({percentage:>5.1f}%) {bar}")
    print()
    
    # 월별
    print("📅 월별 분포:")
    print("-" * 70)
    month_names = {
        1: "1월", 2: "2월", 3: "3월", 4: "4월", 5: "5월", 6: "6월",
        7: "7월", 8: "8월", 9: "9월", 10: "10월", 11: "11월", 12: "12월"
    }
    for month in sorted(stats['month'].keys()):
        count = stats['month'][month]
        percentage = count / with_date * 100
        bar = "█" * int(percentage / 2)
        print(f"  {month_names[month]:>4}: {count:>6,}개 ({percentage:>5.1f}%) {bar}")
    print()
    
    # 계절별
    print("🌸 계절별 분포:")
    print("-" * 70)
    season_order = ["봄 (3-5월)", "여름 (6-8월)", "가을 (9-11월)", "겨울 (12-2월)"]
    for season in season_order:
        count = stats['season'].get(season, 0)
        if count > 0:
            percentage = count / with_date * 100
            bar = "█" * int(percentage / 2)
            print(f"  {season}: {count:>6,}개 ({percentage:>5.1f}%) {bar}")
    print()
    
    # 년-월 상세 (많은 순서대로 상위 10개)
    print("📆 년-월 상세 분포 (상위 10개):")
    print("-" * 70)
    sorted_year_month = sorted(stats['year_month'].items(), key=lambda x: x[1], reverse=True)[:10]
    for year_month, count in sorted_year_month:
        percentage = count / with_date * 100
        bar = "█" * int(percentage / 2)
        print(f"  {year_month}: {count:>6,}개 ({percentage:>5.1f}%) {bar}")
    print()

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("🗓️  데이터셋 시간적 분포 분석")
    print("=" * 70)
    print(f"데이터셋: {DATASET_ROOT}")
    print()
    
    # Train, Val, Test 각각 분석
    for split in ['train', 'val', 'test']:
        stats = analyze_split(split)
        print_analysis(split, stats)
    
    # 전체 통합 분석
    print("=" * 70)
    print("📊 전체 데이터셋 통합 분석")
    print("=" * 70)
    
    all_stats = {
        'year_month': Counter(),
        'year': Counter(),
        'month': Counter(),
        'season': Counter(),
        'total_files': 0,
        'files_with_date': 0,
    }
    
    all_dates = []
    
    for split in ['train', 'val', 'test']:
        stats = analyze_split(split)
        if stats:
            all_stats['total_files'] += stats['total_files']
            all_stats['files_with_date'] += stats['files_with_date']
            all_stats['year_month'].update(stats['year_month'])
            all_stats['year'].update(stats['year'])
            all_stats['month'].update(stats['month'])
            all_stats['season'].update(stats['season'])
            all_dates.extend([stats['date_range'][0], stats['date_range'][1]])
    
    if all_dates:
        all_stats['date_range'] = (min(all_dates), max(all_dates))
        all_stats['year_month'] = dict(all_stats['year_month'])
        all_stats['year'] = dict(all_stats['year'])
        all_stats['month'] = dict(all_stats['month'])
        all_stats['season'] = dict(all_stats['season'])
        
        print_analysis('전체', all_stats)
    
    print("=" * 70)
    print("✅ 분석 완료!")
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

