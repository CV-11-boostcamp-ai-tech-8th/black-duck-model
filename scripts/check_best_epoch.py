#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check Best Epoch from YOLO Training Results
- results.csv를 읽어서 fitness를 계산하고 best epoch를 찾습니다.
"""

import pandas as pd
import sys
from pathlib import Path

DEFAULT_PATH = "/data/ephemeral/home/jsw/black-duck-model/runs/detect/cv-11-final/yolo26s_idx10_e40_b64/results.csv"


def check_best_epoch(results_csv_path):
    """
    results.csv에서 best epoch 찾기
    
    Args:
        results_csv_path: results.csv 파일 경로
    
    Returns:
        best_epoch: 최고 성능 epoch 번호
    """
    # Load the training log
    results = pd.read_csv(results_csv_path)
    
    # Strip spaces from column names
    results.columns = results.columns.str.strip()
    
    # Calculate fitness: 0.1 × mAP50 + 0.9 × mAP50-95
    results["fitness"] = results["metrics/mAP50(B)"] * 0.1 + results["metrics/mAP50-95(B)"] * 0.9
    
    # Find the epoch with the highest fitness
    best_idx = results['fitness'].idxmax()
    best_epoch = results.loc[best_idx, 'epoch']
    
    # Get best metrics
    best_fitness = results.loc[best_idx, 'fitness']
    best_mAP50 = results.loc[best_idx, 'metrics/mAP50(B)']
    best_mAP50_95 = results.loc[best_idx, 'metrics/mAP50-95(B)']
    best_precision = results.loc[best_idx, 'metrics/precision(B)']
    best_recall = results.loc[best_idx, 'metrics/recall(B)']
    
    return {
        'epoch': int(best_epoch),
        'fitness': best_fitness,
        'mAP50': best_mAP50,
        'mAP50-95': best_mAP50_95,
        'precision': best_precision,
        'recall': best_recall,
        'results_df': results
    }


def main():
    """메인 실행 함수"""
    # 기본 경로 설정
    default_path = DEFAULT_PATH
    
    # 커맨드 라인 인자로 경로 받기
    if len(sys.argv) > 1:
        results_csv_path = sys.argv[1]
    else:
        results_csv_path = default_path
    
    # 경로 확인
    results_path = Path(results_csv_path)
    if not results_path.exists():
        print(f"[Error] 파일을 찾을 수 없습니다: {results_csv_path}")
        print(f"\n사용법: python check_best_epoch.py [results.csv 경로]")
        print(f"예시: python check_best_epoch.py {default_path}")
        return
    
    print("=" * 70)
    print("YOLO Best Epoch Checker")
    print("=" * 70)
    print(f"📁 Results CSV: {results_csv_path}")
    print()
    
    # Best epoch 찾기
    result = check_best_epoch(results_csv_path)
    
    # 결과 출력
    print("=" * 70)
    print(f"🏆 Best Model: Epoch {result['epoch']}")
    print("=" * 70)
    print(f"Fitness:       {result['fitness']:.6f}")
    print(f"mAP50-95:      {result['mAP50-95']:.5f}")
    print(f"mAP50:         {result['mAP50']:.5f}")
    print(f"Precision:     {result['precision']:.5f}")
    print(f"Recall:        {result['recall']:.5f}")
    print()
    
    # Top 5 epochs 출력
    print("📊 Top 5 Epochs (by fitness):")
    print("-" * 70)
    top5 = result['results_df'].nlargest(5, 'fitness')[['epoch', 'fitness', 'metrics/mAP50-95(B)', 'metrics/mAP50(B)', 'metrics/recall(B)']]
    print(top5.to_string(index=False))
    print("=" * 70)


if __name__ == '__main__':
    main()

