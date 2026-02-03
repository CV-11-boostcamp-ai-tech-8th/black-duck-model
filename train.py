#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv26s Vehicle Detection - Training
- pretrained YOLOv26s를 vehicle 데이터셋으로 파인튜닝
- 모든 설정은 YAML 파일을 통해 전달됨
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO
import wandb
from dotenv import load_dotenv
import pandas as pd
from omegaconf import OmegaConf


def print_best_epoch_info(results_csv_path):
    """
    results.csv에서 best epoch 정보 출력
    
    Args:
        results_csv_path: results.csv 파일 경로
    
    Returns:
        best_epoch: best epoch 번호 (실패 시 None)
    """
    results_path = Path(results_csv_path)
    if not results_path.exists():
        print(f"[Warning] results.csv를 찾을 수 없습니다: {results_csv_path}")
        return None
    
    try:
        # Load the training log
        results = pd.read_csv(results_csv_path)
        
        # Strip spaces from column names
        results.columns = results.columns.str.strip()
        
        # Calculate fitness: 0.1 × mAP50 + 0.9 × mAP50-95
        results["fitness"] = results["metrics/mAP50(B)"] * 0.1 + results["metrics/mAP50-95(B)"] * 0.9
        
        # Find the epoch with the highest fitness
        best_idx = results['fitness'].idxmax()
        best_epoch = int(results.loc[best_idx, 'epoch'])
        best_fitness = results.loc[best_idx, 'fitness']
        best_mAP50 = results.loc[best_idx, 'metrics/mAP50(B)']
        best_mAP50_95 = results.loc[best_idx, 'metrics/mAP50-95(B)']
        best_precision = results.loc[best_idx, 'metrics/precision(B)']
        best_recall = results.loc[best_idx, 'metrics/recall(B)']
        
        # 결과 출력
        print()
        print("=" * 70)
        print(f"🏆 Best Model Info (Epoch {best_epoch})")
        print("=" * 70)
        print(f"Fitness:       {best_fitness:.6f}  (= 0.1×mAP50 + 0.9×mAP50-95)")
        print(f"mAP50-95:      {best_mAP50_95:.5f}")
        print(f"mAP50:         {best_mAP50:.5f}")
        print(f"Precision:     {best_precision:.5f}")
        print(f"Recall:        {best_recall:.5f}")
        print()
        
        # Top 5 epochs 출력
        print("📊 Top 5 Epochs (by fitness):")
        print("-" * 70)
        top5 = results.nlargest(5, 'fitness')[['epoch', 'fitness', 'metrics/mAP50-95(B)', 'metrics/mAP50(B)', 'metrics/recall(B)']]
        for idx, row in top5.iterrows():
            marker = "✓" if int(row['epoch']) == best_epoch else " "
            print(f"{marker} Epoch {int(row['epoch']):2d}  |  Fitness: {row['fitness']:.6f}  |  mAP50-95: {row['metrics/mAP50-95(B)']:.5f}  |  mAP50: {row['metrics/mAP50(B)']:.5f}  |  Recall: {row['metrics/recall(B)']:.5f}")
        print("=" * 70)
        
        return best_epoch
        
    except Exception as e:
        print(f"[Warning] Best epoch 정보 출력 중 오류 발생: {e}")
        return None


def main(cfg):
    """메인 실행 함수"""
    # Train name 생성
    train_name = f"yolo26s_{cfg.version}_e{cfg.epochs}_b{cfg.batch_size}"
    
    print("=" * 70)
    print("YOLOv26s Vehicle Detection - Training")
    print("=" * 70)
    print(f"버전: {cfg.version}")
    print()
    
    # ========== Step 1: 모델 로드 ==========
    print("[Step 1] Pretrained YOLOv26s 모델 로드")
    print("-" * 70)
    
    model = YOLO(cfg.model_weight)
    print(f"✓ 모델 로드 완료: {model.model_name}")
    print()
    
    # ========== Step 2: Training ==========
    print(f"[Step 2] Vehicle 데이터셋으로 Fine-tuning ({cfg.epochs} epoch)")
    print("-" * 70)
    print(f"데이터셋 설정: {cfg.dataset_config}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Image size: {cfg.image_size}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"※ 데이터 경로는 {cfg.dataset_config}에 정의되어 있습니다.")
    print("※ Validation은 학습 중 자동으로 수행됩니다.")
    print()
    
    # vehicle dataset으로 파인튜닝
    train_results = model.train(
        data=cfg.dataset_config,
        epochs=cfg.epochs,
        imgsz=cfg.image_size,
        batch=cfg.batch_size,
        project=cfg.train_project,
        name=train_name,
        exist_ok=True,
        pretrained=True,  # pretrained weight 유지
        verbose=True,
        amp=cfg.use_amp,
        seed=cfg.seed,
    )
    
    print()
    print("=" * 70)
    print("학습 완료!")
    print("=" * 70)
    print()
    print("📁 저장된 파일:")
    print(f"  - 모델 가중치: runs/detect/{cfg.train_project}/{train_name}/weights/best.pt")
    print(f"  - 학습 로그: runs/detect/{cfg.train_project}/{train_name}/")
    print(f"  - Validation 결과: runs/detect/{cfg.train_project}/{train_name}/val_*.jpg")
    
    # Best epoch 정보 출력
    results_csv_path = f"runs/detect/{cfg.train_project}/{train_name}/results.csv"
    best_epoch = print_best_epoch_info(results_csv_path)


    # ========== Step 3: Test Set Evaluation (best.pt) ==========
    print("[Step 3] Test set evaluation with best.pt")
    print("-" * 70)

    best_model_path = f"runs/detect/{cfg.train_project}/{train_name}/weights/best.pt"
    best_model = YOLO(best_model_path)

    test_metrics = best_model.val(
        data=cfg.dataset_config,
        split="test",
        imgsz=cfg.image_size,
        batch=cfg.batch_size,
    )

    print("✓ Test evaluation 완료")

    test_metrics_dict = test_metrics.results_dict

    print("[Test metrics]")
    for k, v in test_metrics_dict.items():
        print(f"{k}: {v:.5f}")

    # Test 결과를 CSV 파일로 저장
    test_results_csv_path = f"runs/detect/{cfg.train_project}/{train_name}/results_test_set.csv"
    test_results_df = pd.DataFrame([{
        'best_epoch': best_epoch,
        'test/precision': test_metrics_dict["metrics/precision(B)"],
        'test/recall': test_metrics_dict["metrics/recall(B)"],
        'test/mAP50': test_metrics_dict["metrics/mAP50(B)"],
        'test/mAP50-95': test_metrics_dict["metrics/mAP50-95(B)"],
        'test/fitness': test_metrics_dict["fitness"],
    }])
    test_results_df.to_csv(test_results_csv_path, index=False)

    ### 이거 wandb assume 제대로 안 되면 그냥 wandb logging은 포기하고 주석처리하기 ###
    # Wandb에 Test 결과 로깅
    wandb.init(
        project=cfg.train_project,
        entity=cfg.wandb_entity,
        name=train_name,
        resume="allow",
    )

    # wandb에 명시적으로 기록
    wandb.log(
        {
            "test/precision": test_metrics_dict["metrics/precision(B)"],
            "test/recall": test_metrics_dict["metrics/recall(B)"],
            "test/mAP50": test_metrics_dict["metrics/mAP50(B)"],
            "test/mAP50-95": test_metrics_dict["metrics/mAP50-95(B)"],
            "test/fitness": test_metrics_dict["fitness"],
        },
        step=best_epoch if best_epoch is not None else cfg.epochs,
    )

    # summary에도 남기기 (run 페이지 상단에 고정)
    wandb.summary["test/mAP50"] = test_metrics_dict["metrics/mAP50(B)"]
    wandb.summary["test/mAP50-95"] = test_metrics_dict["metrics/mAP50-95(B)"]
    wandb.summary["test/precision"] = test_metrics_dict["metrics/precision(B)"]
    wandb.summary["test/recall"] = test_metrics_dict["metrics/recall(B)"]
    
    wandb.finish()
    print("✓ Wandb 로깅 완료")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO Training with YAML config')
    parser.add_argument('config', type=str, help='YAML 설정 파일 경로')
    
    args = parser.parse_args()
    
    # YAML 설정 로드
    cfg = OmegaConf.load(args.config)
    
    # .env에서 Wandb API key 로드
    load_dotenv()
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    os.environ["WANDB_ENTITY"] = cfg.get('wandb_entity', 'cv_11')
    wandb.login()
    
    main(cfg)

