#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Set Evaluation Script
- 지정된 가중치로 test set에 대한 validation 수행
- train.py의 test 부분을 독립적인 스크립트로 분리
"""

import os
import argparse
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
import wandb
from dotenv import load_dotenv


def evaluate_test_set(
    weight_path,
    data_config,
    imgsz=640,
    batch=64,
    best_epoch=None,
    save_csv=True,
    output_dir=None,
    use_wandb=False,
    wandb_project=None,
    wandb_entity=None,
    run_name=None,
):
    """
    Test set에 대한 evaluation 수행
    
    Args:
        weight_path: 가중치 파일 경로 (.pt)
        data_config: 데이터셋 YAML 설정 파일 경로
        imgsz: 이미지 크기
        batch: 배치 크기
        best_epoch: Best epoch 번호 (CSV/Wandb 기록용)
        save_csv: CSV 파일로 저장 여부
        output_dir: 출력 디렉토리 (None이면 가중치와 같은 위치)
        use_wandb: Wandb 로깅 여부
        wandb_project: Wandb 프로젝트명
        wandb_entity: Wandb entity명
        run_name: Wandb run 이름
    
    Returns:
        test_metrics_dict: Test 결과 딕셔너리
    """
    # 가중치 파일 존재 확인
    weight_path = Path(weight_path)
    if not weight_path.exists():
        raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {weight_path}")
    
    # 데이터 설정 파일 존재 확인
    data_config = Path(data_config)
    if not data_config.exists():
        raise FileNotFoundError(f"데이터 설정 파일을 찾을 수 없습니다: {data_config}")
    
    print("=" * 70)
    print("🧪 Test Set Evaluation")
    print("=" * 70)
    print(f"가중치: {weight_path}")
    print(f"데이터셋: {data_config}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    if best_epoch is not None:
        print(f"Best epoch: {best_epoch}")
    print()
    
    # 모델 로드
    print("📦 모델 로드 중...")
    model = YOLO(str(weight_path))
    print(f"✓ 모델 로드 완료: {model.model_name}")
    print()
    
    # Test set evaluation
    print("🧪 Test set evaluation 수행 중...")
    test_metrics = model.val(
        data=str(data_config),
        split="test",
        imgsz=imgsz,
        batch=batch,
    )
    print("✓ Evaluation 완료")
    print()
    
    # 결과 추출
    test_metrics_dict = test_metrics.results_dict
    
    # Fitness 직접 계산: 0.1 × mAP50 + 0.9 × mAP50-95
    test_mAP50 = test_metrics_dict["metrics/mAP50(B)"]
    test_mAP50_95 = test_metrics_dict["metrics/mAP50-95(B)"]
    test_fitness = 0.1 * test_mAP50 + 0.9 * test_mAP50_95
    test_precision = test_metrics_dict["metrics/precision(B)"]
    test_recall = test_metrics_dict["metrics/recall(B)"]
    
    # 결과 출력
    print("=" * 70)
    print("📊 Test Set Results")
    print("=" * 70)
    print(f"Precision:     {test_precision:.5f}")
    print(f"Recall:        {test_recall:.5f}")
    print(f"mAP50:         {test_mAP50:.5f}")
    print(f"mAP50-95:      {test_mAP50_95:.5f}")
    print(f"Fitness:       {test_fitness:.5f}  (= 0.1×mAP50 + 0.9×mAP50-95)")
    print("=" * 70)
    print()
    
    # CSV 저장
    if save_csv:
        # 출력 디렉토리 결정
        if output_dir is None:
            output_dir = weight_path.parent.parent  # weights/ -> run_dir/
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "results_test_set.csv"
        
        # DataFrame 생성
        results_df = pd.DataFrame([{
            'best_epoch': best_epoch if best_epoch is not None else -1,
            'test/precision': test_precision,
            'test/recall': test_recall,
            'test/mAP50': test_mAP50,
            'test/mAP50-95': test_mAP50_95,
            'test/fitness': test_fitness,
        }])
        
        results_df.to_csv(csv_path, index=False)
        print(f"✓ CSV 저장 완료: {csv_path}")
        print()
    
    # Wandb 로깅
    if use_wandb:
        if wandb_project is None:
            print("⚠️  Wandb 프로젝트명이 지정되지 않았습니다. 로깅 생략.")
        else:
            print("📤 Wandb 로깅 중...")
            
            # Wandb 초기화
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=run_name,
                resume="allow",
            )
            
            # 로그 기록
            log_step = best_epoch if best_epoch is not None else 0
            wandb.log(
                {
                    "test/precision": test_precision,
                    "test/recall": test_recall,
                    "test/mAP50": test_mAP50,
                    "test/mAP50-95": test_mAP50_95,
                    "test/fitness": test_fitness,
                },
                step=log_step,
            )
            
            # Summary에도 기록
            wandb.summary["test/mAP50"] = test_mAP50
            wandb.summary["test/mAP50-95"] = test_mAP50_95
            wandb.summary["test/precision"] = test_precision
            wandb.summary["test/recall"] = test_recall
            wandb.summary["test/fitness"] = test_fitness
            
            wandb.finish()
            print("✓ Wandb 로깅 완료")
            print()
    
    return {
        'precision': test_precision,
        'recall': test_recall,
        'mAP50': test_mAP50,
        'mAP50-95': test_mAP50_95,
        'fitness': test_fitness,
    }


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description='Test set evaluation with specified weights',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 사용법
  python scripts/evaluate_test_set.py \\
      --weight runs/detect/cv-11-final/yolo26s_idx10_e40_b64/weights/best.pt \\
      --data configs/yolo/vehicle_dataset.yaml
  
  # Best epoch 정보와 함께
  python evaluate_test_set.py \\
      --weight runs/detect/cv-11-final/yolo26s_idx10_e40_b64/weights/best.pt \\
      --data configs/yolo/vehicle_dataset.yaml \\
      --best-epoch 35
  
  # Wandb 로깅 포함
  python evaluate_test_set.py \\
      --weight runs/detect/cv-11-final/yolo26s_idx10_e40_b64/weights/best.pt \\
      --data configs/yolo/vehicle_dataset.yaml \\
      --wandb \\
      --wandb-project cv-11-final \\
      --wandb-entity cv_11 \\
      --run-name yolo26s_idx10_e40_b64
        """
    )
    
    # 필수 인자
    parser.add_argument('--weight', type=str, required=True,
                        help='가중치 파일 경로 (.pt)')
    parser.add_argument('--data', type=str, required=True,
                        help='데이터셋 YAML 설정 파일 경로')
    
    # 선택 인자
    parser.add_argument('--imgsz', type=int, default=640,
                        help='이미지 크기 (기본값: 640)')
    parser.add_argument('--batch', type=int, default=64,
                        help='배치 크기 (기본값: 64)')
    parser.add_argument('--best-epoch', type=int, default=None,
                        help='Best epoch 번호 (기록용)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='출력 디렉토리 (기본값: 가중치와 같은 위치)')
    parser.add_argument('--no-csv', action='store_true',
                        help='CSV 저장 안 함')
    
    # Wandb 관련
    parser.add_argument('--wandb', action='store_true',
                        help='Wandb 로깅 활성화')
    parser.add_argument('--wandb-project', type=str, default=None,
                        help='Wandb 프로젝트명')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='Wandb entity명')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Wandb run 이름')
    
    args = parser.parse_args()
    
    # Wandb 사용 시 .env 로드
    if args.wandb:
        load_dotenv()
        WANDB_API_KEY = os.getenv('WANDB_API_KEY')
        if WANDB_API_KEY:
            os.environ["WANDB_API_KEY"] = WANDB_API_KEY
            if args.wandb_entity:
                os.environ["WANDB_ENTITY"] = args.wandb_entity
            wandb.login()
        else:
            print("⚠️  Warning: WANDB_API_KEY가 .env에 없습니다.")
    
    # Evaluation 수행
    try:
        results = evaluate_test_set(
            weight_path=args.weight,
            data_config=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            best_epoch=args.best_epoch,
            save_csv=not args.no_csv,
            output_dir=args.output_dir,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            run_name=args.run_name,
        )
        
        print("=" * 70)
        print("✅ Test set evaluation 완료!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

