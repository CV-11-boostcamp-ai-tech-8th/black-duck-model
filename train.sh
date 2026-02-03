#!/bin/bash

# ============================================================================
# Training 배치 실행 스크립트
# 사용 방법: 아래 CONFIGS 배열을 수정하여 원하는 설정 파일들을 지정
# ============================================================================

# ============================================================================
# 변수 설정 (여기서 수정하세요)
# ============================================================================

# 프로젝트 루트 디렉토리 (스크립트가 실행되는 위치)
PROJECT_ROOT=$(pwd)

# 실행할 config 파일 목록 (순서대로 실행)
CONFIGS=(
    "configs/train/idx10.yaml"
    # "configs/train/idx2.yaml"
    # "configs/train/idx3.yaml"
)

# 로그 디렉토리
LOG_DIR="logs/train"

# Python 실행 명령어
PYTHON_CMD="python"

# Training 스크립트 경로 (현재 디렉토리에 있다고 가정)
TRAIN_SCRIPT="train.py"

# ============================================================================
# 로그 디렉토리 생성
# ============================================================================
mkdir -p "${LOG_DIR}"

# ============================================================================
# 각 config로 Training 실행
# ============================================================================
SUCCESS_COUNT=0
FAIL_COUNT=0
TOTAL_START_TIME=$(date +%s)

echo "=========================================="
echo "🚀 Training 배치 실행 시작"
echo "=========================================="

for CONFIG in "${CONFIGS[@]}"; do
    # 빈 요소나 주석 처리된 항목 건너뛰기
    if [ -z "$CONFIG" ] || [[ "$CONFIG" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    # config 파일 존재 확인
    if [ ! -f "$CONFIG" ]; then
        echo "⚠️  Config 파일을 찾을 수 없습니다: $CONFIG"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi
    
    # config 파일명에서 확장자 제거하여 로그 파일명 생성
    config_base_name=$(basename "$CONFIG" .yaml)
    
    # 로그 파일명: train_{config_name}_{timestamp}.log
    log_file="${LOG_DIR}/train_${config_base_name}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "------------------------------------------"
    echo "🚀 Training 시작: $CONFIG"
    echo "📝 로그 파일: $log_file"
    echo "------------------------------------------"
    
    # Training 실행 (로그 파일에 출력 저장)
    $PYTHON_CMD "$TRAIN_SCRIPT" "$CONFIG" 2>&1 | tee "$log_file"
    
    # 종료 코드 확인
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ Training 완료: $CONFIG"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "❌ Training 실패: $CONFIG"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "⚠️  다음 config로 계속 진행합니다..."
    fi
    
    echo ""
done

TOTAL_END_TIME=$(date +%s)
TOTAL_ELAPSED_TIME=$((TOTAL_END_TIME - TOTAL_START_TIME))

echo "=========================================="
echo "🎉 모든 Training 작업이 완료되었습니다!"
echo "=========================================="
echo "📊 요약:"
echo "  - 성공: ${SUCCESS_COUNT}개"
echo "  - 실패: ${FAIL_COUNT}개"
echo "  - 총 소요 시간: ${TOTAL_ELAPSED_TIME}초 ($((TOTAL_ELAPSED_TIME/60))분)"
echo "=========================================="

