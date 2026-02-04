#!/bin/bash
# Batch Track Runner
# - YAML 설정 파일 배열을 받아 순차적으로 tracking 실행
# - 각 실행 로그를 logs/ 폴더에 저장

set -e  # 에러 발생 시 중단

# ========== 설정 ==========
# YAML 설정 파일 경로 배열 (수정 필요)
CONFIGS=(
    "configs/track/idx1.yaml"
    "configs/track/idx2.yaml"
    "configs/track/idx3.yaml"
    "configs/track/idx4.yaml"
)

# 로그 디렉토리
LOG_DIR="logs/track"

# Python 실행 파일
PYTHON_CMD="python"
TRACK_SCRIPT="track.py"
# ==========================

# 로그 디렉토리 생성
mkdir -p "$LOG_DIR"

# 배치 실행 시작 시간
BATCH_START_TIME=$(date +%s)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "======================================================================"
echo "Batch Track Runner"
echo "======================================================================"
echo "총 설정 파일 수: ${#CONFIGS[@]}"
echo "로그 디렉토리: $LOG_DIR"
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 결과 저장
SUCCESS_COUNT=0
FAILED_COUNT=0
declare -a RESULTS

# 각 설정 파일에 대해 tracking 실행
for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    INDEX=$((i + 1))
    
    echo "----------------------------------------------------------------------"
    echo "[$INDEX/${#CONFIGS[@]}] Tracking 시작"
    echo "----------------------------------------------------------------------"
    echo "설정 파일: $CONFIG"
    
    # YAML에서 version 추출 (yq 없이 grep/sed 사용)
    VERSION=$(grep "^version:" "$CONFIG" | sed 's/version: *"\?\([^"]*\)"\?/\1/' | tr -d ' ')
    VIDEO_NAME=$(grep "^video_path:" "$CONFIG" | sed 's/.*\///' | sed 's/\.[^.]*$//')
    
    echo "버전: $VERSION"
    echo "비디오: $VIDEO_NAME"
    
    # 로그 파일명 생성
    LOG_FILE="$LOG_DIR/track_${VIDEO_NAME}_${VERSION}_${TIMESTAMP}.log"
    echo "로그 파일: $LOG_FILE"
    echo ""
    
    # Tracking 실행 및 로그 저장 (tee로 실시간 출력 + 파일 저장)
    if $PYTHON_CMD "$TRACK_SCRIPT" "$CONFIG" 2>&1 | tee "$LOG_FILE"; then
        echo "✓ 성공"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        RESULTS+=("✓ $VIDEO_NAME (version: $VERSION)")
    else
        echo "✗ 실패"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        RESULTS+=("✗ $VIDEO_NAME (version: $VERSION)")
    fi
    
    echo ""
done

# 배치 실행 종료
BATCH_END_TIME=$(date +%s)
BATCH_DURATION=$((BATCH_END_TIME - BATCH_START_TIME))

# 최종 결과 출력
echo "======================================================================"
echo "Batch Track 완료"
echo "======================================================================"
echo "총 처리 시간: ${BATCH_DURATION}초 ($((BATCH_DURATION / 60))분 $((BATCH_DURATION % 60))초)"
echo ""
echo "결과:"
for result in "${RESULTS[@]}"; do
    echo "  $result"
done
echo ""
echo "성공: $SUCCESS_COUNT/${#CONFIGS[@]}"
echo "실패: $FAILED_COUNT/${#CONFIGS[@]}"
echo ""
echo "📁 로그 저장 위치: $(pwd)/$LOG_DIR"
echo "======================================================================"

