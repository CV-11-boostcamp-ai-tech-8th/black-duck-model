#!/bin/bash
# dataset 디렉토리의 모든 비디오 파일을 순회하여 테스트 실행

set -e  # 에러 발생 시 중단

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="/data/ephemeral/home/dataset"
TEST_SCRIPT="${SCRIPT_DIR}/connect_final_sudden_stop_and_ocr_test_local.py"
JOB_JSON="${SCRIPT_DIR}/ssh_input/task1_1770046870_fd24a5fc/job.json"

# 색상 출력
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================"
echo "모든 비디오 파일 테스트 시작"
echo "============================================================"
echo "[dataset_dir] ${DATASET_DIR}"
echo "[test_script] ${TEST_SCRIPT}"
echo "[job_json] ${JOB_JSON}"
echo ""

# 비디오 파일 찾기 (mp4만)
video_files=($(find "${DATASET_DIR}" -maxdepth 1 -type f -name "*.mp4" | sort))

if [ ${#video_files[@]} -eq 0 ]; then
    echo -e "${RED}[error] 비디오 파일을 찾을 수 없습니다: ${DATASET_DIR}/*.mp4${NC}"
    exit 1
fi

echo -e "${GREEN}[info] 총 ${#video_files[@]}개의 비디오 파일을 찾았습니다.${NC}"
echo ""

# 각 비디오 파일에 대해 테스트 실행
success_count=0
fail_count=0
total_count=${#video_files[@]}

for i in "${!video_files[@]}"; do
    video_file="${video_files[$i]}"
    video_name=$(basename "${video_file}")
    current_num=$((i + 1))
    
    echo "============================================================"
    echo -e "${YELLOW}[${current_num}/${total_count}] ${video_name}${NC}"
    echo "============================================================"
    echo "[video] ${video_file}"
    echo ""
    
    # 테스트 실행
    if python3 "${TEST_SCRIPT}" --job "${JOB_JSON}" --video "${video_file}"; then
        echo ""
        echo -e "${GREEN}[ok] ${video_name} 테스트 성공${NC}"
        success_count=$((success_count + 1))
    else
        echo ""
        echo -e "${RED}[error] ${video_name} 테스트 실패${NC}"
        fail_count=$((fail_count + 1))
    fi
    
    echo ""
    echo "============================================================"
    echo ""
done

# 최종 결과 출력
echo "============================================================"
echo "테스트 완료 요약"
echo "============================================================"
echo -e "${GREEN}[성공] ${success_count}개${NC}"
echo -e "${RED}[실패] ${fail_count}개${NC}"
echo -e "[전체] ${total_count}개"
echo "============================================================"

# 실패가 있으면 종료 코드 1 반환
if [ ${fail_count} -gt 0 ]; then
    exit 1
fi

exit 0

