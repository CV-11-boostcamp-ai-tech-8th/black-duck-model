# black-duck-model

## Quick Start
### 1. Prerequisites
- versions
```bash
(작성 예정)
```
- 가상환경 설정부터 시작할 경우:
```bash
source requirements.sh
```
### 2. Run
- 우선, `black-duck-web`의 backend 서버가 실행되어 있어야 함.
- 기본적으로는 백엔드 서버가 해당 코드를 실행하므로, 사용자가 명시적으로 해당 코드를 실행할 일은 없음.
- 그러나 job.json이 로컬에 있다는 가정 하에, 직접 테스트할 경우:
```bash
cd model_connect
python connect_final.py \
  --job <PATH_TO_JOB_JSON> \
  [--bucket GCS_BUCKET_NAME] \
  [--expire DEFAULT_SIGNED_URL_EXPIRE_SECONDS]
```
- `--job <path>` (required) : job json 경로
- `--bucket <name>` (optional) : GCS bucket 이름
- `--expire <seconds>` (optional, default=3600) : signed URL 만료 시간

## Model Architecture


## Project Structure
```
tree 구조 - rjy -> 개별 workspace는 workspace 안에 하기.
```
### 주요 폴더 설명


## 만든 사람
표(web 파트 참고) - kmj