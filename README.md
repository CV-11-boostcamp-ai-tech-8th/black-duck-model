# black-duck-model

## Quick Start
### 1. Prerequisites
```bash
(작성 예정)
```
### 2. Run
- 우선, `black-duck-web`의 backend 서버가 실행되어 있어야 함.
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
tree 구조 - rjy
```
### 주요 폴더 설명


## 만든 사람
표(web 파트 참고) - kmj