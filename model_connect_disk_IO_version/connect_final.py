#!/usr/bin/env python3
import argparse
import json
import mimetypes
import os
import subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta

import requests

try:
    from google.cloud import storage
    HAVE_GCS = True
except ImportError:
    HAVE_GCS = False


# -----------------------------
# 설정 (필요하면 여기만 수정)
# -----------------------------
DEFAULT_SIGNED_URL_EXPIRE_SECONDS = 3600  # 1시간
UPLOAD_VIDEOS = True  # mp4도 GCS에 올릴지 (원치 않으면 False)


# -----------------------------
# 유틸
# -----------------------------

# [Change]
def _run(cmd: list[str]) -> None:
    """subprocess 실행 (에러면 예외 발생)"""
    env = os.environ.copy()
    env["PATH"] = "/opt/conda/bin:" + env.get("PATH", "")
    subprocess.run(cmd, check=True, env=env)

def extract_first_frame(video_path: Path, out_jpg: Path) -> None:
    _run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(out_jpg),
    ])


def find_service_account_json() -> str | None:
    """
    GOOGLE_APPLICATION_CREDENTIALS가 없으면,
    스크립트 기준 ./secret/*.json 중 첫번째를 자동으로 찾아서 사용.
    """
    env_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if env_path and Path(env_path).exists():
        return env_path

    # connect.py 기준 경로에서 secret 폴더 탐색
    here = Path(__file__).resolve().parent
    secret_dir = here / "secret"
    if secret_dir.exists():
        candidates = sorted(secret_dir.glob("*.json"))
        if candidates:
            return str(candidates[0])

    return None


def make_storage_client():
    """
    가능한 순서:
    1) GOOGLE_APPLICATION_CREDENTIALS 또는 ./secret/*.json (서비스계정)
    2) ADC (예: GCE/GKE, gcloud auth application-default)
    """
    cred_path = find_service_account_json()
    if cred_path:
        print(f"[gcs] using service account json: {cred_path}")
        return storage.Client.from_service_account_json(cred_path)
    print("[gcs] using default credentials (ADC)")
    return storage.Client()


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def guess_content_type(path: Path) -> str:
    # .img 같은 커스텀 확장자는 이미지로 가정
    if path.suffix.lower() == ".img":
        return "image/jpeg"
    ctype, _ = mimetypes.guess_type(str(path))
    return ctype or "application/octet-stream"


def infer_bucket_name(job: dict, cli_bucket: str | None) -> str | None:
    """
    우선순위:
    1) CLI --bucket
    2) ENV GCS_BUCKET_NAME
    3) job['input_url'] 에서 추론 (gs://bucket/... 또는 https://storage.googleapis.com/bucket/...)
    """
    if cli_bucket:
        return cli_bucket

    env_bucket = os.getenv("GCS_BUCKET_NAME")
    if env_bucket:
        return env_bucket

    url = job.get("input_url") or ""
    if url.startswith("gs://"):
        # gs://bucket/object
        try:
            bucket, _obj = url[5:].split("/", 1)
            return bucket
        except Exception:
            return None

    if url.startswith("https://storage.googleapis.com/"):
        # https://storage.googleapis.com/<bucket>/<object>
        rest = url[len("https://storage.googleapis.com/") :]
        parts = rest.split("/", 1)
        if parts and parts[0]:
            return parts[0]

    return None


def upload_to_gcs(local_path: Path, bucket_name: str, object_name: str) -> "storage.Blob":
    if not HAVE_GCS:
        raise RuntimeError("google-cloud-storage 미설치인데 GCS 업로드가 필요합니다.")
    client = make_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(
        str(local_path),
        content_type=guess_content_type(local_path),
    )
    return blob


def signed_url_for_blob(blob: "storage.Blob", expire_seconds: int) -> str:
    return blob.generate_signed_url(
        version="v4",
        expiration=timedelta(seconds=expire_seconds),
        method="GET",
    )


def ensure_uploaded_and_get_url(
    local_path: Path,
    bucket_name: str | None,
    object_name: str,
    expire_seconds: int,
) -> str:
    """
    bucket_name이 있으면 GCS 업로드 후 signed url 반환,
    없으면 로컬 경로 문자열 반환.
    """
    if not local_path or not local_path.exists():
        return None

    if not bucket_name:
        return str(local_path)

    blob = upload_to_gcs(local_path, bucket_name, object_name)
    return signed_url_for_blob(blob, expire_seconds)


def download_input(job: dict) -> Path | None:
    """
    기존 동작 유지: job['input_url'] 있으면 job['input_path']로 다운로드.
    """
    input_path = Path(job["input_path"])
    input_path.parent.mkdir(parents=True, exist_ok=True)

    url = job.get("input_url")
    if url:
        if url.startswith("gs://"):
            if not HAVE_GCS:
                raise RuntimeError("google-cloud-storage 미설치")
            bucket_name, object_path = url[5:].split("/", 1)
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            bucket.blob(object_path).download_to_filename(input_path)
        else:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with input_path.open("wb") as f:
                    for chunk in r.iter_content(8192):
                        if chunk:
                            f.write(chunk)
        return input_path

    if input_path.exists():
        return input_path

    # 입력이 필요 없는 경우도 있을 수 있으니 None 반환
    return None


def pick_first_file(base_dir: Path, patterns: list[str]) -> Path | None:
    for pat in patterns:
        found = sorted(base_dir.glob(pat))
        if found:
            return found[0]
    return None


def scan_clips(task_out_dir: Path) -> list[dict]:
    """
    task_out_dir 안의 clip_* 폴더를 스캔해서 mp4/thumbnail/plate 이미지 경로를 모음.
    """
    results = []
    clip_dirs = sorted([p for p in task_out_dir.iterdir() if p.is_dir() and p.name.startswith("clip_")])

    for clip_dir in clip_dirs:
        # mp4 찾기
        mp4_path = pick_first_file(clip_dir, ["*.mp4"])

        # thumbnail(.img 우선, 없으면 jpg/png)
        thumb_path = pick_first_file(clip_dir, ["*.img", "*.jpg", "*.jpeg", "*.png", "*.webp"])

        # license plate 이미지: license_plate_image.{jpg,jpeg,png} 형식으로 찾기
        plate_path = pick_first_file(clip_dir, [
            "license_plate_image.jpg",
            "license_plate_image.jpeg",
            "license_plate_image.png",
            "license_plate_image.webp",
        ])

        # license plate 텍스트: license_plate_text.txt 파일 읽기
        plate_text_path = clip_dir / "license_plate_text.txt"
        plate_text = None
        if plate_text_path.exists():
            try:
                plate_text = plate_text_path.read_text(encoding="utf-8").strip()
            except Exception as e:
                print(f"[warning] {plate_text_path} 읽기 실패: {e}")

        # 없으면 썸네일로 대체(요구사항 없지만 결과 구조 유지)
        if plate_path is None:
            plate_path = thumb_path

        results.append(
            {
                "clip_dir": clip_dir,
                "mp4_path": mp4_path,
                "thumb_path": thumb_path,
                "plate_path": plate_path,
                "plate_text": plate_text,
            }
        )

    return results


def run_model(
    job: dict,
    out_dir: Path,
    input_file: Path,
    bucket_name: str | None,
    expire_seconds: int,
) -> None:
    """
    실제 모델을 실행하여 급정거 감지 → clip 생성 → OCR 수행
    
    Args:
        job: job.json 내용
        out_dir: 출력 디렉토리 (task 단위)
        input_file: 입력 영상 파일
        bucket_name: GCS 버킷 이름 (None이면 로컬 경로 사용)
        expire_seconds: signed url 만료 시간
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print("0", flush=True)
    
    # 1) task 폴더에 job.json 저장
    job_json_path = out_dir / "job.json"
    if not job_json_path.exists():
        job_json_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")

    print("1", flush=True)
    
    # 2) input_file 확인
    if not input_file or not Path(input_file).exists():
        raise RuntimeError(f"input_file not found: {input_file}")
    
    input_file = Path(input_file)
    
    print("2", flush=True)
    # sudden_stop 모듈에서 급정거 감지 + clip 생성 + 차량 crop 한 번에 처리
    try:
        from sudden_stop.main import detect_and_generate_clips
    except ImportError as e:
        raise RuntimeError(f"sudden_stop 모듈을 import할 수 없습니다: {e}")

    print("3", flush=True)
    print("input_file: ", input_file)
    print("out_dir: ", out_dir)
    
    clips = detect_and_generate_clips(
        input_video_path=input_file,
        output_dir=out_dir,
    )

    print("4", flush=True)
    
    # 반환값 검증
    if clips is None:
        raise RuntimeError("sudden_stop.detect_and_generate_clips()가 None을 반환했습니다.")
    if not isinstance(clips, list):
        raise RuntimeError(f"sudden_stop.detect_and_generate_clips()가 리스트가 아닌 {type(clips)}를 반환했습니다.")
    if len(clips) == 0:
        print("[warning] 급정거 이벤트가 감지되지 않았습니다.")
        return
        
    print(clips[0])
    # 각 clip의 구조 검증
    for i, clip in enumerate(clips):
        if not hasattr(clip, 'clip_video_path'):
            raise RuntimeError(f"clip[{i}]에 clip_video_path 속성이 없습니다.")
        if not hasattr(clip, 'vehicle_crops'):
            raise RuntimeError(f"clip[{i}]에 vehicle_crops 속성이 없습니다.")
        if not isinstance(clip.clip_video_path, Path):
            raise RuntimeError(f"clip[{i}].clip_video_path가 Path 타입이 아닙니다: {type(clip.clip_video_path)}")
        if not isinstance(clip.vehicle_crops, list):
            raise RuntimeError(f"clip[{i}].vehicle_crops가 리스트가 아닙니다: {type(clip.vehicle_crops)}")
        if not clip.clip_video_path.exists():
            raise RuntimeError(f"clip[{i}].clip_video_path가 존재하지 않습니다: {clip.clip_video_path}")
    
    # 3) 각 clip에 대해 후처리 (썸네일, OCR)
    for i, clip in enumerate(clips, start=1):
        clip_dir = clip.clip_video_path.parent
        
        # 3-1) 썸네일 생성
        thumb_path = clip_dir / f"clip_{i}.jpg"
        if not thumb_path.exists():
            extract_first_frame(clip.clip_video_path, thumb_path)
        
        # 3-2) OCR 수행
        try:
            from ocr.interface_final import detect_license_plate
        except ImportError as e:
            raise RuntimeError(f"ocr 모듈을 import할 수 없습니다: {e}")
        
        clip_output_dir = clip_dir
        plate_result = detect_license_plate(
            vehicle_crops=clip.vehicle_crops,
            clip_output_dir=clip_output_dir,
        )
        
        if plate_result is None:
            print(f"[warning] clip[{i}]에서 번호판을 감지하지 못했습니다.")
        else:
            # plate_result 구조 검증
            if not hasattr(plate_result, 'plate_image_path'):
                raise RuntimeError(f"plate_result에 plate_image_path 속성이 없습니다.")
            if not hasattr(plate_result, 'plate_text_path'):
                raise RuntimeError(f"plate_result에 plate_text_path 속성이 없습니다.")
            
            # 경로 검증
            plate_image_path = Path(plate_result.plate_image_path)
            plate_text_path = Path(plate_result.plate_text_path)
            
            if not plate_image_path.exists():
                raise RuntimeError(f"plate_image_path가 존재하지 않습니다: {plate_image_path}")
            if not plate_text_path.exists():
                raise RuntimeError(f"plate_text_path가 존재하지 않습니다: {plate_text_path}")
        print("5", flush=True)


# -----------------------------
# 메인
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="job.json path")
    ap.add_argument("--bucket", default=None, help="(optional) GCS bucket name (없으면 ENV GCS_BUCKET_NAME 또는 input_url에서 추론)")
    ap.add_argument("--expire", type=int, default=DEFAULT_SIGNED_URL_EXPIRE_SECONDS, help="signed url expire seconds")
    args = ap.parse_args()

    job_path = Path(args.job)
    job = json.loads(job_path.read_text(encoding="utf-8"))

    job_id = job["job_id"]
    out_root = Path(job["out_root"])
    out_dir = out_root / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # (기존 동작 유지) input 파일 필요 시 다운로드
    input_file = download_input(job)
    if input_file is None:
        raise RuntimeError("input_file을 다운로드할 수 없습니다. job['input_path'] 또는 job['input_url']을 확인하세요.")
    print("input_file: ", input_file, flush=True)

    # bucket 결정
    bucket_name = infer_bucket_name(job, args.bucket)
    print("bucket_name: ", bucket_name, flush=True)

    # -----------------------------
    # 실제 모델 실행
    # -----------------------------
    run_model(job, out_dir, input_file=Path(input_file), bucket_name=bucket_name, expire_seconds=args.expire)
    # -----------------------------
    print("after run model", flush=True)

    # clip 폴더 스캔
    clip_items = scan_clips(out_dir)

    # event_type_id 결정(여러개면 첫번째를 기본으로 사용)
    event_type_ids = job.get("event_type_ids") or [1]
    default_event_type_id = event_type_ids[0]

    results = []
    for i, item in enumerate(clip_items, start=1):
        clip_dir: Path = item["clip_dir"]
        mp4_path: Path | None = item["mp4_path"]
        thumb_path: Path | None = item["thumb_path"]
        plate_path: Path | None = item["plate_path"]
        plate_text: str | None = item.get("plate_text")

        # 업로드 object 경로(버킷 내)
        # 예: results/<job_id>/<clip_dir_name>/<filename>
        base_obj = f"results/{job_id}/{clip_dir.name}"

        clip_url = None
        if mp4_path and mp4_path.exists():
            if UPLOAD_VIDEOS and bucket_name:
                clip_url = ensure_uploaded_and_get_url(
                    mp4_path,
                    bucket_name,
                    f"{base_obj}/{mp4_path.name}",
                    args.expire,
                )
            else:
                clip_url = str(mp4_path)

        thumb_url = None
        if thumb_path and thumb_path.exists():
            thumb_url = ensure_uploaded_and_get_url(
                thumb_path,
                bucket_name,
                f"{base_obj}/{thumb_path.name}",
                args.expire,
            )

        plate_url = None
        if plate_path and plate_path.exists():
            # thumb랑 같은 파일이면 중복 업로드 피하려고, 이미 thumb_url 있으면 재사용
            if thumb_path and plate_path.resolve() == thumb_path.resolve() and thumb_url:
                plate_url = thumb_url
            else:
                plate_url = ensure_uploaded_and_get_url(
                    plate_path,
                    bucket_name,
                    f"{base_obj}/{plate_path.name}",
                    args.expire,
                )

        results.append(
            {
                "event_type_id": default_event_type_id,
                "clip_path": clip_url,
                "thumbnail_img": thumb_url,          # ✅ 추가된 필드
                "occurred_time": now_iso_utc(),      # 정보 없으니 현재시간(필요 시 교체)
                "license_plate_img": plate_url,
                "license_plate_text": plate_text if plate_text else "",  # license_plate_text.txt에서 읽은 값 사용
            }
        )

    # 최종 result.json
    result = {
        "job_id": job_id,
        "task_id": job.get("task_id"),
        "video_id": job.get("video_id"),
        "input_file": str(input_file) if input_file else None,
        "results": results,
    }

    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] job_id={job_id} wrote {result_path}")
    print(f"[bucket] {bucket_name}")
    print(f"[clips] {len(results)} found")


if __name__ == "__main__":
    main()

