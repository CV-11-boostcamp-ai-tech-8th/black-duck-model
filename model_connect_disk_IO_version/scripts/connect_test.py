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

def _run(cmd: list[str]) -> None:
    """subprocess 실행 (에러면 예외 발생)"""
    subprocess.run(cmd, check=True)

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

    # connect_test.py 기준 경로에서 secret 폴더 탐색
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

        # license plate 이미지: 파일명에 plate 들어간 이미지 우선
        plate_path = pick_first_file(clip_dir, ["*plate*.jpg", "*plate*.jpeg", "*plate*.png", "*plate*.webp", "*plate*.img"])

        # 없으면 썸네일로 대체(요구사항 없지만 결과 구조 유지)
        if plate_path is None:
            plate_path = thumb_path

        results.append(
            {
                "clip_dir": clip_dir,
                "mp4_path": mp4_path,
                "thumb_path": thumb_path,
                "plate_path": plate_path,
            }
        )

    return results

def make_demo_task_assets(job: dict, out_dir: Path, n_clips: int = 2) -> None:
    """
    out_dir (= out_root/job_id) 아래에:
      - job.json (task당 1개)
      - clip_1..clip_n 폴더 및 clip_*.mp4 / clip_*.img 생성
    또한 input_path에 파일이 없으면 dummy input.mp4도 생성해서 '실감나게' 맞춘다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) task 폴더에 job.json 생성 (clip 폴더들과 같은 위치)
    job_json_path = out_dir / "job.json"
    if not job_json_path.exists():
        job_json_path.write_text(
            json.dumps(job, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 2) input_path에 더미 input.mp4 생성(없을 때만)
    #    (실제론 download_input이 받아오지만, 데모에선 로컬만으로도 동작하게)
    input_path = Path(job.get("input_path", ""))
    if input_path:
        input_path.parent.mkdir(parents=True, exist_ok=True)
        if not input_path.exists():
            input_path.write_bytes(b"")  # 빈 파일(더미)

    # 3) clip 폴더/파일 생성
    for i in range(1, n_clips + 1):
        clip_dir = out_dir / f"clip_{i}"
        clip_dir.mkdir(parents=True, exist_ok=True)

        mp4_path = clip_dir / f"clip_{i}.mp4"
        if not mp4_path.exists():
            mp4_path.write_bytes(b"")  # 더미 mp4

        # .img는 실제 이미지가 아닐 수 있으니, 데모에선 텍스트로라도 "있다"를 보여줌
        img_path = clip_dir / f"clip_{i}.img"
        if not img_path.exists():
            img_path.write_text(
                f"DEMO THUMBNAIL for {job.get('job_id')} / clip_{i}\n",
                encoding="utf-8",
            )

        # (선택) 번호판 이미지도 있는 척 하나 더 만들고 싶으면 켜도 됨
        plate_path = clip_dir / f"plate_{i}.jpg"
        if not plate_path.exists():
            plate_path.write_bytes(b"")

def make_demo_task_assets_from_input(
    job: dict,
    out_dir: Path,
    input_file: Path,
    n_clips: int = 2,
) -> None:
    """
    input_file(다운받은 input.mp4)을 이용해 데모 결과물 생성

    out_dir (= out_root/job_id) 아래에:
      - job.json (task당 1개)
      - clip_1..clip_n 폴더
        - clip_1.mp4 : input 첫 1초
        - clip_2.mp4 : input 마지막 1초 (n_clips>=2일 때)
        - clip_1.jpg / clip_2.jpg : input 첫 프레임 기반 썸네일
      - license_plate.jpg : input 첫 프레임 기반(번호판 이미지 데모)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) task 폴더에 job.json 저장(없으면)
    job_json_path = out_dir / "job.json"
    if not job_json_path.exists():
        job_json_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2) input_file 확인
    if not input_file or not Path(input_file).exists():
        raise RuntimeError(f"input_file not found: {input_file}")

    input_file = Path(input_file)

    # 3) input의 첫 프레임을 jpg로 뽑기 (공통으로 재사용)
    first_frame_jpg = out_dir / "first_frame.jpg"
    if not first_frame_jpg.exists():
        _run([
            "ffmpeg", "-y",
            "-i", str(input_file),
            "-frames:v", "1",
            "-q:v", "2",
            str(first_frame_jpg),
        ])

    # 4) 번호판 이미지(데모) = 첫 프레임 복사
    license_plate_jpg = out_dir / "license_plate.jpg"
    if not license_plate_jpg.exists():
        license_plate_jpg.write_bytes(first_frame_jpg.read_bytes())

    # 5) clip_1: 첫 1초 잘라서 저장
    clip1_dir = out_dir / "clip_1"
    clip1_dir.mkdir(parents=True, exist_ok=True)

    clip1_mp4 = clip1_dir / "clip_1.mp4"
    if not clip1_mp4.exists():
        # re-encode 방식(호환성 좋음). 빠른 복사(stream copy)보다 안정적.
        _run([
            "ffmpeg", "-y",
            "-ss", "0",
            "-i", str(input_file),
            "-t", "1",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(clip1_mp4),
        ])

    clip1_thumb = clip1_dir / "clip_1.jpg"
    if not clip1_thumb.exists():
        extract_first_frame(clip1_mp4, clip1_thumb)


    # 6) clip_2: 마지막 1초 잘라서 저장 (n_clips>=2일 때)
    if n_clips >= 2:
        clip2_dir = out_dir / "clip_2"
        clip2_dir.mkdir(parents=True, exist_ok=True)

        clip2_mp4 = clip2_dir / "clip_2.mp4"
        if not clip2_mp4.exists():
            # 마지막 1초: -sseof -1 사용 (ffmpeg에서 end 기준 seek)
            _run([
                "ffmpeg", "-y",
                "-sseof", "-1",
                "-i", str(input_file),
                "-t", "1",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-movflags", "+faststart",
                str(clip2_mp4),
            ])

        clip2_thumb = clip2_dir / "clip_2.jpg"
        if not clip2_thumb.exists():
            extract_first_frame(clip2_mp4, clip2_thumb)


    # (원하면) first_frame.jpg는 남겨도 되고 지워도 됨
    # first_frame_jpg.unlink(missing_ok=True)

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

    # bucket 결정
    bucket_name = infer_bucket_name(job, args.bucket)

    # -----------------------------
    # DEMO CLIP 생성 (필요할 때만 ON)
    # -----------------------------
    # make_demo_task_assets(job, out_dir, n_clips=2)  # <-- ON: 주석 해제
    make_demo_task_assets_from_input(job, out_dir, input_file=Path(input_file), n_clips=2)
    # -----------------------------

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
                "license_plate_text": f"DUMMY-{clip_dir.name}",  # 실제 값 있으면 여기서 교체
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
