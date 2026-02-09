#!/usr/bin/env python3
import argparse
import json
import mimetypes
import os
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta

import requests
import cv2
import numpy as np
from PIL import Image

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


def save_numpy_array_to_temp_file(array: np.ndarray, suffix: str = ".jpg") -> Path:
    """
    numpy array를 임시 파일로 저장 (GCS 업로드용)
    
    Args:
        array: numpy array 이미지
        suffix: 파일 확장자
    
    Returns:
        임시 파일 경로
    """
    # BGR to RGB 변환 (OpenCV 형식인 경우)
    if len(array.shape) == 3 and array.shape[2] == 3:
        array_rgb = array[:, :, ::-1]
    else:
        array_rgb = array
    
    # 임시 파일 생성
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()
    
    # PIL Image로 변환하여 저장
    img = Image.fromarray(array_rgb)
    img.save(temp_path, "PNG" if suffix == ".png" else "JPEG")
    
    return temp_path


def save_video_frames_to_temp_file(frames: list[np.ndarray], fps: float, suffix: str = ".mp4") -> Path:
    """
    프레임 리스트를 임시 비디오 파일로 저장 (GCS 업로드용)
    
    Args:
        frames: 프레임 리스트 (numpy array)
        fps: 프레임레이트
        suffix: 파일 확장자
    
    Returns:
        임시 파일 경로
    """
    if not frames:
        raise ValueError("frames 리스트가 비어있습니다")
    
    # 임시 파일 생성
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()
    
    # 첫 프레임으로 크기 결정
    h, w = frames[0].shape[:2]
    
    # 비디오 writer 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (w, h))
    
    # 프레임 저장
    for frame in frames:
        writer.write(frame)
    
    writer.release()
    
    return temp_path


def ensure_uploaded_and_get_url_from_memory(
    data: np.ndarray | list[np.ndarray] | None,
    bucket_name: str | None,
    object_name: str,
    expire_seconds: int,
    is_video: bool = False,
    fps: float = 30.0,
) -> str | None:
    """
    메모리 데이터를 임시 파일로 저장 후 업로드 (필요시에만)
    
    Args:
        data: numpy array (이미지) 또는 list[np.ndarray] (비디오 프레임)
        bucket_name: GCS 버킷 이름 (None이면 로컬 경로 반환)
        object_name: GCS object 이름
        expire_seconds: signed url 만료 시간
        is_video: 비디오인지 여부
        fps: 비디오 프레임레이트 (is_video=True일 때만 사용)
    
    Returns:
        signed URL 또는 None (bucket_name이 없으면)
    """
    if data is None:
        return None
    
    # bucket_name이 없으면 None 반환 (로컬 경로 불필요)
    if not bucket_name:
        return None
    
    # 임시 파일로 저장
    if is_video and isinstance(data, list):
        temp_path = save_video_frames_to_temp_file(data, fps)
    else:
        temp_path = save_numpy_array_to_temp_file(data, suffix=".jpg")
    
    try:
        # GCS 업로드
        blob = upload_to_gcs(temp_path, bucket_name, object_name)
        url = signed_url_for_blob(blob, expire_seconds)
        return url
    finally:
        # 임시 파일 삭제
        if temp_path.exists():
            temp_path.unlink()


def download_input(job: dict) -> Path | None:
    """
    기존 동작 유지: job['input_url'] 있으면 job['input_path']로 다운로드.
    """
    input_path = Path(job["input_path"])
    input_path.parent.mkdir(parents=True, exist_ok=True)

    # 로컬 파일이 이미 있으면 다운로드 건너뛰기
    if input_path.exists():
        print(f"[info] 로컬 파일이 이미 존재합니다: {input_path}")
        return input_path

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
            try:
                with requests.get(url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with input_path.open("wb") as f:
                        for chunk in r.iter_content(8192):
                            if chunk:
                                f.write(chunk)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 400:
                    raise RuntimeError(
                        f"GCS signed URL이 만료되었거나 잘못되었습니다. "
                        f"새로운 URL을 받아야 합니다. 원본 에러: {e}"
                    )
                raise
        return input_path

    if input_path.exists():
        return input_path

    # 입력이 필요 없는 경우도 있을 수 있으니 None 반환
    return None


def run_model(
    job: dict,
    out_dir: Path,
    input_file: Path,
    bucket_name: str | None,
    expire_seconds: int,
) -> list:
    """
    실제 모델을 실행하여 급정거 감지 → clip 생성 → OCR 수행 (메모리 기반)
    
    Args:
        job: job.json 내용
        out_dir: 출력 디렉토리 (task 단위)
        input_file: 입력 영상 파일
        bucket_name: GCS 버킷 이름 (None이면 로컬 경로 사용)
        expire_seconds: signed url 만료 시간
    
    Returns:
        list[dict]: 각 clip의 결과 정보 (메모리 기반)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1) task 폴더에 job.json 저장
    job_json_path = out_dir / "job.json"
    if not job_json_path.exists():
        job_json_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # 2) input_file 확인
    if not input_file or not Path(input_file).exists():
        raise RuntimeError(f"input_file not found: {input_file}")
    
    input_file = Path(input_file)
    
    # sudden_stop 모듈에서 급정거 감지 + clip 생성 + 차량 crop 한 번에 처리
    try:
        from sudden_stop.main import detect_and_generate_clips
    except ImportError as e:
        raise RuntimeError(f"sudden_stop 모듈을 import할 수 없습니다: {e}")
    
    # 메모리 기반으로 clip 생성 (save_to_disk=False)
    clips = detect_and_generate_clips(
        input_video_path=input_file,
        output_dir=out_dir,
        save_to_disk=False,  # 메모리에만 저장
    )
    
    # 반환값 검증
    if clips is None:
        raise RuntimeError("sudden_stop.detect_and_generate_clips()가 None을 반환했습니다.")
    if not isinstance(clips, list):
        raise RuntimeError(f"sudden_stop.detect_and_generate_clips()가 리스트가 아닌 {type(clips)}를 반환했습니다.")
    if len(clips) == 0:
        print("[warning] 급정거 이벤트가 감지되지 않았습니다.")
        return []
    
    # 3) 각 clip에 대해 OCR 수행
    try:
        from ocr.interface_final import detect_license_plate
    except ImportError as e:
        raise RuntimeError(f"ocr 모듈을 import할 수 없습니다: {e}")
    
    clip_results = []
    
    for clip in clips:
        # OCR 수행 (clip_output_dir은 detect_and_recognize 내부에서 필요할 수 있으므로 임시 디렉토리 생성)
        clip_output_dir = out_dir / f"clip_{clip.clip_id}"
        clip_output_dir.mkdir(parents=True, exist_ok=True)
        
        plate_result = detect_license_plate(
            vehicle_crops=clip.vehicle_crops,
            clip_output_dir=clip_output_dir,
        )
        
        # plate_result에서 이미지와 텍스트를 메모리로 로드
        plate_image = None
        plate_text = None
        if plate_result:
            # 이미지 파일 읽기
            if hasattr(plate_result, 'plate_image_path') and plate_result.plate_image_path:
                plate_image_path = Path(plate_result.plate_image_path)
                if plate_image_path.exists():
                    plate_image = cv2.imread(str(plate_image_path))
                    if plate_image is not None:
                        plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
            
            # 텍스트 파일 읽기
            if hasattr(plate_result, 'plate_text_path') and plate_result.plate_text_path:
                plate_text_path = Path(plate_result.plate_text_path)
                if plate_text_path.exists():
                    with open(plate_text_path, 'r', encoding='utf-8') as f:
                        plate_text = f.read().strip()
        
        # clip 결과 구성
        clip_result = {
            "clip_id": clip.clip_id,
            "video_frames": clip.video_frames,  # 메모리에 저장
            "thumbnail": clip.thumbnail,  # 메모리에 저장
            "vehicle_crops": clip.vehicle_crops,  # 메모리에 저장
            "plate_image": plate_image,  # 메모리에 저장 (numpy array)
            "plate_text": plate_text,  # 메모리에 저장 (string)
            "fps": clip.fps,  # 프레임레이트
        }
        
        clip_results.append(clip_result)
    
    return clip_results


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

    # bucket 결정
    bucket_name = infer_bucket_name(job, args.bucket)

    # -----------------------------
    # 실제 모델 실행 (메모리 기반)
    # -----------------------------
    clip_results = run_model(
        job, out_dir, 
        input_file=Path(input_file), 
        bucket_name=bucket_name, 
        expire_seconds=args.expire
    )
    # -----------------------------

    # event_type_id 결정(여러개면 첫번째를 기본으로 사용)
    event_type_ids = job.get("event_type_ids") or [1]
    default_event_type_id = event_type_ids[0]

    # clip_results를 result.json 형식으로 변환
    results = []
    for clip_result in clip_results:
        clip_id = clip_result["clip_id"]
        video_frames = clip_result.get("video_frames")
        thumbnail = clip_result.get("thumbnail")
        plate_image = clip_result.get("plate_image")
        plate_text = clip_result.get("plate_text", "")
        
        # 업로드 object 경로(버킷 내)
        base_obj = f"results/{job_id}/clip_{clip_id}"
        
        # 비디오 URL (필요시에만 업로드)
        clip_url = None
        if video_frames and UPLOAD_VIDEOS and bucket_name:
            fps = clip_result.get("fps", 30.0)  # clip에서 fps 가져오기
            clip_url = ensure_uploaded_and_get_url_from_memory(
                video_frames,
                bucket_name,
                f"{base_obj}/clip_{clip_id}.mp4",
                args.expire,
                is_video=True,
                fps=fps,
            )
        
        # 썸네일 URL
        thumb_url = None
        if thumbnail is not None and bucket_name:
            thumb_url = ensure_uploaded_and_get_url_from_memory(
                thumbnail,
                bucket_name,
                f"{base_obj}/thumbnail.jpg",
                args.expire,
                is_video=False,
            )
        
        # 번호판 이미지 URL
        plate_url = None
        if plate_image is not None and bucket_name:
            plate_url = ensure_uploaded_and_get_url_from_memory(
                plate_image,
                bucket_name,
                f"{base_obj}/license_plate_image.png",
                args.expire,
                is_video=False,
            )
        elif thumbnail is not None and bucket_name:
            # 번호판 이미지가 없으면 썸네일 사용
            plate_url = thumb_url
        
        results.append(
            {
                "event_type_id": default_event_type_id,
                "clip_path": clip_url,
                "thumbnail_img": thumb_url,
                "occurred_time": now_iso_utc(),
                "license_plate_img": plate_url,
                "license_plate_text": plate_text if plate_text else "",
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

