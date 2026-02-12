#!/usr/bin/env python3
import argparse, json, os, time
from pathlib import Path
from datetime import datetime, timezone

import requests
try:
    from google.cloud import storage
    HAVE_GCS = True
except ImportError:
    HAVE_GCS = False


def download_input(job: dict) -> Path | None:
    """
    Always materialize input at job['input_path']
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
            print("download for HTTP")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with input_path.open("wb") as f:
                    for chunk in r.iter_content(8192):
                        if chunk:
                            f.write(chunk)
        return input_path

    # local input already exists at input_path
    if input_path.exists():
        return input_path

    raise RuntimeError("No input_url and input_path does not exist")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="job.json path")
    args = ap.parse_args()

    job_path = Path(args.job)
    job = json.loads(job_path.read_text())
    job_id = job["job_id"]
    out_root = Path(job["out_root"])
    print("out_root : ", out_root)
    out_dir = out_root / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 입력 확보 (필요 시 다운로드)
    input_file = download_input(job)
    # input_file = download_input(job, in_dir ) # 지금은 video_dir를 변경하여, 다운받는 파일의 위치를 지정가능

    # ----- 여기서 실제 모델 호출 대신 더미 출력 생성 -----
    now_iso = datetime(2001, 9, 2, tzinfo=timezone.utc).isoformat()
    event_type_ids = job.get("event_type_ids") or [1]
    results = [
        {
            "event_type_id": event_type_ids[0],
            "clip_path": f"{out_dir}/dummy_clip.mp4",
            "occurred_time": now_iso,
            "thumbnail_img": f"{out_dir}/thumbnail_img.jpg", # url로 바꾸기
            "license_plate_img": f"{out_dir}/dummy_plate.jpg",
            "license_plate_text": f"DUMMY-{job_id}",
        }
    ]

    result = {"job_id": job_id, "results": results, "input_file": str(input_file) if input_file else None}
    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    print(f"[dummy] job_id={job_id} wrote {result_path}")
    print(f"[HAVE_GCS] : {HAVE_GCS}")


if __name__ == "__main__":
    main()
