import json
import os
import re
import shutil

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / ".dgpu" / "jobs"
ARCHIVE_DIR = ROOT / "checkpoints" / "studio_runs"

ID_PATTERN = re.compile(r"^[a-f0-9]{12}$")


def job_path(job_id):
    if not ID_PATTERN.fullmatch(job_id):
        raise ValueError("Invalid job ID")
    return JOBS_DIR / (job_id + ".json")


def save_job(record):
    if not record.get("job_id"):
        return

    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    destination = job_path(record["job_id"])
    temporary = destination.with_suffix(".json.part")

    temporary.write_text(
        json.dumps(record, indent=2, allow_nan=False),
        encoding="utf-8"
    )
    os.replace(temporary, destination)


def load_job(job_id):
    path = job_path(job_id)

    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def list_jobs(limit=50):
    if not JOBS_DIR.exists():
        return []

    paths = sorted(
        JOBS_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime_ns,
        reverse=True
    )

    records = []

    for path in paths[:limit]:
        try:
            record = load_job(path.stem)
            if record:
                records.append(record)
        except (ValueError, OSError, json.JSONDecodeError):
            continue

    return records


def load_latest_job():
    records = list_jobs(limit=1)
    return records[0] if records else None


def archive_checkpoint(source, job_id):
    job_path(job_id)

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    destination = ARCHIVE_DIR / (job_id + ".pt")
    temporary = ARCHIVE_DIR / (job_id + ".pt.part")

    if destination.exists():
        raise FileExistsError("Checkpoint archive already exists")

    try:
        shutil.copyfile(source, temporary)

        if temporary.stat().st_size == 0:
            raise ValueError("Empty checkpoint")

        os.replace(temporary, destination)

    finally:
        temporary.unlink(missing_ok=True)

    return destination
