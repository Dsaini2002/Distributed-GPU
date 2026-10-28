import hashlib
import json
import re
import shutil
import uuid
import urllib.request

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / ".dgpu" / "recovery"
ID_RE = re.compile(r"^[a-f0-9]{12}$")


def directory(job_id):
    if not ID_RE.fullmatch(job_id):
        raise ValueError("Invalid job ID")
    return BASE / job_id


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_epoch(job_id, epoch, stages, blocks, fetch):
    base = directory(job_id)
    base.mkdir(parents=True, exist_ok=True)

    final = base / f"epoch_{epoch:05d}"
    temp = base / f".tmp_{uuid.uuid4().hex}"

    if final.exists():
        raise ValueError("Epoch checkpoint already exists")

    temp.mkdir()

    try:
        files = []

        for index, stage in enumerate(stages):
            data = fetch(
                stage, "snapshot",
                {"stage_id": stage["stage_id"]},
                binary=True
            )

            if not data:
                raise ValueError("Empty worker snapshot")

            name = f"stage_{index}.pt"
            (temp / name).write_bytes(data)

            files.append({
                "name": name,
                "size": len(data),
                "sha256": digest(data)
            })

        manifest = {
            "job_id": job_id,
            "epoch": epoch,
            "blocks": blocks,
            "stages": stages,
            "files": files
        }

        (temp / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8"
        )

        temp.rename(final)
        return final

    finally:
        if temp.exists():
            shutil.rmtree(temp)


def latest_checkpoint(job_id):
    base = directory(job_id)

    if not base.exists():
        return None

    folders = sorted(
        base.glob("epoch_*"),
        reverse=True
    )

    for folder in folders:
        try:
            info = json.loads(
                (folder / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )

            if info["job_id"] != job_id:
                continue

            if len(info["files"]) != len(info["stages"]):
                continue

            valid = True

            for index, item in enumerate(info["files"]):
                expected = f"stage_{index}.pt"

                if item["name"] != expected:
                    valid = False
                    break

                data = (folder / expected).read_bytes()

                if (
                    len(data) != item["size"]
                    or digest(data) != item["sha256"]
                ):
                    valid = False
                    break

            if valid:
                info["checkpoint_dir"] = str(folder)
                return info

        except (OSError, KeyError, ValueError, TypeError):
            continue

    return None


def restore_stage(stage, checkpoint, index, token):
    filename = (
        Path(checkpoint["checkpoint_dir"])
        / f"stage_{index}.pt"
    )

    data = filename.read_bytes()

    url = (
        "http://" + stage["address"]
        + ":45410/restore"
    )

    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": "Bearer " + token,
            "X-Stage-Id": stage["stage_id"],
            "Content-Type": "application/octet-stream"
        },
        method="POST"
    )

    with urllib.request.urlopen(
        request, timeout=240
    ) as response:
        response.read()
