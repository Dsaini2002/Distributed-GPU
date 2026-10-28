from pathlib import Path


def replace_once(source, old, new, label):
    if old not in source:
        raise RuntimeError(
            f"Cannot patch {label}; source differs from expected version"
        )

    return source.replace(old, new, 1)


path = Path("studio/train_engine.py")
original = path.read_text(encoding="utf-8-sig")

if "from .history_store import" not in original:
    source = original

    source = replace_once(
        source,
        "from .model_partition import plan_model",
        """from .model_partition import plan_model
from .history_store import (
    save_job, load_latest_job, archive_checkpoint
)
from datetime import datetime, timezone
import re

EPOCH_RE = re.compile(
    r"^Epoch\\s+(\\d+)/(\\d+)\\s+\\|\\s+MSE:\\s+([0-9.eE+-]+)"
)""",
        "imports"
    )

    source = replace_once(
        source,
        'job = {"status": "idle", "logs": []}',
        '''job = load_latest_job() or {"status": "idle", "logs": []}

if job.get("status") == "running":
    job["status"] = "interrupted"
    job["error"] = "Controller restarted during training"
    job.setdefault("logs", []).append(
        "Training interrupted. Restart workers before a new job."
    )
    save_job(job)''',
        "startup state"
    )

    source = replace_once(
        source,
        """def update(**values):
    with lock:
        job.update(values)


def log(message):
    print("[DGPU]", message)
    with lock:
        job.setdefault("logs", []).append(str(message))
        job["logs"] = job["logs"][-100:]


def get_status():
    with lock:
        return dict(job, logs=list(job.get("logs", [])))""",
        """def update(**values):
    with lock:
        job.update(values)
        save_job(job)


def log(message):
    print("[DGPU]", message)

    with lock:
        job.setdefault("logs", []).append(str(message))
        job["logs"] = job["logs"][-100:]

        match = EPOCH_RE.match(str(message))

        if match:
            job.setdefault("history", []).append({
                "epoch": int(match.group(1)),
                "total_epochs": int(match.group(2)),
                "mse": float(match.group(3))
            })

        save_job(job)


def get_status():
    with lock:
        return dict(
            job,
            logs=list(job.get("logs", [])),
            history=list(job.get("history", []))
        )""",
        "state functions"
    )

    source = replace_once(
        source,
        """final = merge_stages(stages, blocks)
        log("Merged checkpoint: " + str(final))
        update(status="completed", output=str(final))""",
        """final = merge_stages(stages, blocks)

        archived = archive_checkpoint(
            final, get_status()["job_id"]
        )

        log("Merged checkpoint: " + str(final))
        log("Archived checkpoint: " + str(archived))

        update(
            status="completed",
            output=str(final),
            archived_output=str(archived)
        )""",
        "checkpoint archiving"
    )

    source = replace_once(
        source,
        '''job.update({
            "status": "running",
            "logs": [],
            "epoch": 0,
            "loss": None,
            "stages": []
        })''',
        '''job.update({
            "job_id": uuid.uuid4().hex[:12],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config": dict(config),
            "status": "running",
            "logs": [],
            "history": [],
            "epoch": 0,
            "loss": None,
            "stages": []
        })
        save_job(job)''',
        "job creation"
    )

    path.write_text(source, encoding="utf-8")
    print("Training engine persistence enabled")
else:
    print("Training engine already patched")


path = Path("studio/server.py")
source = path.read_text(encoding="utf-8-sig")

if "studio_history_router" not in source:
    source += """

from .history_ui import router as studio_history_router
app.include_router(studio_history_router)
"""
    path.write_text(source, encoding="utf-8")
    print("Job history routes enabled")


path = Path("studio/monitor_api.py")
source = path.read_text(encoding="utf-8-sig")

old = 'history = history_from_logs(status.get("logs", []))'
new = '''history = status.get("history") or history_from_logs(
        status.get("logs", [])
    )'''

if old in source:
    path.write_text(
        source.replace(old, new, 1),
        encoding="utf-8"
    )
    print("Live monitor persisted history enabled")


for filename in ["studio/train_ui.py", "studio/monitor_ui.py"]:
    path = Path(filename)
    source = path.read_text(encoding="utf-8-sig")

    if 'href="/history"' in source:
        continue

    anchor = '<a href="/train">Training</a>'

    if anchor in source:
        source = source.replace(
            anchor,
            anchor + '<a href="/history">Job History</a>',
            1
        )

    elif '<h1>Distributed-GPU Training Studio</h1>' in source:
        source = source.replace(
            '<h1>Distributed-GPU Training Studio</h1>',
            '<h1>Distributed-GPU Training Studio</h1>'
            '<p><a href="/history">Job History</a></p>',
            1
        )

    path.write_text(source, encoding="utf-8")

print("PERSISTENT TRAINING HISTORY PATCH COMPLETE")
