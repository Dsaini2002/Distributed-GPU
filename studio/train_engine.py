import csv
import io
import json
import os
import uuid
import random
import threading
import importlib.util
import urllib.request

from pathlib import Path

import torch
import torch.nn as nn

from .model_partition import plan_model
from .recovery import (
    save_epoch, latest_checkpoint, restore_stage
)
from .history_store import (
    load_job,
    save_job, load_latest_job, archive_checkpoint
)
from datetime import datetime, timezone
import re

EPOCH_RE = re.compile(
    r"^Epoch\s+(\d+)/(\d+)\s+\|\s+MSE:\s+([0-9.eE+-]+)"
)

ROOT = Path(__file__).resolve().parent.parent
lock = threading.Lock()
job = load_latest_job() or {"status": "idle", "logs": []}

if job.get("status") == "running":
    job["status"] = "interrupted"
    job["error"] = "Controller restarted during training"
    job.setdefault("logs", []).append(
        "Training interrupted. Restart workers before a new job."
    )
    save_job(job)


def update(**values):
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
        )


def safe_file(folder, filename, suffix):
    if Path(filename).name != filename:
        raise ValueError("Select a file from the dropdown")

    path = ROOT / folder / filename

    if path.suffix.lower() != suffix or not path.is_file():
        raise ValueError("File missing or unsupported")

    return path


def list_options():
    return {
        "models": sorted(
            p.name for p in (ROOT / "models").glob("*.py")
        ),
        "datasets": sorted(
            p.name for p in (ROOT / "data").glob("*.csv")
        )
    }


def get_blocks(path):
    spec = importlib.util.spec_from_file_location(
        "dgpu_selected_model", str(path)
    )
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load model")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.get_model()

    if not isinstance(model, nn.Sequential):
        raise ValueError("Currently requires nn.Sequential")

    blocks = []

    for layer in model.children():
        if isinstance(layer, nn.Linear):
            if blocks and blocks[-1]["out"] != layer.in_features:
                raise ValueError("Linear dimensions do not match")

            blocks.append({
                "in": layer.in_features,
                "out": layer.out_features,
                "relu": False
            })
        elif isinstance(layer, nn.ReLU):
            if not blocks or blocks[-1]["relu"]:
                raise ValueError("Unexpected ReLU")
            blocks[-1]["relu"] = True
        else:
            raise ValueError(
                "Unsupported layer: " + type(layer).__name__
            )

    if not blocks or blocks[-1]["out"] != 1:
        raise ValueError("Regression output must be one value")

    return blocks


def read_data(path, input_size):
    with path.open(
        newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames

        if not columns or columns.count("target") != 1:
            raise ValueError("CSV requires a target column")

        features = [x for x in columns if x != "target"]

        if len(features) != input_size:
            raise ValueError(
                f"Expected {input_size} numeric features"
            )

        if len(set(columns)) != len(columns):
            raise ValueError("Duplicate CSV columns")

        rows = []
        for row in reader:
            import math

            x = [float(row[c]) for c in features]
            y = float(row["target"])

            if not all(math.isfinite(v) for v in x + [y]):
                raise ValueError("CSV has non-finite values")

            rows.append((x, y))

            if len(rows) > 20000:
                raise ValueError(
                    "MVP limit: 20,000 training samples"
                )

    if not rows:
        raise ValueError("Dataset is empty")

    return rows


def request_worker(stage, route, payload, binary=False):
    token = os.environ["DGPU_TOKEN"]

    url = (
        "http://" + stage["address"]
        + ":45410/" + route
    )

    body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + token
        },
        method="POST"
    )

    with urllib.request.urlopen(
        request, timeout=240
    ) as response:
        data = response.read()

    if binary:
        return data

    return json.loads(data)


def build_stage_model(blocks):
    layers = []

    for b in blocks:
        layers.append(nn.Linear(b["in"], b["out"]))
        if b["relu"]:
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)


def merge_stages(stages, blocks):
    all_layers = []

    for stage in stages:
        data = request_worker(
            stage, "weights",
            {"stage_id": stage["stage_id"]},
            binary=True
        )

        state = torch.load(
            io.BytesIO(data),
            map_location="cpu",
            weights_only=True
        )

        stage_model = build_stage_model(stage["blocks"])
        stage_model.load_state_dict(state, strict=True)
        all_layers.extend(list(stage_model.children()))

    full_model = nn.Sequential(*all_layers)

    output_dir = ROOT / "checkpoints"
    output_dir.mkdir(exist_ok=True)

    final = output_dir / "studio_merged_model.pt"
    temp = output_dir / "studio_merged_model.pt.part"

    torch.save({
        "blocks": blocks,
        "model_state_dict": full_model.state_dict(),
        "task": "regression"
    }, temp)

    temp.replace(final)
    return final


def train(config, resume_state=None):
    configured = []

    try:
        from .server import snapshot

        model_path = safe_file(
            "models", config["model"], ".py"
        )
        data_path = safe_file(
            "data", config["dataset"], ".csv"
        )

        blocks = get_blocks(model_path)
        rows = read_data(data_path, blocks[0]["in"])

        if resume_state is None:
            stages = plan_model(
                blocks,
                snapshot(),
                config["batch_size"]
            )
            first_epoch = 0
        else:
            if blocks != resume_state["blocks"]:
                raise ValueError(
                    "Model architecture changed since checkpoint"
                )

            stages = [
                dict(s) for s in resume_state["stages"]
            ]
            first_epoch = resume_state["epoch"]

            active_nodes = snapshot()

            for stage in stages:
                matching = any(
                    n["online"]
                    and n["hostname"] == stage["hostname"]
                    and n.get("address") == stage["address"]
                    and any(
                        g["index"] == stage["gpu_index"]
                        and g["name"] == stage["gpu"]
                        for g in n["gpus"]
                    )
                    for n in active_nodes
                )

                if not matching:
                    raise ValueError(
                        "Original worker unavailable: "
                        + stage["hostname"]
                    )

            # Single-job prototype: clear stale model stages.
            cleared = set()

            for stage in stages:
                if stage["address"] in cleared:
                    continue

                request_worker(stage, "clear", {})
                cleared.add(stage["address"])

        run_id = uuid.uuid4().hex[:12]

        for i, stage in enumerate(stages):
            stage["stage_id"] = f"{run_id}_{i}"

        update(stages=[
            {
                "computer": s["hostname"],
                "gpu": s["gpu"],
                "layers": len(s["blocks"]),
                "estimated_mib": s["estimated_mib"]
            }
            for s in stages
        ])

        log(
            f"Training {len(blocks)} Linear layers "
            f"using {len(stages)} GPU stages"
        )

        for index, stage in enumerate(stages):
            request_worker(stage, "configure", {
                "stage_id": stage["stage_id"],
                "gpu_index": stage["gpu_index"],
                "blocks": stage["blocks"],
                "lr": config["lr"]
            })

            configured.append(stage)

            if resume_state is not None:
                restore_stage(
                    stage,
                    resume_state,
                    index,
                    os.environ["DGPU_TOKEN"]
                )

            log(
                f"{stage['hostname']} / {stage['gpu']} "
                f"= {len(stage['blocks'])} layers"
            )

        batch_size = config["batch_size"]

        for epoch in range(first_epoch, config["epochs"]):
            indices = list(range(len(rows)))
            random.Random(42 + epoch).shuffle(indices)

            total_loss = 0.0
            seen = 0

            for start in range(0, len(indices), batch_size):
                batch_ids = indices[start:start + batch_size]
                inputs = [rows[i][0] for i in batch_ids]
                targets = [rows[i][1] for i in batch_ids]

                values = inputs

                for stage in stages:
                    reply = request_worker(
                        stage, "forward",
                        {
                            "stage_id": stage["stage_id"],
                            "x": values
                        }
                    )
                    values = reply["y"]

                if len(values) != len(targets):
                    raise ValueError("Output batch size mismatch")

                differences = [
                    float(v[0]) - float(y)
                    for v, y in zip(values, targets)
                ]

                n = len(differences)
                loss = sum(d*d for d in differences) / n
                total_loss += loss * n
                seen += n

                gradient = [
                    [2.0 * d / n] for d in differences
                ]

                for stage in reversed(stages):
                    reply = request_worker(
                        stage, "backward",
                        {
                            "stage_id": stage["stage_id"],
                            "grad": gradient
                        }
                    )
                    gradient = reply["grad"]

            average = total_loss / seen
            update(epoch=epoch + 1, loss=round(average, 8))

            log(
                f"Epoch {epoch+1}/{config['epochs']} "
                f"| MSE: {average:.6f}"
            )

            saved_dir = save_epoch(
                get_status()["job_id"],
                epoch + 1,
                stages,
                blocks,
                request_worker
            )

            log(
                "Recovery checkpoint saved: "
                + str(saved_dir)
            )

        final = merge_stages(stages, blocks)

        archived = archive_checkpoint(
            final, get_status()["job_id"]
        )

        log("Merged checkpoint: " + str(final))
        log("Archived checkpoint: " + str(archived))

        update(
            status="completed",
            output=str(final),
            archived_output=str(archived)
        )

    except Exception as error:
        log("ERROR: " + str(error))
        update(status="failed", error=str(error))

    finally:
        for stage in configured:
            try:
                request_worker(
                    stage, "reset",
                    {"stage_id": stage["stage_id"]}
                )
            except Exception:
                pass


def start_training(config):
    with lock:
        if job["status"] == "running":
            raise ValueError("A training job is already running")

        job.clear()
        job.update({
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
        save_job(job)

    threading.Thread(
        target=train,
        args=(config,),
        daemon=True
    ).start()

    return {"status": "running"}



def resume_training(job_id):
    checkpoint = latest_checkpoint(job_id)

    if checkpoint is None:
        raise ValueError(
            "No valid recovery checkpoint for this job"
        )

    saved = load_job(job_id)

    if saved is None:
        raise ValueError("Job not found")

    if saved.get("status") not in ("failed", "interrupted"):
        raise ValueError(
            "Only failed or interrupted jobs can resume"
        )

    config = saved.get("config")

    if not config:
        raise ValueError("Training configuration missing")

    with lock:
        if job.get("status") == "running":
            raise ValueError("Another training job is running")

        job.clear()
        job.update(saved)

        completed = checkpoint["epoch"]

        job["status"] = "running"
        job["error"] = None
        job["epoch"] = completed

        job["history"] = [
            item for item in job.get("history", [])
            if item["epoch"] <= completed
        ]

        history = job["history"]
        job["loss"] = (
            history[-1]["mse"] if history else None
        )

        job.setdefault("logs", []).append(
            f"Resume requested from epoch {completed}"
        )

        save_job(job)

    threading.Thread(
        target=train,
        args=(dict(config), checkpoint),
        daemon=True
    ).start()

    return {
        "status": "running",
        "job_id": job_id,
        "resumed_from_epoch": completed
    }
