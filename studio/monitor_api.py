import json
import os
import hmac
import re

from pathlib import Path

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from evaluate_model import load_model, load_csv, evaluate
from .train_engine import get_status, safe_file

router = APIRouter()
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "checkpoints" / "studio_merged_model.pt"
REPORT_PATH = ROOT / "outputs" / "studio_evaluation_report.json"

EPOCH_PATTERN = re.compile(
    r"^Epoch\s+(\d+)/(\d+)\s+\|\s+MSE:\s+([0-9.eE+-]+)"
)


def authorize(value):
    token = os.environ.get("DGPU_TOKEN", "")

    if not token or not hmac.compare_digest(
        value, "Bearer " + token
    ):
        raise HTTPException(401, "Invalid authentication token")


def history_from_logs(logs):
    result = []

    for line in logs:
        match = EPOCH_PATTERN.search(line.strip())

        if match:
            result.append({
                "epoch": int(match.group(1)),
                "total_epochs": int(match.group(2)),
                "mse": float(match.group(3))
            })

    return result


def trained_model_path():
    status = get_status()

    from .server import snapshot
    online_names = {
        n["hostname"] for n in snapshot()
        if n["online"]
    }

    offline_workers = []

    if status.get("status") == "running":
        offline_workers = sorted(set(
            stage["computer"]
            for stage in status.get("stages", [])
            if stage["computer"] not in online_names
        ))
    state = status.get("status")

    if state == "running":
        raise HTTPException(
            409, "Training is still running"
        )

    if state == "failed":
        raise HTTPException(
            409, "Last training failed"
        )

    if state == "completed":
        saved = status.get("output")

        if not saved or Path(saved).resolve() != MODEL_PATH.resolve():
            raise HTTPException(
                409, "Unexpected model output path"
            )

    if not MODEL_PATH.is_file():
        raise HTTPException(404, "No trained model found")

    if MODEL_PATH.stat().st_size == 0:
        raise HTTPException(409, "Saved model is empty")

    return MODEL_PATH


class EvaluationRequest(BaseModel):
    train_dataset: str
    val_dataset: str


@router.get("/api/studio/monitor")
def monitor():
    status = get_status()
    history = status.get("history") or history_from_logs(
        status.get("logs", [])
    )

    return {
        "status": status.get("status", "idle"),
        "epoch": status.get("epoch", 0),
        "loss": status.get("loss"),
        "stages": status.get("stages", []),
        "logs": status.get("logs", []),
        "history": history,
        "output": status.get("output"),
        "error": status.get("error"),
        "offline_workers": offline_workers
    }


@router.post("/api/studio/evaluate")
def evaluate_job(
    request: EvaluationRequest,
    authorization: str = Header(default="")
):
    authorize(authorization)
    checkpoint = trained_model_path()

    try:
        train_path = safe_file(
            "data", request.train_dataset, ".csv"
        )
        val_path = safe_file(
            "data", request.val_dataset, ".csv"
        )

        model, blocks = load_model(checkpoint)

        if not blocks or blocks[-1]["out"] != 1:
            raise ValueError(
                "Only single-output regression is supported"
            )

        feature_count = blocks[0]["in"]

        train_x, train_y, train_columns = load_csv(
            train_path, feature_count
        )

        val_x, val_y, val_columns = load_csv(
            val_path, feature_count
        )

        if train_columns != val_columns:
            raise ValueError(
                "Training and validation columns differ"
            )

        train_metrics = evaluate(
            model, train_x, train_y
        )

        val_metrics = evaluate(
            model, val_x, val_y
        )

    except (ValueError, RuntimeError, KeyError) as error:
        raise HTTPException(422, str(error))

    result = {
        "model": checkpoint.name,
        "train_dataset": train_path.name,
        "validation_dataset": val_path.name,
        "training": train_metrics,
        "validation": val_metrics,
        "feature_columns": train_columns,
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT_PATH.with_suffix(".json.part")

    temporary.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )
    temporary.replace(REPORT_PATH)

    return result


@router.get("/api/studio/download/model")
def download_model(
    authorization: str = Header(default="")
):
    authorize(authorization)
    path = trained_model_path()

    return FileResponse(
        str(path),
        filename="studio_merged_model.pt",
        media_type="application/octet-stream",
        headers={"Cache-Control": "no-store"}
    )


@router.get("/api/studio/download/report")
def download_report(
    authorization: str = Header(default="")
):
    authorize(authorization)
    model_path = trained_model_path()

    if not REPORT_PATH.is_file():
        raise HTTPException(404, "Run evaluation first")

    report = json.loads(
        REPORT_PATH.read_text(encoding="utf-8")
    )

    if report.get("checkpoint_mtime_ns") != (
        model_path.stat().st_mtime_ns
    ):
        raise HTTPException(
            409, "Evaluation report belongs to an older model"
        )

    return FileResponse(
        str(REPORT_PATH),
        filename="studio_evaluation_report.json",
        media_type="application/json",
        headers={"Cache-Control": "no-store"}
    )
