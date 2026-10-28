from pathlib import Path


def patch(text, old, new, label):
    if old not in text:
        raise RuntimeError(
            f"Patch failed: {label}. "
            "Source code differs from expected version."
        )

    return text.replace(old, new, 1)


# Extend GPU worker with snapshot/restore
path = Path("studio/worker_api.py")
source = path.read_text(encoding="utf-8-sig")

if '"/snapshot"' not in source:
    source = patch(
        source,
        "FastAPI, Header, HTTPException",
        "FastAPI, Header, HTTPException, Request",
        "worker imports"
    )

    source += '''

@app.post("/snapshot")
def snapshot_stage(
    p: dict,
    authorization: str = Header(default="")
):
    authorize(authorization)

    with lock:
        stage = stages.get(p.get("stage_id"))

        if stage is None:
            raise HTTPException(404, "Stage missing")

        if stage["y"] is not None:
            raise HTTPException(409, "Backward incomplete")

        buffer = io.BytesIO()

        torch.save({
            "model": stage["model"].state_dict(),
            "optimizer": stage["optimizer"].state_dict()
        }, buffer)

        return Response(
            content=buffer.getvalue(),
            media_type="application/octet-stream"
        )


@app.post("/restore")
async def restore_stage(
    request: Request,
    x_stage_id: str = Header(default=""),
    authorization: str = Header(default="")
):
    authorize(authorization)

    raw = await request.body()

    if not raw:
        raise HTTPException(400, "Empty snapshot")

    with lock:
        stage = stages.get(x_stage_id)

        if stage is None:
            raise HTTPException(404, "Stage missing")

        if stage["y"] is not None:
            raise HTTPException(409, "Stage is busy")

        data = torch.load(
            io.BytesIO(raw),
            map_location="cpu",
            weights_only=True
        )

        stage["model"].load_state_dict(
            data["model"], strict=True
        )

        stage["optimizer"].load_state_dict(
            data["optimizer"]
        )

        return {"ok": True}


@app.post("/clear")
def clear_stages(
    authorization: str = Header(default="")
):
    authorize(authorization)

    with lock:
        stages.clear()

    return {"ok": True}
'''

    path.write_text(source, encoding="utf-8")
    print("Worker snapshot and restore endpoints added")


# Extend controller training engine
path = Path("studio/train_engine.py")
source = path.read_text(encoding="utf-8-sig")

if "def resume_training(" not in source:

    source = patch(
        source,
        "from .history_store import (",
        """from .recovery import (
    save_epoch, latest_checkpoint, restore_stage
)
from .history_store import (
    load_job,""",
        "recovery imports"
    )

    source = patch(
        source,
        "def train(config):",
        "def train(config, resume_state=None):",
        "training signature"
    )

    old = '''stages = plan_model(
            blocks,
            snapshot(),
            config["batch_size"]
        )'''

    new = '''if resume_state is None:
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
                cleared.add(stage["address"])'''

    source = patch(
        source, old, new, "resume partition"
    )

    source = patch(
        source,
        '''        for stage in stages:
            request_worker(stage, "configure", {''',
        '''        for index, stage in enumerate(stages):
            request_worker(stage, "configure", {''',
        "stage configuration loop"
    )

    source = patch(
        source,
        '''            configured.append(stage)

            log(''',
        '''            configured.append(stage)

            if resume_state is not None:
                restore_stage(
                    stage,
                    resume_state,
                    index,
                    os.environ["DGPU_TOKEN"]
                )

            log(''',
        "optimizer restore"
    )

    source = patch(
        source,
        '''for epoch in range(config["epochs"]):''',
        '''for epoch in range(first_epoch, config["epochs"]):''',
        "epoch continuation"
    )

    source = patch(
        source,
        '''            log(
                f"Epoch {epoch+1}/{config['epochs']} "
                f"| MSE: {average:.6f}"
            )''',
        '''            log(
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
            )''',
        "epoch snapshot"
    )

    source += '''


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
'''

    path.write_text(source, encoding="utf-8")
    print("Controller resume logic added")


# Register resume API
path = Path("studio/server.py")
source = path.read_text(encoding="utf-8-sig")

if "studio_resume_router" not in source:
    source += """

from .resume_api import router as studio_resume_router
app.include_router(studio_resume_router)
"""

    path.write_text(source, encoding="utf-8")


# Add resume button to job history
path = Path("studio/history_ui.py")
source = path.read_text(encoding="utf-8-sig")

if "resumeSelectedJob" not in source:
    source = patch(
        source,
        '<h3>Training Logs</h3>',
        '''<button onclick="resumeSelectedJob()">
    Resume Interrupted Job
  </button>
  <h3>Training Logs</h3>''',
        "resume button"
    )

    source = patch(
        source,
        "</script>",
        '''
async function resumeSelectedJob() {
  if (!activeJob) {
    alert("Select a job first");
    return;
  }

  const response = await fetch(
    "/api/studio/jobs/" + activeJob + "/resume",
    {
      method: "POST",
      headers: headers()
    }
  );

  const result = await response.json();

  if (!response.ok) {
    alert(JSON.stringify(result));
    return;
  }

  alert(
    "Resuming from epoch " +
    result.resumed_from_epoch
  );

  location.href = "/monitor";
}
</script>''',
        "resume UI function"
    )

    path.write_text(source, encoding="utf-8")


print("PHASE 15 RECOVERY PATCH APPLIED")
