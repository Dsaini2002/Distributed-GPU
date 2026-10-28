import io
import os
import hmac
import threading

import torch
import torch.nn as nn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import Response

app = FastAPI(title="DGPU GPU Worker")
stages = {}
lock = threading.RLock()


def authorize(header):
    token = os.environ.get("DGPU_TOKEN", "")
    if not token or not hmac.compare_digest(
        header, "Bearer " + token
    ):
        raise HTTPException(401, "Unauthorized")


def build_model(blocks):
    layers = []
    for block in blocks:
        layers.append(nn.Linear(block["in"], block["out"]))
        if block["relu"]:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


@app.get("/health")
def health():
    return {"ok": True, "cuda": torch.cuda.is_available()}


@app.post("/configure")
def configure(p: dict, authorization: str = Header(default="")):
    authorize(authorization)

    with lock:
        if not torch.cuda.is_available():
            raise HTTPException(409, "CUDA unavailable")

        sid = p["stage_id"]
        gpu = int(p["gpu_index"])
        device = torch.device(f"cuda:{gpu}")

        if gpu < 0 or gpu >= torch.cuda.device_count():
            raise HTTPException(400, "Invalid GPU index")

        if sid in stages:
            raise HTTPException(409, "Stage already configured")

        model = build_model(p["blocks"]).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(p["lr"]),
            foreach=False
        )

        stages[sid] = {
            "model": model,
            "optimizer": optimizer,
            "device": device,
            "x": None,
            "y": None
        }

    return {"ok": True, "stage_id": sid}


@app.post("/forward")
def forward(p: dict, authorization: str = Header(default="")):
    authorize(authorization)

    with lock:
        s = stages[p["stage_id"]]

        if s["y"] is not None:
            raise HTTPException(409, "Previous backward incomplete")

        s["optimizer"].zero_grad(set_to_none=True)

        x = torch.tensor(
            p["x"],
            dtype=torch.float32,
            device=s["device"],
            requires_grad=True
        )

        y = s["model"](x)
        s["x"], s["y"] = x, y

        return {"y": y.detach().cpu().tolist()}


@app.post("/backward")
def backward(p: dict, authorization: str = Header(default="")):
    authorize(authorization)

    with lock:
        s = stages[p["stage_id"]]

        if s["y"] is None:
            raise HTTPException(409, "Forward required")

        gradient = torch.tensor(
            p["grad"],
            dtype=torch.float32,
            device=s["device"]
        )

        if gradient.shape != s["y"].shape:
            raise HTTPException(400, "Gradient shape mismatch")

        s["y"].backward(gradient)

        input_gradient = s["x"].grad.detach().cpu().tolist()

        s["optimizer"].step()
        s["x"] = None
        s["y"] = None

        return {"grad": input_gradient}


@app.post("/weights")
def weights(p: dict, authorization: str = Header(default="")):
    authorize(authorization)

    with lock:
        s = stages[p["stage_id"]]

        if s["y"] is not None:
            raise HTTPException(409, "Backward not complete")

        buffer = io.BytesIO()
        torch.save(s["model"].state_dict(), buffer)

        return Response(
            content=buffer.getvalue(),
            media_type="application/octet-stream"
        )


@app.post("/reset")
def reset(p: dict, authorization: str = Header(default="")):
    authorize(authorization)

    with lock:
        stages.pop(p["stage_id"], None)

    return {"ok": True}


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
