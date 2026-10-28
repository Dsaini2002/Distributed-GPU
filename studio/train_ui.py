import os
import hmac

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .train_engine import (
    list_options, get_status, start_training
)

router = APIRouter()


class TrainingRequest(BaseModel):
    model: str
    dataset: str
    epochs: int = Field(default=10, ge=1, le=100)
    batch_size: int = Field(default=32, ge=1, le=128)
    lr: float = Field(default=0.001, gt=0, le=0.1)


@router.get("/api/studio/options")
def options():
    return list_options()


@router.get("/api/studio/job")
def job():
    return get_status()


@router.post("/api/studio/start")
def start(
    config: TrainingRequest,
    authorization: str = Header(default="")
):
    token = os.environ.get("DGPU_TOKEN", "")

    if not token or not hmac.compare_digest(
        authorization, "Bearer " + token
    ):
        raise HTTPException(401, "Invalid token")

    try:
        return start_training(config.model_dump())
    except ValueError as error:
        raise HTTPException(409, str(error))


@router.get("/train", response_class=HTMLResponse)
def page():
    return """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Distributed-GPU Studio | Training</title>
<style>
body {
  margin:0; padding:35px;
  font-family:Arial,sans-serif;
  background:#101827; color:#edf4ff;
}
main {max-width:800px;margin:auto}
h1 {color:#77baff}
.card {
  background:#1c2940; padding:20px;
  border-radius:12px; margin:12px 0;
}
label {display:block;margin-top:13px}
input,select {
  display:block; padding:11px; width:100%;
  box-sizing:border-box; margin-top:5px;
  border-radius:7px; border:1px solid #64748b;
}
button {
  margin-top:20px; padding:13px 20px;
  background:#318ce7; color:white;
  border:0; border-radius:8px; cursor:pointer;
}
pre {
  white-space:pre-wrap; overflow-wrap:anywhere;
  background:#0b1320; padding:15px; border-radius:8px;
}
small {color:#a9bdd4}
</style>
</head>
<body>
<main>
<a href="/" style="color:#7dbaff">Cluster Dashboard</a>
<h1>Distributed-GPU Training Studio</h1><p><a href="/upload">Upload Model or Dataset</a></p>
<p>Automatic N-GPU model-parallel regression training</p>

<div class="card">
  <label>Model
    <select id="model"></select>
  </label>
  <label>Dataset
    <select id="dataset"></select>
  </label>
  <label>Epochs
    <input id="epochs" type="number" value="15" min="1">
  </label>
  <label>Batch Size
    <input id="batch" type="number" value="32" min="1">
  </label>
  <label>Learning Rate
    <input id="lr" type="number" value="0.001" step="0.001">
  </label>
  <label>Cluster authentication token
    <input id="token" type="password" autocomplete="off">
  </label>
  <button onclick="startTraining()">Start Training</button>
  <p><small>Supports trusted nn.Sequential Linear/ReLU
  regression models only.</small></p>
</div>

<div class="card">
  <h2>Training Status</h2>
  <div id="status">Idle</div>
  <pre id="logs">No training started.</pre>
</div>
</main>

<script>
async function init() {
  const response = await fetch("/api/studio/options");
  const data = await response.json();

  for (const [id, values] of [
    ["model", data.models],
    ["dataset", data.datasets]
  ]) {
    const selector = document.getElementById(id);
    for (const value of values) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      selector.appendChild(option);
    }
  }

  const model = document.getElementById("model");
  const dataset = document.getElementById("dataset");

  if ([...model.options].some(x => x.value === "my_model.py"))
    model.value = "my_model.py";

  if ([...dataset.options].some(x => x.value === "demo_train.csv"))
    dataset.value = "demo_train.csv";
}

async function startTraining() {
  const payload = {
    model: document.getElementById("model").value,
    dataset: document.getElementById("dataset").value,
    epochs: Number(document.getElementById("epochs").value),
    batch_size: Number(document.getElementById("batch").value),
    lr: Number(document.getElementById("lr").value)
  };

  const token = document.getElementById("token").value;

  try {
    const response = await fetch("/api/studio/start", {
      method:"POST",
      headers:{
        "Content-Type":"application/json",
        "Authorization":"Bearer " + token
      },
      body:JSON.stringify(payload)
    });

    const data = await response.json();

    if (!response.ok) {
      alert(JSON.stringify(data));
    } else {
      refresh();
    }
  } catch (error) {
    alert(String(error));
  }
}

async function refresh() {
  try {
    const response = await fetch("/api/studio/job");
    const data = await response.json();

    document.getElementById("status").textContent =
      "Status: " + data.status +
      " | Epoch: " + (data.epoch || 0) +
      " | Loss: " + (data.loss ?? "-");

    document.getElementById("logs").textContent =
      (data.logs || []).join("\\n") || "No logs.";
  } catch (error) {
    document.getElementById("status").textContent =
      "Controller disconnected";
  }
}

init();
refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""

