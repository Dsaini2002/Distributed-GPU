import hmac
import os

from pathlib import Path

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from .history_store import (
    list_jobs, load_job, ARCHIVE_DIR
)

router = APIRouter()


def authorize(value):
    token = os.environ.get("DGPU_TOKEN", "")

    if not token or not hmac.compare_digest(
        value, "Bearer " + token
    ):
        raise HTTPException(401, "Invalid token")


@router.get("/api/studio/jobs")
def jobs(authorization: str = Header(default="")):
    authorize(authorization)

    records = []

    for record in list_jobs():
        records.append({
            "job_id": record.get("job_id"),
            "status": record.get("status"),
            "started_at": record.get("started_at"),
            "config": record.get("config", {}),
            "epoch": record.get("epoch", 0),
            "loss": record.get("loss"),
            "archived": bool(record.get("archived_output"))
        })

    return {"jobs": records}


@router.get("/api/studio/jobs/{job_id}")
def job_detail(
    job_id: str,
    authorization: str = Header(default="")
):
    authorize(authorization)

    try:
        record = load_job(job_id)
    except ValueError:
        raise HTTPException(400, "Invalid job ID")

    if record is None:
        raise HTTPException(404, "Job not found")

    return record


@router.get("/api/studio/jobs/{job_id}/model")
def download_job_model(
    job_id: str,
    authorization: str = Header(default="")
):
    authorize(authorization)

    try:
        record = load_job(job_id)
    except ValueError:
        raise HTTPException(400, "Invalid job ID")

    if not record:
        raise HTTPException(404, "Job not found")

    expected = ARCHIVE_DIR / (job_id + ".pt")
    saved = record.get("archived_output")

    if not saved or Path(saved).resolve() != expected.resolve():
        raise HTTPException(404, "Archived model unavailable")

    if not expected.is_file():
        raise HTTPException(404, "Checkpoint missing")

    return FileResponse(
        str(expected),
        filename=f"dgpu_{job_id}.pt",
        media_type="application/octet-stream"
    )


@router.get("/history", response_class=HTMLResponse)
def history_page():
    return """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Distributed-GPU Job History</title>
<style>
body {
  margin:0; padding:30px; background:#101827;
  color:#edf4ff; font-family:Arial,sans-serif;
}
main {max-width:980px;margin:auto}
h1 {color:#77baff}
a {color:#77baff;margin-right:14px}
.card {
  background:#1c2940; border-radius:12px;
  padding:20px; margin:14px 0;
}
button {
  background:#318ce7; color:white;
  border:0; padding:11px 15px;
  border-radius:8px; cursor:pointer; margin:5px;
}
input {
  padding:10px; border-radius:7px; border:0;
  min-width:260px;
}
pre {
  white-space:pre-wrap; overflow-wrap:anywhere;
  background:#101827; padding:14px;
}
svg {width:100%;height:240px;background:#101827}
small {color:#afc0d6}
</style>
</head>
<body>
<main>
<a href="/">Cluster</a>
<a href="/train">Training</a>
<a href="/monitor">Live Monitor</a>

<h1>Distributed-GPU Training History</h1>

<div class="card">
  <label>Cluster token</label>
  <input id="token" type="password" autocomplete="off">
  <button onclick="loadJobs()">Load History</button>
  <p><small>Token stays in this page only.</small></p>
</div>

<div class="card">
  <h2>Training Jobs</h2>
  <div id="jobs">Enter token and load history.</div>
</div>

<div class="card">
  <h2 id="title">Select a Job</h2>
  <div id="details"></div>
  <svg id="chart" viewBox="0 0 800 240"
       preserveAspectRatio="none"></svg>
  <button id="download" onclick="downloadModel()" disabled>
    Download Archived Model
  </button>
  <h3>Training Logs</h3>
  <pre id="logs">No job selected.</pre>
</div>
</main>

<script>
let activeJob = null;

function headers() {
  return {
    "Authorization":
      "Bearer "+document.getElementById("token").value
  };
}

async function api(path) {
  const response = await fetch(path,{headers:headers()});
  const data = await response.json();

  if (!response.ok)
    throw new Error(JSON.stringify(data));

  return data;
}

async function loadJobs() {
  const root = document.getElementById("jobs");
  root.textContent = "Loading...";

  try {
    const data = await api("/api/studio/jobs");
    root.replaceChildren();

    if (!data.jobs.length)
      root.textContent = "No saved training jobs yet.";

    for (const job of data.jobs) {
      const div = document.createElement("div");
      div.className = "card";

      const summary = document.createElement("p");
      const config = job.config || {};

      summary.textContent =
        "Job " + job.job_id +
        " | " + job.status +
        " | " + (config.model || "Unknown model") +
        " | Epoch " + (job.epoch || 0) +
        " | Loss " + (job.loss ?? "-");

      const button = document.createElement("button");
      button.textContent = "View Job";
      button.onclick = () => showJob(job.job_id);

      div.append(summary, button);
      root.appendChild(div);
    }

  } catch(error) {
    root.textContent = String(error);
  }
}

function draw(history) {
  const svg = document.getElementById("chart");
  const ns = "http://www.w3.org/2000/svg";
  svg.replaceChildren();

  if (!history || !history.length)
    return;

  const logValues = history.map(p =>
    Math.log10(Math.max(Number(p.mse),1e-12))
  );

  const low = Math.min(...logValues) - 0.2;
  const high = Math.max(...logValues) + 0.2;
  const range = Math.max(0.1,high-low);

  const points = logValues.map((v,i) => {
    const x = logValues.length === 1
      ? 400 : 45 + i*715/(logValues.length-1);

    const y = 205-(v-low)/range*175;
    return x+","+y;
  });

  const line = document.createElementNS(ns,"polyline");
  line.setAttribute("points",points.join(" "));
  line.setAttribute("fill","none");
  line.setAttribute("stroke","#68dfa9");
  line.setAttribute("stroke-width","3");

  svg.appendChild(line);
}

async function showJob(id) {
  try {
    const job = await api("/api/studio/jobs/"+id);
    activeJob = job.job_id;

    document.getElementById("title").textContent =
      "Job: "+job.job_id;

    const details = document.getElementById("details");
    details.textContent =
      "Status: "+job.status+
      " | Model: "+(job.config?.model || "-")+
      " | Dataset: "+(job.config?.dataset || "-")+
      " | Epochs completed: "+(job.epoch || 0)+
      " | Final loss: "+(job.loss ?? "-");

    document.getElementById("logs").textContent =
      (job.logs || []).join("\\n");

    document.getElementById("download").disabled =
      !job.archived_output;

    draw(job.history || []);

  } catch(error) {
    alert(String(error));
  }
}

async function downloadModel() {
  if (!activeJob) return;

  const response = await fetch(
    "/api/studio/jobs/"+activeJob+"/model",
    {headers:headers()}
  );

  if (!response.ok) {
    alert(await response.text());
    return;
  }

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");

  a.href = url;
  a.download = "dgpu_"+activeJob+".pt";
  document.body.appendChild(a);
  a.click();
  a.remove();

  setTimeout(()=>URL.revokeObjectURL(url),5000);
}
</script>
</body>
</html>
"""
