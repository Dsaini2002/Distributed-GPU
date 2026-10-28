from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/monitor", response_class=HTMLResponse)
def monitor_page():
    return """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Distributed-GPU Live Monitor</title>
<style>
* {box-sizing:border-box}
body {
  margin:0; background:#101827; color:#eff6ff;
  font:15px Arial,sans-serif; padding:30px;
}
main {max-width:1050px;margin:auto}
h1 {color:#79baff}
a {color:#79baff;margin-right:15px}
.card {
  background:#1c2940; border-radius:14px;
  padding:21px; margin:14px 0;
}
.grid {display:grid;grid-template-columns:1fr 1fr;gap:15px}
.value {font-size:28px;font-weight:bold;color:#69e1aa}
small {color:#b2c3d8}
select,input {
  background:#eef5ff; border:0; border-radius:6px;
  padding:10px; margin:7px 8px 7px 0;
  max-width:100%;
}
button {
  background:#318ce7; color:white;
  border:0;border-radius:7px;padding:12px;
  cursor:pointer;margin:6px 8px 6px 0;
}
button:hover {background:#1d73c1}
pre {
  background:#0b1320;padding:15px;border-radius:8px;
  overflow-wrap:anywhere;white-space:pre-wrap;
  max-height:290px;overflow-y:auto;
}
progress {width:100%;height:19px}
#plot {
  background:#111f32;border-radius:10px;width:100%;
  height:260px;
}
@media(max-width:650px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<main>
<a href="/">Cluster</a>
<a href="/train">Training</a><a href="/history">Job History</a>
<a href="/upload">Upload</a>

<h1>Distributed-GPU Live Monitor</h1>

<div class="grid">
  <div class="card">
    <small>Training status</small>
    <div id="status" class="value">Loading...</div>
    <p id="epoch">Epoch: -</p>
    <progress id="progress" max="100" value="0"></progress>
  </div>
  <div class="card">
    <small>Latest training MSE</small>
    <div id="loss" class="value">-</div>
    <p id="cluster">Checking GPUs...</p>
  </div>
</div>

<div class="card">
  <h2>Live Training Loss</h2>
  <small>Logarithmic scale — updates every 2 seconds</small>
  <svg id="plot" viewBox="0 0 800 260"
       preserveAspectRatio="none"></svg>
  <p id="graphInfo">No epoch data yet.</p>
</div>

<div class="card">
  <h2>Connected GPU Resources</h2>
  <div id="gpus">Loading...</div>
</div>

<div class="card">
  <h2>Automatic Stage Allocation</h2>
  <div id="stages">No job running.</div>
</div>

<div class="card">
  <h2>Evaluate Trained Model</h2>
  <p><small>Select training and validation CSV files.</small></p>

  <label>Training dataset</label>
  <select id="trainDataset"></select>

  <label>Validation dataset</label>
  <select id="valDataset"></select>

  <p>
    <label>Authenticated with your login session</label>
    <input id="token" type="hidden" value="">
  </p>

  <button onclick="runEvaluation()">Evaluate Model</button>
  <button onclick="download('model')">Download Model (.pt)</button>
  <button onclick="download('report')">Download Report (.json)</button>

  <pre id="evaluation">Evaluation not run yet.</pre>
</div>

<div class="card">
  <h2>Training Logs</h2>
  <pre id="logs">No logs.</pre>
</div>
</main>

<script>
const ns = "http://www.w3.org/2000/svg";

function svgNode(tag, attrs) {
  const el = document.createElementNS(ns, tag);
  for (const [k,v] of Object.entries(attrs))
    el.setAttribute(k, String(v));
  return el;
}

function drawGraph(history) {
  const svg = document.getElementById("plot");
  svg.replaceChildren();

  const left=52, right=775, top=20, bottom=220;

  svg.append(
    svgNode("line",{
      x1:left,y1:top,x2:left,y2:bottom,
      stroke:"#7890ab", "stroke-width":1.5
    }),
    svgNode("line",{
      x1:left,y1:bottom,x2:right,y2:bottom,
      stroke:"#7890ab", "stroke-width":1.5
    })
  );

  if (!history.length) return;

  const values = history.map(p =>
    Math.log10(Math.max(p.mse, 1e-12))
  );

  const min = Math.min(...values) - 0.2;
  const max = Math.max(...values) + 0.2;
  const span = Math.max(max-min, 0.1);

  const points = history.map((p,i) => {
    const x = history.length === 1
      ? (left + right)/2
      : left + i*(right-left)/(history.length-1);

    const y = bottom-(values[i]-min)/span*(bottom-top);
    return [x,y];
  });

  const polyline = svgNode("polyline",{
    points:points.map(p=>p.join(",")).join(" "),
    fill:"none",
    stroke:"#5ee3ad",
    "stroke-width":3,
    "stroke-linecap":"round",
    "stroke-linejoin":"round"
  });

  svg.appendChild(polyline);

  for (const [x,y] of points) {
    svg.appendChild(svgNode("circle",{
      cx:x,cy:y,r:3.4,fill:"#9ceccc"
    }));
  }

  document.getElementById("graphInfo").textContent =
    history.length + " epochs | First MSE: " +
    history[0].mse + " | Latest MSE: " +
    history[history.length-1].mse;
}

function makeLine(root,text) {
  const p=document.createElement("p");
  p.textContent=text;
  root.appendChild(p);
}

async function refresh() {
  try {
    const [jobResponse,nodesResponse] = await Promise.all([
      fetch("/api/studio/monitor"),
      fetch("/api/nodes")
    ]);

    if (!jobResponse.ok || !nodesResponse.ok)
      throw new Error("Server request failed");

    const job=await jobResponse.json();
    const cluster=await nodesResponse.json();
    const history=job.history || [];
    const last=history[history.length-1];

    document.getElementById("status").textContent =
      job.status;

    document.getElementById("loss").textContent =
      job.loss == null ? "-" : job.loss;

    const current=last ? last.epoch : (job.epoch||0);
    const total=last ? last.total_epochs : 0;

    document.getElementById("epoch").textContent =
      "Epoch: "+current+(total?" / "+total:"");

    document.getElementById("progress").value =
      total ? Math.min(100,current/total*100) : 0;

    document.getElementById("cluster").textContent =
      cluster.online_computers+" computers | "+
      cluster.online_gpus+" GPUs online";

    const gpuRoot=document.getElementById("gpus");
    gpuRoot.replaceChildren();

    for (const node of cluster.nodes) {
      if (!node.online) continue;

      for (const gpu of node.gpus) {
        makeLine(gpuRoot,node.hostname+" | "+
          gpu.name+" | Free VRAM: "+
          gpu.free_mib+" / "+gpu.total_mib+" MiB");
      }
    }

    const stageRoot=document.getElementById("stages");
    stageRoot.replaceChildren();

    for (const s of (job.stages||[])) {
      makeLine(stageRoot,s.computer+" | "+s.gpu+
        " | "+s.layers+" Linear layers"+
        " | estimated "+s.estimated_mib+" MiB");
    }

    document.getElementById("logs").textContent =
      (job.logs||[]).join("\\n") || "No logs.";

    drawGraph(history);

  } catch (error) {
    document.getElementById("status").textContent =
      "Connection error";
  }
}

async function setup() {
  const res=await fetch("/api/studio/options");
  const options=await res.json();

  for (const id of ["trainDataset","valDataset"]) {
    const root=document.getElementById(id);

    for (const name of options.datasets) {
      const option=document.createElement("option");
      option.value=name;
      option.textContent=name;
      root.appendChild(option);
    }
  }

  if (options.datasets.includes("demo_train.csv"))
    document.getElementById("trainDataset").value =
      "demo_train.csv";

  if (options.datasets.includes("demo_val.csv"))
    document.getElementById("valDataset").value =
      "demo_val.csv";
}

function authHeader() {
  return {
    "Authorization":
      "Bearer "+document.getElementById("token").value
  };
}

async function runEvaluation() {
  const result=document.getElementById("evaluation");
  result.textContent="Evaluating model...";

  try {
    const response=await fetch("/api/studio/evaluate",{
      method:"POST",
      headers:{
        ...authHeader(),
        "Content-Type":"application/json"
      },
      body:JSON.stringify({
        train_dataset:
          document.getElementById("trainDataset").value,
        val_dataset:
          document.getElementById("valDataset").value
      })
    });

    const data=await response.json();

    result.textContent=response.ok
      ? JSON.stringify({
          training:data.training,
          validation:data.validation
        },null,2)
      : JSON.stringify(data);

  } catch (error) {
    result.textContent=String(error);
  }
}

async function download(kind) {
  try {
    const response=await fetch(
      "/api/studio/download/"+kind,
      {headers:authHeader()}
    );

    if (!response.ok) {
      alert(await response.text());
      return;
    }

    const blob=await response.blob();
    const url=URL.createObjectURL(blob);
    const a=document.createElement("a");

    a.href=url;
    a.download=kind==="model"
      ?"studio_merged_model.pt"
      :"studio_evaluation_report.json";

    document.body.appendChild(a);
    a.click();
    a.remove();

    setTimeout(()=>URL.revokeObjectURL(url),5000);

  } catch(error) {
    alert(String(error));
  }
}

setup().catch(error => {
  document.getElementById("evaluation").textContent =
    String(error);
});

refresh();
setInterval(refresh,2000);
</script>
</body>
</html>
"""
