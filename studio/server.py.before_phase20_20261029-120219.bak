import os
import time
import hmac
import threading

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .planner import create_plan

TOKEN = os.environ.get("DGPU_TOKEN")
if not TOKEN:
    raise RuntimeError("DGPU_TOKEN environment variable required")

app = FastAPI(title="Distributed-GPU Studio")
registry = {}
lock = threading.Lock()
TIMEOUT = 20


class GPU(BaseModel):
    index: int = Field(ge=0)
    name: str
    free_mib: int = Field(ge=0)
    total_mib: int = Field(ge=0)


class Node(BaseModel):
    node_id: str = Field(min_length=1, max_length=100)
    hostname: str = Field(min_length=1, max_length=100)
    gpus: list[GPU]


@app.post("/api/heartbeat")
def heartbeat(
    request: Request,
    node: Node,
    authorization: str = Header(default="")
):
    if not hmac.compare_digest(
        authorization, "Bearer " + TOKEN
    ):
        raise HTTPException(401, "Invalid token")

    with lock:
        registry[node.node_id] = {
            "address": request.client.host,
            **node.model_dump(),
            "last_seen": time.time()
        }

    return {"status": "ok"}


def snapshot():
    now = time.time()

    with lock:
        nodes = [dict(n) for n in registry.values()]

    for node in nodes:
        node["online"] = now - node["last_seen"] < TIMEOUT

    return nodes


@app.get("/api/nodes")
def nodes():
    all_nodes = snapshot()
    online = [n for n in all_nodes if n["online"]]

    return {
        "online_computers": len(online),
        "online_gpus": sum(len(n["gpus"]) for n in online),
        "nodes": all_nodes
    }


@app.get("/api/plan")
def plan(
    blocks: int = Query(16, ge=1, le=10000),
    block_mib: int = Query(256, ge=1, le=4096)
):
    online = [n for n in snapshot() if n["online"]]

    try:
        return create_plan(online, blocks, block_mib)
    except ValueError as error:
        raise HTTPException(409, str(error))


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Distributed-GPU Studio</title>
<style>
body {
  background:#101827; color:#eff6ff;
  font-family:Arial,sans-serif; margin:0; padding:32px;
}
main {max-width:950px; margin:auto}
h1 {color:#77baff}
.card {
  background:#1c2940; padding:18px;
  border-radius:12px; margin:12px 0;
}
.status {color:#61dfaa}
input {
  padding:9px; width:100px; border-radius:6px;
  border:1px solid #64748b;
}
button {
  padding:10px 16px; background:#3995ee;
  color:white; border:0; border-radius:7px; cursor:pointer;
}
small {color:#a9bdd4}
</style>
</head>
<body>
<main>
<h1>Distributed-GPU Studio</h1>
<p>N-Computer Cluster Dashboard</p>

<div class="card">
  <h2 id="summary">Connecting...</h2>
  <small>Automatically updates every 3 seconds</small>
</div>

<h2>Connected Computers</h2>
<div id="nodes"></div>

<h2>GPU Allocation Preview</h2>
<div class="card">
  <label>Model blocks: <input id="blocks" type="number"
      value="16" min="1"></label>
  <label>MiB/block: <input id="cost" type="number"
      value="256" min="1"></label>
  <button onclick="refresh()">Calculate</button>
  <p><small>Estimated memory allocation only.
  Training is not enabled yet.</small></p>
</div>
<div id="plan"></div>

<h2>Model Training</h2>
<div class="card">
  <p>Model upload and one-click training:
  planned for the next development stage.</p>
  <button disabled>Train Model (Coming Next)</button>
</div>
</main>
<script>
function card(text) {
  const element = document.createElement("div");
  element.className = "card";
  element.textContent = text;
  return element;
}

async function refresh() {
  try {
    const response = await fetch("/api/nodes");
    const data = await response.json();

    document.getElementById("summary").textContent =
      data.online_computers + " computers | " +
      data.online_gpus + " GPUs online";

    const root = document.getElementById("nodes");
    root.replaceChildren();

    for (const n of data.nodes) {
      const gpuInfo = n.gpus.map(g =>
        g.name + " (" + g.free_mib +
        "/" + g.total_mib + " MiB free/total)"
      ).join(" | ");

      root.appendChild(card(
        n.hostname + " - " +
        (n.online ? "ONLINE" : "OFFLINE") +
        " - " + (gpuInfo || "No CUDA GPU")
      ));
    }

    const blocks = Number(
      document.getElementById("blocks").value
    );
    const cost = Number(
      document.getElementById("cost").value
    );

    const planResponse = await fetch(
      "/api/plan?blocks=" + blocks +
      "&block_mib=" + cost
    );

    const plan = await planResponse.json();
    const output = document.getElementById("plan");
    output.replaceChildren();

    if (!planResponse.ok) {
      output.appendChild(card(plan.detail));
      return;
    }

    for (const stage of plan.stages) {
      output.appendChild(card(
        stage.computer + " | " + stage.gpu +
        " | Blocks " + stage.start_block +
        "-" + stage.end_block +
        " | " + stage.blocks + " blocks" +
        " | ~" + stage.estimated_mib + " MiB"
      ));
    }
  } catch (error) {
    document.getElementById("summary").textContent =
      "Controller connection error";
  }
}
refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""



from .train_ui import router as studio_training_router
app.include_router(studio_training_router)



from .upload_ui import router as studio_upload_router
app.include_router(studio_upload_router)

from .monitor_api import router as studio_monitor_api_router
from .monitor_ui import router as studio_monitor_ui_router

app.include_router(studio_monitor_api_router)
app.include_router(studio_monitor_ui_router)


from .history_ui import router as studio_history_router
app.include_router(studio_history_router)


from .resume_api import router as studio_resume_router
app.include_router(studio_resume_router)


from .session_auth import install as install_session_auth
install_session_auth(app)

# DGPU_MODEL_ZOO_BEGIN
from studio_zoo.api import router as _dgpu_zoo_router
app.include_router(_dgpu_zoo_router)
# DGPU_MODEL_ZOO_END

# DGPU_PHASE19_BEGIN
from studio_multi.api import router as _dgpu_phase19_router
app.include_router(_dgpu_phase19_router)
# DGPU_PHASE19_END
