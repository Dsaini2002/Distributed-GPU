"""Phase19 controller page and authenticated job API (separate from Phase18)."""
import hmac
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from studio_zoo.models import CATALOG
from .coordinator import worker_url
from .core import SUPPORTED

router = APIRouter(tags=['phase19'])
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / '.dgpu' / 'phase19_runs'
_LOCK = threading.Lock()
_CURRENT = {'id': None, 'process': None}


def _auth(authorization: str = Header(default='')):
    token = os.environ.get('DGPU_TOKEN', '')
    given = authorization[7:] if authorization.startswith('Bearer ') else ''
    if not token:
        raise HTTPException(503, 'DGPU_TOKEN missing on controller')
    if not hmac.compare_digest(token.encode(), given.encode()):
        raise HTTPException(401, 'Cluster token required')


class Launch(BaseModel):
    model: str
    dataset: str
    workers: list[str] = Field(default_factory=lambda: ['192.168.50.2:45420'], min_length=1, max_length=24)
    epochs: int = Field(default=3, ge=1, le=100)
    batch_size: int = Field(default=4, ge=1, le=64)
    learning_rate: float = Field(default=0.001, gt=0, le=0.05)


def _save_json(path: Path, data: dict):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    tmp.replace(path)


@router.get('/api/distributed-zoo/catalog')
def catalog():
    return {'models': [{
        'key': CATALOG[key].key, 'family': CATALOG[key].family,
        'description': CATALOG[key].description,
        'suggested_batch': min(CATALOG[key].suggested_batch, 8),
    } for key in sorted(SUPPORTED)],
    'algorithm': 'local_sgd_weighted_average', 'requires': 'controller CUDA + each remote CUDA',
    'unsupported': ['distilgpt2_lora', 'yolo11n_detection']}


@router.post('/api/distributed-zoo/train', dependencies=[Depends(_auth)])
def start(spec: Launch):
    if spec.model not in SUPPORTED:
        raise HTTPException(400, 'Model not supported in distributed training')
    root = (ROOT / 'data').resolve()
    dataset = (ROOT / spec.dataset).resolve()
    if not dataset.is_relative_to(root) or dataset == root or not dataset.exists():
        raise HTTPException(400, 'Dataset must exist inside project data/')
    if dataset.is_symlink():
        raise HTTPException(400, 'Dataset symlinks forbidden')
    try:
        addresses = [worker_url(x) for x in spec.workers]
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if len(set(addresses)) != len(addresses):
        raise HTTPException(400, 'Duplicate workers')
    RUNS.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        active = _CURRENT['process']
        if active and active.poll() is None:
            raise HTTPException(409, 'A Phase19 training job is running')
        # Avoid colliding with local Model Zoo training when feasible.
        try:
            from studio_zoo.api import _CURRENT as local_job
            active_local = local_job.get('process')
            if active_local and active_local.poll() is None:
                raise HTTPException(409, 'Phase18 local Model Zoo training is already running')
        except ImportError:
            pass
        job_id = uuid.uuid4().hex[:16]
        run_dir = RUNS / job_id
        run_dir.mkdir()
        state = {'id': job_id, 'state': 'running', 'model': spec.model,
                 'dataset': spec.dataset, 'workers': addresses,
                 'algorithm': 'local_sgd_weighted_average',
                 'checkpoint': None, 'returncode': None}
        _save_json(run_dir / 'status.json', state)
        args = [sys.executable, '-u', '-m', 'studio_multi.coordinator',
                '--model', spec.model, '--dataset', str(dataset),
                '--workers', ';'.join(spec.workers), '--epochs', str(spec.epochs),
                '--batch-size', str(spec.batch_size), '--lr', str(spec.learning_rate),
                '--output', str(run_dir)]
        try:
            with (run_dir / 'train.log').open('w', encoding='utf-8') as logfile:
                process = subprocess.Popen(args, cwd=str(ROOT), stdout=logfile,
                                           stderr=subprocess.STDOUT,
                                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except Exception as exc:
            state.update(state='failed', error=type(exc).__name__)
            _save_json(run_dir / 'status.json', state)
            raise HTTPException(500, 'Cannot start trainer subprocess') from exc
        _CURRENT.update(id=job_id, process=process)

        def wait_for_end():
            code = process.wait()
            with _LOCK:
                current_state = json.loads((run_dir / 'status.json').read_text(encoding='utf-8'))
                current_state.update(state='completed' if code == 0 else 'failed',
                                     returncode=code,
                                     checkpoint=str(run_dir / 'model.pt') if code == 0 else None)
                _save_json(run_dir / 'status.json', current_state)
                if _CURRENT['id'] == job_id:
                    _CURRENT.update(id=None, process=None)
        threading.Thread(target=wait_for_end, daemon=True).start()
    return {'id': job_id, 'state': 'running'}


@router.get('/api/distributed-zoo/status/{job_id}', dependencies=[Depends(_auth)])
def status(job_id: str):
    if not re.fullmatch('[a-f0-9]{16}', job_id):
        raise HTTPException(404, 'Job not found')
    path = RUNS / job_id / 'status.json'
    if not path.exists():
        raise HTTPException(404, 'Job not found')
    saved = json.loads(path.read_text(encoding='utf-8'))
    log = RUNS / job_id / 'train.log'
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 7000))
            saved['log'] = stream.read().decode('utf-8', errors='replace')[-7000:]
    return saved


PAGE = '''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DGPU Phase 19 Distributed Model Zoo</title><style>
body{font:16px system-ui;background:#0f1929;color:#eaf2ff;max-width:930px;margin:32px auto;padding:0 18px}h1{color:#8ebaff}
a{color:#96c7ff}p{line-height:1.55} .box{background:#1c2b43;border-radius:12px;padding:22px;margin:18px 0}
label{display:block;margin-top:13px}input,select,button{font:inherit;box-sizing:border-box;padding:10px;border-radius:7px}
input,select{width:100%;background:#111c2c;color:#fff;border:1px solid #526785}button{background:#437fcb;color:white;border:0;margin-top:16px;cursor:pointer}button:disabled{opacity:.5}
pre{background:#0d1522;padding:12px;white-space:pre-wrap;max-height:370px;overflow:auto}small{color:#bccbe4}
</style></head><body><h1>Phase 19 · Distributed Model Zoo</h1>
<p><a href="/">Dashboard</a> · <a href="/model-zoo">Local Model Zoo</a> · <a href="/train">Legacy model-parallel trainer</a></p>
<p>Separate CUDA GPU on Laptop A and each remote worker. <b>Local SGD + weighted model averaging</b> after every epoch. Not DDP/FSDP, not model splitting or pooled VRAM. Each worker must run Phase 19 server on port 45420. Keep other training idle.</p>
<div class="box"><label>Private cluster token <small>(page memory only)</small></label><input id="token" type="password" autocomplete="off">
<label>Model</label><select id="model"></select><p id="description"></p>
<label>Dataset path on Laptop A under data/</label><input id="dataset" value="data/zoo_demo/tabular_regression.csv">
<label>Remote workers (IP:port; separate with semicolons)</label><input id="workers" value="192.168.50.2:45420">
<label>Epochs</label><input id="epochs" type="number" min="1" max="100" value="3">
<label>Batch size</label><input id="batch" type="number" min="1" max="64" value="4">
<label>Learning rate</label><input id="lr" type="number" min="0.000001" max="0.05" step="0.0001" value="0.001">
<button id="start">Start multi-GPU training</button></div>
<div class="box"><b id="state">Ready for test</b><pre id="log">Connect worker(s) before starting.</pre></div>
<script>
const $=id=>document.getElementById(id);let presets=[],jobId=null;
/* DGPU_P20_WORKER_PREFILL */
const p20Workers = new URLSearchParams(location.search).get('workers');
if(p20Workers && /^[0-9.;:]+$/.test(p20Workers)) $('workers').value=p20Workers;

async function load(){const r=await fetch('/api/distributed-zoo/catalog');const d=await r.json();presets=d.models;const sel=$('model');
for(const p of presets){const x=document.createElement('option');x.value=p.key;x.textContent=p.family+' · '+p.key;sel.append(x)}
sel.value='tabular_mlp_regression';update()}
function update(){const p=presets.find(x=>x.key===$('model').value);if(p){$('description').textContent=p.description;$('batch').value=p.suggested_batch}}
$('model').onchange=update;
async function poll(){try{const r=await fetch('/api/distributed-zoo/status/'+jobId,{headers:{Authorization:'Bearer '+$('token').value}});const d=await r.json();if(!r.ok)throw Error(JSON.stringify(d));
$('state').textContent='Job '+jobId+': '+d.state;$('log').textContent=d.log||'Preparing workers...';
if(d.state==='running')setTimeout(poll,2000);else $('start').disabled=false;}catch(e){$('state').textContent='Status fetch failed';$('log').textContent=String(e);$('start').disabled=false}}
$('start').onclick=async()=>{const token=$('token').value;if(!token){alert('Enter your private cluster token');return}
$('start').disabled=true;
const spec={model:$('model').value,dataset:$('dataset').value,workers:$('workers').value.split(';').map(s=>s.trim()).filter(Boolean),epochs:Number($('epochs').value),batch_size:Number($('batch').value),learning_rate:Number($('lr').value)};
try{const r=await fetch('/api/distributed-zoo/train',{method:'POST',headers:{Authorization:'Bearer '+token,'Content-Type':'application/json'},body:JSON.stringify(spec)});
const d=await r.json();if(!r.ok)throw Error(JSON.stringify(d));jobId=d.id;poll()}catch(e){$('state').textContent='Failed to start';$('log').textContent=String(e);$('start').disabled=false}};
load().catch(e=>$('log').textContent=String(e));
</script></body></html>'''


@router.get('/distributed-zoo', response_class=HTMLResponse)
def page():
    return PAGE
