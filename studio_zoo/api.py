"""Optional authenticated Model Zoo pages and trainer API for existing FastAPI app.

Important: training occurs on the controller's local GPU only. This is *not*
connected to the existing multi-GPU pipeline engine yet.
"""
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

from .models import CATALOG, catalog_dicts

router = APIRouter(tags=['model-zoo'])
_ROOT = Path(__file__).resolve().parents[1]
_RUNS = _ROOT / '.dgpu' / 'model_zoo_runs'
_LOCK = threading.Lock()
_CURRENT = {'run_id': None, 'process': None}


def _auth(authorization: str = Header(default='')):
    token = os.environ.get('DGPU_TOKEN')
    if not token:
        raise HTTPException(503, 'Controller DGPU_TOKEN not configured')
    bearer = authorization.removeprefix('Bearer ').strip() if authorization.startswith('Bearer ') else ''
    if not hmac.compare_digest(bearer.encode('utf-8'), token.encode('utf-8')):
        raise HTTPException(401, 'Cluster token required')
    return True


class TrainRequest(BaseModel):
    model: str
    dataset: str
    epochs: int = Field(default=1, ge=1, le=100)
    batch_size: int = Field(default=4, ge=1, le=256)
    learning_rate: float = Field(default=0.001, gt=0, le=0.1)


def _json_write(target: Path, obj: dict):
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    temp.replace(target)


def _dataset_path(relative: str) -> Path:
    root = (_ROOT / 'data').resolve()
    target = (_ROOT / relative).resolve()
    if not target.is_relative_to(root) or target == root:
        raise HTTPException(400, 'Dataset must be inside project data/ folder')
    if not target.exists():
        raise HTTPException(400, 'Dataset path does not exist on Laptop A')
    if target.is_symlink():
        raise HTTPException(400, 'Dataset symlinks are not supported')
    return target


@router.get('/api/model-zoo/catalog')
def catalog():
    return {'models': catalog_dicts(), 'execution': 'controller_local_only',
            'note': 'New architecture presets run on Laptop A, not two-GPU distributed model parallel'}


@router.post('/api/model-zoo/train', dependencies=[Depends(_auth)])
def start_training(spec: TrainRequest):
    if spec.model not in CATALOG:
        raise HTTPException(400, f'Unknown model: {spec.model}')
    dataset = _dataset_path(spec.dataset)
    model = CATALOG[spec.model]
    if model.task in ('image_classification', 'binary_segmentation') and not dataset.is_dir():
        raise HTTPException(400, 'This model needs a dataset directory')
    if model.task not in ('image_classification', 'binary_segmentation') and not dataset.is_file():
        raise HTTPException(400, 'This model needs a dataset file')
    _RUNS.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        current = _CURRENT['process']
        if current is not None and current.poll() is None:
            raise HTTPException(409, 'Another Model Zoo job is running')
        run_id = uuid.uuid4().hex[:16]
        run_dir = _RUNS / run_id
        run_dir.mkdir()
        log_path = run_dir / 'train.log'
        args = [sys.executable, '-u', '-m', 'studio_zoo.train',
                '--model', spec.model, '--dataset', str(dataset), '--output', str(run_dir),
                '--epochs', str(spec.epochs), '--batch-size', str(spec.batch_size),
                '--lr', str(spec.learning_rate)]
        status = {'id': run_id, 'model': spec.model, 'dataset': spec.dataset,
                  'state': 'running', 'execution': 'controller_local_only',
                  'checkpoint': None, 'returncode': None}
        _json_write(run_dir / 'status.json', status)
        try:
            log_handle = log_path.open('w', encoding='utf-8')
            try:
                process = subprocess.Popen(args, cwd=str(_ROOT), stdout=log_handle,
                                           stderr=subprocess.STDOUT,
                                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            finally:
                log_handle.close()
        except Exception as exc:
            status.update(state='failed', error=f'Failed to launch Python: {type(exc).__name__}')
            _json_write(run_dir / 'status.json', status)
            raise HTTPException(500, 'Could not start trainer') from exc
        _CURRENT.update(run_id=run_id, process=process)

        def finish():
            return_code = process.wait()
            with _LOCK:
                saved = json.loads((run_dir / 'status.json').read_text(encoding='utf-8'))
                saved.update(state='completed' if return_code == 0 else 'failed', returncode=return_code,
                             checkpoint=str(run_dir / 'model.pt') if return_code == 0 else None)
                _json_write(run_dir / 'status.json', saved)
                if _CURRENT['run_id'] == run_id:
                    _CURRENT.update(run_id=None, process=None)

        threading.Thread(target=finish, daemon=True).start()
    return {'id': run_id, 'state': 'running'}


@router.get('/api/model-zoo/status/{run_id}', dependencies=[Depends(_auth)])
def status(run_id: str):
    if not re.fullmatch(r'[0-9a-f]{16}', run_id):
        raise HTTPException(404, 'Job not found')
    directory = _RUNS / run_id
    f = directory / 'status.json'
    if not f.exists():
        raise HTTPException(404, 'Job not found')
    state = json.loads(f.read_text(encoding='utf-8'))
    log_file = directory / 'train.log'
    if log_file.exists():
        with log_file.open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 8000))
            state['log'] = stream.read().decode('utf-8', errors='replace')[-8000:]
    else:
        state['log'] = ''
    return state


PAGE = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Distributed-GPU Model Zoo</title>
<style>body{font:16px system-ui;background:#101827;color:#eff6ff;max-width:1000px;margin:36px auto;padding:0 20px}h1{color:#82baff}label{display:block;margin:14px 0 6px}.panel{background:#1e2b43;border-radius:12px;padding:22px;margin:20px 0}input,select,button{padding:10px;border-radius:7px;font:inherit;max-width:100%;box-sizing:border-box}input,select{background:#101827;color:#fff;border:1px solid #546582;width:100%}button{cursor:pointer;background:#4080ce;color:white;border:0;margin-top:16px}button:disabled{opacity:.5}pre{background:#0d1320;padding:12px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:360px;overflow:auto}small{color:#b5c9e8}.warn{color:#ffd58c}a{color:#92c6ff}</style></head>
<body><h1>Distributed-GPU Model Zoo</h1><p><a href="/">Dashboard</a> · <a href="/train">Existing two-GPU trainer</a></p>
<p class="warn">New models train on the controller GPU (Laptop A) only. The existing two-GPU Linear/ReLU training remains separate. Do not run both trainers simultaneously.</p>
<div class="panel"><label>Cluster token (kept only in this page's memory; never saved)</label><input type="password" id="token" autocomplete="off">
<label>Model architecture</label><select id="model"></select><p id="description"><small></small></p>
<label>Dataset path on Laptop A (inside D:\Distributed-GPU\data)</label><input id="dataset" value="data/demo_train.csv">
<label>Epochs</label><input type="number" id="epochs" value="1" min="1" max="100">
<label>Batch size</label><input type="number" id="batch" value="4" min="1" max="256">
<label>Learning rate</label><input type="number" id="lr" value="0.001" step="0.0001" min="0.000001" max="0.1">
<button id="start">Start local training</button><small> Training output: .dgpu/model_zoo_runs/&lt;run_id&gt;/model.pt</small></div>
<div class="panel"><b id="state">No job started</b><pre id="log">Select a model and provide its matching dataset.</pre></div>
<script>
let catalog = [], runId = null;
const $=id=>document.getElementById(id);
async function load(){let r=await fetch('/api/model-zoo/catalog');let data=await r.json();catalog=data.models;$('model').replaceChildren();for(const p of catalog){let o=document.createElement('option');o.value=p.key;o.textContent=p.family+' · '+p.key;$('model').append(o)} update()}
function update(){let p=catalog.find(x=>x.key===$('model').value);if(!p)return; $('description').textContent=p.description+' | Dataset: '+p.dataset+' | Execution: local GPU only';$('batch').value=p.suggested_batch;}
$('model').onchange=update;
async function poll(){if(!runId)return;try{const r=await fetch('/api/model-zoo/status/'+runId,{headers:{Authorization:'Bearer '+$('token').value}});const d=await r.json();if(!r.ok)throw Error(JSON.stringify(d));$('state').textContent='Job '+runId+' — '+d.state;$('log').textContent=d.log || 'Starting...';if(d.state==='running')setTimeout(poll,2500);else $('start').disabled=false;}catch(e){$('state').textContent='Status error';$('log').textContent=String(e);$('start').disabled=false}}
$('start').onclick=async()=>{let token=$('token').value;if(!token){alert('Enter the private cluster token');return;} $('start').disabled=true;const params={model:$('model').value,dataset:$('dataset').value,epochs:Number($('epochs').value),batch_size:Number($('batch').value),learning_rate:Number($('lr').value)};try{let r=await fetch('/api/model-zoo/train',{method:'POST',headers:{'Content-Type':'application/json',Authorization:'Bearer '+token},body:JSON.stringify(params)});let d=await r.json();if(!r.ok)throw Error(JSON.stringify(d));runId=d.id;poll();}catch(e){$('state').textContent='Start failed';$('log').textContent=String(e);$('start').disabled=false}}
load().catch(e=>$('log').textContent=String(e));
</script></body></html>'''


@router.get('/model-zoo', response_class=HTMLResponse)
def model_zoo_page():
    return PAGE
