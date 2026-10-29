"""FastAPI host control panel. Owner approves a worker after fingerprint check."""
import hmac
import os
import secrets
import socket
import threading
import time
import uuid
from collections import defaultdict, deque

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .crypto import parse_public_key, wrap_token
from .network import private_ipv4, run_discovery

router = APIRouter(tags=['phase20-pairing'])
_LOCK = threading.Lock()
_SESSION = {'code': None, 'expires': 0.0, 'attempts': 0}
_PENDING = {}
_APPROVED = {}
_ATTEMPTS = defaultdict(deque)
_STOP_DISCOVERY = threading.Event()


def admin(authorization: str = Header(default='')):
    correct = os.environ.get('DGPU_TOKEN', '')
    supplied = authorization[7:] if authorization.startswith('Bearer ') else ''
    if not correct or not hmac.compare_digest(supplied.encode(), correct.encode()):
        raise HTTPException(401, 'Host token required')


def is_valid_ip(request):
    peer = request.client.host if request.client else ''
    try: return private_ipv4(peer)
    except ValueError as exc: raise HTTPException(403, 'LAN-only endpoint') from exc


class PairRequest(BaseModel):
    code: str = Field(min_length=8, max_length=12)
    name: str = Field(min_length=1, max_length=64)
    gpu: str = Field(default='Unknown', max_length=128)
    public_key: str = Field(min_length=200, max_length=2500)
    worker_port: int = Field(default=45420, ge=1024, le=65535)


class PollRequest(BaseModel):
    request_id: str
    poll_secret: str


class Approval(BaseModel):
    request_id: str
    fingerprint: str


@router.on_event('startup')
def _discovery_start():
    _STOP_DISCOVERY.clear()
    threading.Thread(target=run_discovery, args=(_STOP_DISCOVERY,), daemon=True).start()


@router.on_event('shutdown')
def _discovery_stop():
    _STOP_DISCOVERY.set()


@router.post('/api/cluster/open', dependencies=[Depends(admin)])
def open_pairing():
    with _LOCK:
        alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
        code = ''.join(secrets.choice(alphabet) for _ in range(8))
        _SESSION.update(code=code, expires=time.monotonic()+300, attempts=0)
        _PENDING.clear()
        return {'code':code[:4]+'-'+code[4:], 'expires_seconds':300}


@router.post('/api/cluster/close', dependencies=[Depends(admin)])
def close_pairing():
    with _LOCK:
        _SESSION.update(code=None, expires=0)
        _PENDING.clear()
    return {'ok':True}


@router.post('/api/cluster/request')
def request_pairing(spec: PairRequest, request: Request):
    peer = is_valid_ip(request)
    now = time.monotonic()
    try:
        public_key, fp = parse_public_key(spec.public_key)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    with _LOCK:
        recent = _ATTEMPTS[peer]
        while recent and recent[0] < now-120: recent.popleft()
        if len(recent) >= 6:
            raise HTTPException(429, 'Pairing rate limit reached')
        recent.append(now)
        candidate = spec.code.replace('-', '').upper().strip()
        code = _SESSION['code']
        if code is None or now > _SESSION['expires'] or not hmac.compare_digest(code, candidate):
            raise HTTPException(403, 'Invalid or expired join code')
        if len(_PENDING) >= 8:
            raise HTTPException(429, 'Too many pending devices')
        req_id = uuid.uuid4().hex
        poll_secret = secrets.token_urlsafe(32)
        _PENDING[req_id] = {'id':req_id, 'secret':poll_secret, 'ip':peer,
                'name':spec.name, 'gpu':spec.gpu, 'fingerprint':fp,
                'key':public_key, 'port':spec.worker_port, 'status':'pending',
                'created':now, 'ciphertext':None}
        return {'request_id':req_id, 'poll_secret':poll_secret, 'fingerprint':fp,
                'status':'pending', 'expires_seconds':max(0, int(_SESSION['expires']-now))}


@router.post('/api/cluster/poll')
def poll_pairing(spec: PollRequest, request: Request):
    peer = is_valid_ip(request)
    with _LOCK:
        entry = _PENDING.get(spec.request_id)
        if not entry or time.monotonic() > _SESSION['expires'] or time.monotonic() - entry['created'] > 300:
            raise HTTPException(404, 'Pairing expired')
        if entry['ip'] != peer or not hmac.compare_digest(entry['secret'],spec.poll_secret):
            raise HTTPException(403, 'Invalid polling credentials')
        response = {'status':entry['status'], 'fingerprint':entry['fingerprint']}
        if entry['status'] == 'approved': response['encrypted_token'] = entry['ciphertext']
        return response


@router.get('/api/cluster/pending', dependencies=[Depends(admin)])
def pending():
    with _LOCK:
        now = time.monotonic()
        return {'pairing_open':bool(_SESSION['code'] and now < _SESSION['expires']),
                'seconds_left':max(0,int(_SESSION['expires']-now)),
                'devices':[{'id':d['id'],'name':d['name'],'ip':d['ip'],'gpu':d['gpu'],
                           'fingerprint':d['fingerprint'],'status':d['status']}
                           for d in _PENDING.values() if now - d['created'] < 300]}


@router.post('/api/cluster/approve', dependencies=[Depends(admin)])
def approve(spec: Approval):
    with _LOCK:
        entry = _PENDING.get(spec.request_id)
        if not entry or entry['status'] != 'pending' or time.monotonic() > _SESSION['expires']:
            raise HTTPException(404, 'Pending request not found or expired')
        if not hmac.compare_digest(entry['fingerprint'], spec.fingerprint.upper()):
            raise HTTPException(409, 'Worker fingerprint mismatch')
        try:
            entry['ciphertext'] = wrap_token(entry['key'], os.environ['DGPU_TOKEN'])
        except (ValueError, KeyError):
            raise HTTPException(503, 'Host cluster token unavailable')
        entry['status'] = 'approved'
        _APPROVED[entry['ip']] = {'name': entry['name'], 'gpu':entry['gpu'],
                                  'ip':entry['ip'], 'port':entry['port'], 'fingerprint':entry['fingerprint']}
        return {'ok':True, 'worker':entry['ip']+':'+str(entry['port'])}


@router.get('/api/cluster/workers', dependencies=[Depends(admin)])
def workers():
    with _LOCK:
        return {'workers': list(_APPROVED.values())}


PAGE = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DGPU · Create Cluster</title>
<style>body{font:16px system-ui;max-width:850px;margin:36px auto;padding:0 18px;background:#101927;color:#edf4ff}a{color:#9ac7ff}section{background:#1b2a40;padding:20px;border-radius:12px;margin:16px 0}input,button{font:inherit;padding:11px;border-radius:7px;border:1px solid #647a99}input{background:#101927;color:white;max-width:430px;width:90%}button{background:#3b79c7;color:white;cursor:pointer;margin:6px}button:disabled{opacity:.5}code{font:24px ui-monospace;color:#b7e5bc}small{color:#bac8df}li{padding:9px 0}strong{color:#d4e7ff}</style></head><body>
<h1>Create Cluster</h1><p><a href="/">Dashboard</a> · <a href="/distributed-zoo">Distributed Training</a></p>
<section><p>Enter your existing <b>private cluster token</b> (stays in this tab; do not share it with other people).</p><input id="token" type="password" placeholder="Host cluster token" autocomplete="off"><button onclick="openPairing()">Create temporary join code</button><button onclick="closePairing()">Close pairing</button><p><code id="code">----</code> <small id="expires"></small></p><small>On Laptop B, open the local Join page and enter this code. Verify the displayed device fingerprint matches before approval.</small></section>
<section><h2>Pending requests</h2><ul id="pending"><li>No pending requests</li></ul></section><section><h2>Approved devices (this controller session)</h2><ul id="approved"><li>No devices yet</li></ul><button onclick="goTrain()">Train with approved GPUs</button><p><small>Training API remains protected by your existing cluster token. This beta pairing works only on a trusted dedicated LAN.</small></p></section><p id="feedback" role="status"></p>
<script>
let code=''; let approvedWorkers=[];const $=id=>document.getElementById(id);
async function api(path,method='GET',payload){let r=await fetch(path,{method,headers:{Authorization:'Bearer '+$('token').value,'Content-Type':'application/json'},body:payload?JSON.stringify(payload):undefined});const v=await r.json();if(!r.ok)throw Error(v.detail||r.status);return v}
async function openPairing(){try{let d=await api('/api/cluster/open','POST');$('code').textContent=d.code;show('Temporary join code created.');update()}catch(e){show(String(e))}}
async function closePairing(){try{await api('/api/cluster/close','POST');$('code').textContent='----';show('Pairing closed.');update()}catch(e){show(String(e))}}
async function approve(id,fp){if(!confirm('Approve only if Laptop B shows fingerprint '+fp+' ?'))return;try{await api('/api/cluster/approve','POST',{request_id:id,fingerprint:fp});show('Device approved. It should start its worker.');update()}catch(e){show(String(e))}}
function show(s){$('feedback').textContent=s}
function goTrain(){if(!approvedWorkers.length){show('No approved GPU workers yet.');return;} const addresses=approvedWorkers.map(w=>w.ip+':'+w.port).join(';'); location.href='/distributed-zoo?workers='+encodeURIComponent(addresses);}
function txt(el,s){el.textContent=s}
async function update(){if(!$('token').value)return;try{const d=await api('/api/cluster/pending');$('expires').textContent=d.pairing_open?'Expires in '+d.seconds_left+'s':'No active pairing code';let ul=$('pending');ul.replaceChildren();for(const x of d.devices){let li=document.createElement('li');li.textContent=x.name+' | '+x.gpu+' | '+x.ip+' | fingerprint '+x.fingerprint+' | '+x.status+' ';if(x.status==='pending'){let b=document.createElement('button');b.textContent='Approve';b.onclick=()=>approve(x.id,x.fingerprint);li.append(b)}ul.append(li)}if(!d.devices.length)ul.textContent='No pending requests';const w=await api('/api/cluster/workers');approvedWorkers=w.workers;let list=$('approved');list.replaceChildren();for(const x of w.workers){let li=document.createElement('li');txt(li,x.name+' | '+x.ip+':'+x.port+' | '+x.gpu);list.append(li)}if(!w.workers.length)list.textContent='No devices yet'}catch(e){show('Status: '+String(e))}}
setInterval(update,3000);
</script></body></html>'''


@router.get('/cluster', response_class=HTMLResponse)
def cluster_page():
    return PAGE
