"""Worker-side local Join UI and no-console CUDA worker supervisor.

Run only on the worker machine, bound to loopback for UI and private LAN for GPU API.
"""
import base64
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .crypto import new_pairing_key, unwrap_token
from .network import private_ipv4, discover, controller_ip_for

ROOT = Path(__file__).resolve().parents[1]
DGPU = ROOT / '.dgpu'
CONFIG = DGPU / 'phase20_agent.json'
SECRET_FILE = DGPU / 'phase20_token.dpapi'
STATE_LOCK = threading.RLock()
STATE = {'status': 'unpaired', 'message': 'Enter host and join code',
         'fingerprint':'', 'host':'', 'gpu':'Unknown', 'worker':'stopped'}
TOKEN = None
CURRENT_WORKER = None
LAST_LAUNCH = 0.0
WORKER_LOG = None
STOP = threading.Event()


def log(message):
    DGPU.mkdir(exist_ok=True)
    with (DGPU / 'phase20_agent.log').open('a',encoding='utf-8') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S')+' '+message+'\n')


def _ps(script, input_value):
    exe = os.path.join(os.environ.get('SystemRoot','C:\\Windows'), 'System32',
                       'WindowsPowerShell', 'v1.0', 'powershell.exe')
    proc = subprocess.run([exe,'-NoProfile','-NonInteractive','-Command',script],
                          input=input_value, capture_output=True, text=True,
                          timeout=15, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW',0))
    if proc.returncode: raise RuntimeError('Windows DPAPI credential operation failed')
    return proc.stdout.strip().lstrip('\ufeff')


def save_private_token(token):
    # Token goes over stdin only, never in the command line or logs.
    encrypted = _ps('$s=[Console]::In.ReadToEnd(); $v=ConvertTo-SecureString -String $s -AsPlainText -Force; ConvertFrom-SecureString -SecureString $v', token)
    if not encrypted: raise RuntimeError('Cannot encrypt token with DPAPI')
    DGPU.mkdir(parents=True, exist_ok=True)
    temp = SECRET_FILE.with_suffix('.tmp')
    temp.write_text(encrypted, encoding='utf-8')
    temp.replace(SECRET_FILE)


def load_private_token():
    if not SECRET_FILE.exists(): return None
    encrypted = SECRET_FILE.read_text(encoding='utf-8').strip()
    script = '''$s=[Console]::In.ReadToEnd() | ConvertTo-SecureString; $p=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($s); try { [Console]::Out.Write([Runtime.InteropServices.Marshal]::PtrToStringBSTR($p)) } finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($p) }'''
    return _ps(script, encrypted)


def post_json(host, endpoint, data, timeout=8):
    # Do not use system proxy and do not follow redirects with pairing requests.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs): return None
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    req=urllib.request.Request('http://'+host+':45400'+endpoint,
             data=json.dumps(data).encode('utf-8'),
             headers={'Content-Type':'application/json'}, method='POST')
    with opener.open(req, timeout=timeout) as response:
        return json.load(response)


def gpu_name():
    try:
        import torch
        if torch.cuda.is_available(): return torch.cuda.get_device_name(0)
    except Exception: pass
    return 'No CUDA GPU detected'


def save_config(host, bind_ip):
    DGPU.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG.with_suffix('.tmp')
    tmp.write_text(json.dumps({'host':host,'worker_ip':bind_ip,'port':45420},indent=2),encoding='utf-8')
    tmp.replace(CONFIG)


def _set(**kwargs):
    with STATE_LOCK: STATE.update(kwargs)


def join_worker(host, code):
    global TOKEN
    with STATE_LOCK:
        if STATE['status'] in ('requesting','awaiting_approval'):
            return
        STATE.update(status='requesting', message='Requesting host approval...',host=host, fingerprint='')
    try:
        private_key, public_b64, fp = new_pairing_key()
        ip = controller_ip_for(host)
        _set(fingerprint=fp, gpu=gpu_name())
        result = post_json(host, '/api/cluster/request', {
            'code':code,'name':socket.gethostname(),'gpu':gpu_name(),
            'public_key':public_b64,'worker_port':45420}, 10)
        if result.get('fingerprint')!=fp: raise RuntimeError('Pairing fingerprint mismatch')
        _set(status='awaiting_approval',message='Approve on Host. Verify fingerprint '+fp)
        begin = time.monotonic()
        while time.monotonic()-begin < 300 and not STOP.is_set():
            time.sleep(2)
            try:
                reply = post_json(host,'/api/cluster/poll',
                      {'request_id':result['request_id'],'poll_secret':result['poll_secret']},8)
            except urllib.error.URLError:
                continue
            if reply.get('status')=='approved':
                if reply.get('fingerprint')!=fp: raise RuntimeError('Approval fingerprint mismatch')
                token=unwrap_token(private_key, reply['encrypted_token'])
                save_private_token(token)
                save_config(host,ip)
                TOKEN=token
                _set(status='paired',message='Paired with '+host+'. GPU worker will start automatically.')
                log('Pairing approved by host '+host+'; encrypted token saved locally')
                return
        raise RuntimeError('Pairing expired. Create a fresh host join code.')
    except Exception as exc:
        _set(status='error',message=str(exc)[:180])
        log('Pairing failed: '+type(exc).__name__+' (details in local UI)')


def port_open(ip, port):
    try:
        with socket.create_connection((ip,port),timeout=0.6): return True
    except OSError: return False


def worker_watchdog():
    global CURRENT_WORKER, LAST_LAUNCH, WORKER_LOG
    while not STOP.wait(8):
        try:
            if TOKEN is None or not CONFIG.exists(): continue
            settings=json.loads(CONFIG.read_text(encoding='utf-8'))
            ip=private_ipv4(settings['worker_ip'])
            if port_open(ip,45420):
                _set(worker='online (port 45420)')
                continue
            if CURRENT_WORKER and CURRENT_WORKER.poll() is None:
                _set(worker='starting or blocked')
                continue
            if time.monotonic()-LAST_LAUNCH < 20: continue
            LAST_LAUNCH=time.monotonic()
            env=os.environ.copy();env['DGPU_TOKEN']=TOKEN
            if WORKER_LOG: WORKER_LOG.close()
            DGPU.mkdir(parents=True,exist_ok=True)
            WORKER_LOG=(DGPU/'phase20_worker.log').open('ab',buffering=0)
            CURRENT_WORKER=subprocess.Popen([sys.executable,'-u','-m','uvicorn',
                              'studio_multi.worker:app','--host',ip,'--port','45420'],
                              cwd=str(ROOT),env=env,stdout=WORKER_LOG,stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            _set(worker='launching (PID '+str(CURRENT_WORKER.pid)+')')
            log('Launched GPU worker PID '+str(CURRENT_WORKER.pid)+' on '+ip+':45420')
        except Exception as exc:
            _set(worker='worker error: '+type(exc).__name__)
            log('Worker supervisor error: '+str(exc)[:150])


JOIN_HTML='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DGPU · Join Cluster</title><style>body{font:16px system-ui;max-width:650px;margin:40px auto;padding:0 18px;background:#101927;color:#edf4ff}section{padding:22px;margin:15px 0;background:#1b2a40;border-radius:12px}input,button{font:inherit;padding:10px;border-radius:6px;margin:7px 0}input{background:#101927;color:white;border:1px solid #8094b4;width:95%;box-sizing:border-box}button{border:0;background:#4082ca;color:white;cursor:pointer}small{color:#c4d1e0}strong{color:#b6e7be}code{color:#aee5af}</style></head><body><h1>Join GPU Cluster</h1><p>This page runs only on your laptop. No PowerShell needed for joining after one-time installation.</p><section><p><button onclick="find()">Find Host Automatically</button></p><label>Host LAN IP (fallback)</label><input id="host" value="192.168.50.1"><label>Temporary join code</label><input id="code" placeholder="ABCD-EFGH" autocomplete="off"><button onclick="join()">Request to Join</button><p>Device fingerprint: <strong id="fp">—</strong></p><small>Compare this fingerprint with the Host screen before it approves your device.</small></section><section><h2>Status</h2><p id="status">Ready</p><p>GPU: <span id="gpu">Loading...</span></p><p>Worker: <span id="worker">Unknown</span></p></section><script>const $=id=>document.getElementById(id);async function call(path,data){let r=await fetch(path,{method:data?'POST':'GET',headers:{'Content-Type':'application/json'},body:data?JSON.stringify(data):undefined});let d=await r.json();if(!r.ok)throw Error(d.error||r.status);return d}async function find(){try{let d=await call('/api/agent/discover');if(d.hosts.length){$('host').value=d.hosts[0].ip;$('status').textContent='Found '+d.hosts[0].name}else $('status').textContent='No host discovered. Enter host IP manually.'}catch(e){$('status').textContent=String(e)}}async function join(){try{const d=await call('/api/agent/join',{host:$('host').value,code:$('code').value});$('status').textContent=d.message}catch(e){$('status').textContent=String(e)}}async function status(){try{const d=await call('/api/agent/status');$('status').textContent=d.message;$('fp').textContent=d.fingerprint||'—';$('gpu').textContent=d.gpu;$('worker').textContent=d.worker;}catch(e){$('status').textContent=String(e)}}setInterval(status,2000);status();</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): return

    def _respond(self, data, code=200, html=False):
        body=data.encode('utf-8') if isinstance(data,str) else json.dumps(data).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type','text/html; charset=utf-8' if html else 'application/json')
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self' 'unsafe-inline'")
        self.end_headers();self.wfile.write(body)

    def do_GET(self):
        if self.path in ('/','/join'):
            self._respond(JOIN_HTML,html=True)
        elif self.path=='/api/agent/status':
            with STATE_LOCK: self._respond(dict(STATE))
        elif self.path=='/api/agent/discover':
            self._respond({'hosts':discover(timeout=1.2)})
        else: self._respond({'error':'Not found'},404)

    def do_POST(self):
        # Block cross-site browser requests to the local control panel.
        origin=self.headers.get('Origin','')
        if origin not in ('http://127.0.0.1:45430','http://localhost:45430'):
            self._respond({'error':'Invalid origin'},403);return
        if self.path!='/api/agent/join' or self.headers.get('Content-Type','').split(';')[0].strip()!='application/json':
            self._respond({'error':'Not found / invalid content type'},404);return
        try:
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<=2048: raise ValueError('Invalid request length')
            spec=json.loads(self.rfile.read(n))
            host=private_ipv4(spec['host'])
            code=str(spec['code']).strip().upper()
            if len(code.replace('-',''))!=8: raise ValueError('Join code must have eight characters')
            with STATE_LOCK:
                if STATE['status'] in ('requesting','awaiting_approval'): raise ValueError('Pairing already in progress')
            threading.Thread(target=join_worker,args=(host,code),daemon=True).start()
            self._respond({'message':'Pairing started. Approve on Host.'})
        except (KeyError,ValueError,TypeError, json.JSONDecodeError) as exc:
            self._respond({'error':str(exc)},400)


def main():
    global TOKEN
    if sys.platform!='win32': raise RuntimeError('Worker pairing service supports Windows only')
    DGPU.mkdir(exist_ok=True)
    try:
        TOKEN=load_private_token()
        if TOKEN and CONFIG.exists():
            settings=json.loads(CONFIG.read_text(encoding='utf-8'))
            _set(status='paired',message='Previously paired with '+settings['host']+'.',host=settings['host'],gpu=gpu_name())
    except Exception as exc:
        TOKEN=None
        log('Cannot reload saved DPAPI token: '+str(exc)[:120])
    _set(gpu=gpu_name())
    threading.Thread(target=worker_watchdog,daemon=True).start()
    try:
        server=ThreadingHTTPServer(('127.0.0.1',45430),Handler)
        log('Join UI running on http://127.0.0.1:45430/join')
        server.serve_forever(poll_interval=0.5)
    except OSError as exc:
        log('Cannot bind local Join UI: '+str(exc))
        raise
    finally: STOP.set()

if __name__=='__main__': main()
