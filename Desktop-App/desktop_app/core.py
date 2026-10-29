"""Secure local bridge between native desktop GUI and installed DGPU Phase19/20 APIs.

No HTTP server is started by this client. UI-to-Python calls use pywebview's private
JavaScript bridge. Existing cluster secrets never enter the JS frontend.
"""
from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

APP_NAME = 'DistributedGPUStudio'
HOST_PORT = 45400
AGENT_PORT = 45430
WORKER_PORT = 45420
MAX_RESPONSE = 2 * 1024 * 1024
USER_SETTINGS = Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / APP_NAME / 'settings.json'
CREATE_NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def valid_private_ip(value: str) -> str:
    try:
        address = ipaddress.ip_address(str(value).strip())
    except ValueError as exc:
        raise ValueError('Please enter a valid private IPv4 address.') from exc
    if not isinstance(address, ipaddress.IPv4Address) or not address.is_private or address.is_loopback or address.is_link_local or address.is_multicast or address.is_unspecified or address.is_reserved:
        raise ValueError('Only a private LAN IPv4 address is accepted.')
    return str(address)


def _json_read(path: Path) -> dict:
    try:
        content = json.loads(path.read_text(encoding='utf-8-sig'))
        return content if isinstance(content, dict) else {}
    except (OSError, ValueError):
        return {}


def _json_save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    tmp.replace(path)


def locate_project(override: str = '') -> Path | None:
    candidates = [override, os.environ.get('DGPU_PROJECT_ROOT', ''), 'D:/Distributed-GPU',
                  str(Path.home() / 'Distributed-GPU'), str(Path.cwd())]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if (path / 'studio' / 'server.py').is_file() or (path / 'studio_pairing' / 'agent.py').is_file():
            return path.resolve()
    return None


def _power_shell(script: str, input_value: str) -> str:
    if sys.platform != 'win32':
        raise RuntimeError('Windows DPAPI is required. Run this app on Windows.')
    exe = os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'System32',
                       'WindowsPowerShell', 'v1.0', 'powershell.exe')
    proc = subprocess.run([exe, '-NoProfile', '-NonInteractive', '-Command', script],
                          input=input_value, capture_output=True, text=True,
                          timeout=18, creationflags=CREATE_NO_WINDOW)
    if proc.returncode or not proc.stdout.strip():
        raise RuntimeError('Windows could not access the protected cluster credential in this account.')
    return proc.stdout.strip().lstrip('\ufeff')


def decrypt_existing_token(root: Path) -> str | None:
    file = root / '.dgpu' / 'worker_config.json'
    if not file.is_file():
        return None
    obj = _json_read(file)
    blob = obj.get('protected_token')
    if not isinstance(blob, str) or not blob:
        raise RuntimeError('Existing worker_config.json has no protected_token. It was not modified.')
    script = ("$ErrorActionPreference='Stop'; "
              "$s=[Console]::In.ReadToEnd() | ConvertTo-SecureString; "
              "$p=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($s); "
              "try { [Console]::Out.Write([Runtime.InteropServices.Marshal]::PtrToStringBSTR($p)) } "
              "finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($p) }")
    return _power_shell(script, blob)


def create_new_token(root: Path, controller_running: bool) -> tuple[str, bool]:
    """One-time initialization; NEVER silently rotate or replace existing keys."""
    file = root / '.dgpu' / 'worker_config.json'
    if file.exists():
        return decrypt_existing_token(root) or '', False
    if controller_running:
        raise RuntimeError('Controller is already running but its token file is missing. Nothing was changed. Recover the existing token before creating another.')
    token = secrets.token_urlsafe(32)  # 256 bits; 43-character Base64url string
    script = ("$ErrorActionPreference='Stop'; "
              "$raw=[Console]::In.ReadToEnd(); "
              "$s=ConvertTo-SecureString -String $raw -AsPlainText -Force; "
              "ConvertFrom-SecureString -SecureString $s")
    protected = _power_shell(script, token)
    file.parent.mkdir(parents=True, exist_ok=True)
    # Create exclusively; preserve the original file even on races.
    try:
        with file.open('x', encoding='utf-8') as stream:
            json.dump({'protected_token': protected}, stream, indent=2)
    except FileExistsError:
        return decrypt_existing_token(root) or '', False
    return token, True


def api_json(url: str, method: str = 'GET', payload: dict | None = None, token: str | None = None,
             timeout: float = 5, origin: str | None = None) -> dict:
    data = json.dumps(payload).encode('utf-8') if payload is not None else None
    headers = {'Accept': 'application/json'}
    if data is not None:
        headers['Content-Type'] = 'application/json'
    if token is not None:
        headers['Authorization'] = 'Bearer ' + token
    if origin:
        headers['Origin'] = origin
    req = urllib.request.Request(url, data=data, method=method, headers=headers)

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args: Any, **kwargs: Any) -> None:
            return None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(req, timeout=timeout) as reply:
            body = reply.read(MAX_RESPONSE + 1)
            if len(body) > MAX_RESPONSE:
                raise RuntimeError('Server response exceeded safety limit')
            return json.loads(body.decode('utf-8'))
    except urllib.error.HTTPError as exc:
        error_body = exc.read(2048).decode('utf-8', errors='replace')
        try:
            detail = json.loads(error_body).get('detail') or json.loads(error_body).get('error')
        except ValueError:
            detail = None
        raise RuntimeError(f'{exc.code}: {detail or "Server rejected request"}') from exc
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        raise RuntimeError('Service is offline or not reachable. Check LAN and background agent.') from exc
    except (ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError('Unexpected response from cluster service.') from exc


def tcp_open(host: str, port: int, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class DesktopBridge:
    def __init__(self):
        self.settings = _json_read(USER_SETTINGS)
        self.root = locate_project(str(self.settings.get('root', '')))
        self.host = str(self.settings.get('host', '192.168.50.1'))
        self.role = str(self.settings.get('role', 'host')) if self.settings.get('role') in ('host', 'worker') else 'host'
        self._token: str | None = None
        self._token_lock = threading.Lock()

    def _save_settings(self) -> None:
        _json_save(USER_SETTINGS, {'root': str(self.root) if self.root else '',
                                  'host': self.host, 'role': self.role})

    def _host_api(self, path: str, method: str = 'GET', payload: dict | None = None,
                  authenticated: bool = False) -> dict:
        token = self._get_token() if authenticated else None
        return api_json(f'http://{self.host}:{HOST_PORT}{path}', method, payload, token)

    def _get_token(self) -> str:
        with self._token_lock:
            if self._token:
                return self._token
            if not self.root:
                raise RuntimeError('Choose your Distributed-GPU project folder in Settings.')
            token = decrypt_existing_token(self.root)
            if not token:
                raise RuntimeError('Host has no cluster credential yet. Choose Create Cluster to initialize it.')
            self._token = token
            return token

    def _agent_api(self, path: str, method: str = 'GET', payload: dict | None = None) -> dict:
        # Phase 20 agent rejects POST unless Origin is its own loopback URL.
        return api_json(f'http://127.0.0.1:{AGENT_PORT}{path}', method, payload,
                        origin=f'http://127.0.0.1:{AGENT_PORT}' if method == 'POST' else None)

    @staticmethod
    def safe(action, *args, **kwargs) -> dict:
        try:
            return {'ok': True, 'data': action(*args, **kwargs)}
        except Exception as exc:
            return {'ok': False, 'error': str(exc)[:300]}

    def settings_update(self, role: str, host: str, root: str) -> dict:
        def run():
            if role not in ('host', 'worker'):
                raise ValueError('Select Create Cluster or Join Cluster.')
            host_ip = valid_private_ip(host)
            path = locate_project(root)
            if not path:
                raise ValueError('Selected project folder must contain studio/server.py or studio_pairing/agent.py.')
            self.role, self.host, self.root = role, host_ip, path
            self._token = None
            self._save_settings()
            return {'role': self.role, 'host': self.host, 'root': str(self.root)}
        return self.safe(run)

    def initialize_host(self) -> dict:
        def run():
            if self.role != 'host':
                raise RuntimeError('Choose Host mode in Settings first.')
            if not self.root:
                raise RuntimeError('Configure your project folder in Settings first.')
            token, created = create_new_token(self.root, tcp_open(self.host, HOST_PORT))
            if not token:
                raise RuntimeError('Host token is empty; nothing was modified.')
            self._token = token
            return {'created': created, 'weak_legacy_token': len(token.encode('utf-8')) < 16,
                    'controller_online': tcp_open(self.host, HOST_PORT)}
        return self.safe(run)

    def dashboard(self) -> dict:
        def run():
            online = tcp_open(self.host, HOST_PORT)
            agent_online = tcp_open('127.0.0.1', AGENT_PORT)
            info: dict[str, Any] = {'role': self.role, 'host': self.host,
                                    'root': str(self.root) if self.root else '',
                                    'controller_online': online, 'agent_online': agent_online,
                                    'local_gpu': self.gpu_info(), 'workers': [], 'worker_status': None,
                                    'token_available': bool(self.root and (self.root/'.dgpu'/'worker_config.json').exists())}
            if online and self.role == 'host':
                try:
                    info['workers'] = self._host_api('/api/cluster/workers', authenticated=True).get('workers', [])
                except RuntimeError as exc:
                    info['host_error'] = str(exc)
            if self.role == 'worker' and agent_online:
                try:
                    info['worker_status'] = self._agent_api('/api/agent/status')
                except RuntimeError:
                    pass
            for worker in info['workers']:
                try:
                    worker['online'] = tcp_open(str(worker['ip']), int(worker.get('port', WORKER_PORT)))
                except (ValueError, TypeError, KeyError):
                    worker['online'] = False
            return info
        return self.safe(run)

    @staticmethod
    def gpu_info() -> dict:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                    timeout=3, creationflags=CREATE_NO_WINDOW)
            if result.returncode == 0 and result.stdout.strip():
                cols = [v.strip() for v in result.stdout.splitlines()[0].split(',')]
                if len(cols) == 4:
                    return {'name': cols[0], 'util': int(cols[1]),
                            'used': int(cols[2]), 'total': int(cols[3])}
        except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
            pass
        return {'name': 'GPU status unavailable', 'util': None, 'used': None, 'total': None}

    def create_cluster(self) -> dict:
        def run():
            self.initialize_host_or_raise()
            return self._host_api('/api/cluster/open', method='POST', authenticated=True)
        return self.safe(run)

    def initialize_host_or_raise(self) -> None:
        state = self.initialize_host()
        if not state['ok']:
            raise RuntimeError(state['error'])
        if not tcp_open(self.host, HOST_PORT):
            raise RuntimeError('Secure token is ready. Start the controller service first, then click Create Cluster again.')

    def pairing_status(self) -> dict:
        def run():
            status = self._host_api('/api/cluster/pending', authenticated=True)
            status['approved'] = self._host_api('/api/cluster/workers', authenticated=True).get('workers', [])
            return status
        return self.safe(run)

    def approve(self, request_id: str, fingerprint: str) -> dict:
        def run():
            if not isinstance(request_id, str) or not 10 <= len(request_id) <= 64:
                raise ValueError('Invalid request ID.')
            if not isinstance(fingerprint, str) or len(fingerprint) != 12:
                raise ValueError('Invalid fingerprint.')
            # Verify the request is still in Host's actual pending set.
            pending = self._host_api('/api/cluster/pending', authenticated=True).get('devices', [])
            if not any(x.get('id') == request_id and x.get('fingerprint') == fingerprint and x.get('status') == 'pending' for x in pending):
                raise RuntimeError('Request expired or changed. Ask the other user to request a fresh code.')
            return self._host_api('/api/cluster/approve', method='POST',
                                  payload={'request_id': request_id, 'fingerprint': fingerprint}, authenticated=True)
        return self.safe(run)

    def discover_hosts(self) -> dict:
        return self.safe(lambda: self._agent_api('/api/agent/discover').get('hosts', []))

    def join_cluster(self, host: str, code: str) -> dict:
        def run():
            host_ip = valid_private_ip(host)
            token = str(code).strip().upper()
            if len(token.replace('-', '')) != 8:
                raise ValueError('Join code should be eight characters.')
            self.host = host_ip
            self._save_settings()
            return self._agent_api('/api/agent/join', method='POST', payload={'host': host_ip, 'code': token})
        return self.safe(run)

    def worker_status(self) -> dict:
        return self.safe(lambda: self._agent_api('/api/agent/status'))

    def catalog(self) -> dict:
        def run():
            # Every /api/* path is gated by the existing session middleware.
            # The native bridge uses the existing DPAPI credential, never a web cookie.
            errors = {}
            datasets = {}
            for mode, path in (
                ('single_gpu', '/api/model-zoo/catalog'),
                ('multi_gpu', '/api/distributed-zoo/catalog'),
            ):
                try:
                    datasets[mode] = self._host_api(path, authenticated=True).get('models', [])
                except RuntimeError as exc:
                    errors[mode] = str(exc)
            if not datasets:
                raise RuntimeError('Model library could not connect. ' + '; '.join(
                    f'{"This PC" if k == "single_gpu" else "Multiple GPUs"}: {v}'
                    for k, v in errors.items()))
            joined = {}
            for mode in ('single_gpu', 'multi_gpu'):
                for item in datasets.get(mode, []):
                    key = item.get('key')
                    if not isinstance(key, str) or not key:
                        continue
                    if key not in joined:
                        joined[key] = dict(item)
                        joined[key]['modes'] = []
                    # Combine metadata from the local model library with the GPU-capable subset.
                    joined[key]['modes'].append(mode)
                    for field in ('task', 'dataset', 'description', 'family', 'suggested_batch'):
                        if item.get(field) and not joined[key].get(field):
                            joined[key][field] = item[field]
                    joined[key]['experimental'] = key in ('distilgpt2_lora', 'yolo11n_detection')
            return {'models': list(joined.values()), 'errors': errors,
                    'mode_labels': {'single_gpu': 'This PC', 'multi_gpu': 'Multiple GPUs'}}
        return self.safe(run)

    def start_training(self, spec: dict) -> dict:
        def run():
            if self.role != 'host':
                raise RuntimeError('Start training from the Host computer.')
            if not isinstance(spec, dict):
                raise ValueError('Invalid training configuration.')
            catalog_state = self.catalog()
            if not catalog_state['ok']:
                raise RuntimeError(catalog_state['error'])
            models = catalog_state['data']['models']
            model_key = str(spec.get('model', ''))
            model = next((m for m in models if m.get('key') == model_key), None)
            if model is None:
                raise ValueError('This model is not available in the installed Model Zoo.')
            mode = str(spec.get('mode', 'multi_gpu'))
            if mode not in ('single_gpu', 'multi_gpu') or mode not in model['modes']:
                raise ValueError('This model is not supported in the chosen training mode.')
            dataset = str(spec.get('dataset', '')).strip().replace('\\', '/')
            if (not dataset.startswith('data/') or '..' in Path(dataset).parts
                    or '\x00' in dataset):
                raise ValueError('Select a dataset inside the Host data folder.')
            epochs, batch = int(spec.get('epochs', 3)), int(spec.get('batch_size', 8))
            rate = float(spec.get('learning_rate', 0.001))
            if not (1 <= epochs <= 100 and 1 <= batch <= (64 if mode == 'multi_gpu' else 256)
                    and 0 < rate <= (0.05 if mode == 'multi_gpu' else 0.1)):
                raise ValueError('Training settings out of range for this mode.')
            payload = {'model': model_key, 'dataset': dataset, 'epochs': epochs,
                       'batch_size': batch, 'learning_rate': rate}
            if mode == 'multi_gpu':
                workers = self._host_api('/api/cluster/workers', authenticated=True).get('workers', [])
                addresses = [str(w['ip']) + ':' + str(w.get('port', WORKER_PORT)) for w in workers
                             if tcp_open(str(w['ip']), int(w.get('port', WORKER_PORT)))]
                if not addresses:
                    raise RuntimeError('No approved online GPU workers. Connect another computer or choose This PC.')
                payload['workers'] = addresses
                path = '/api/distributed-zoo/train'
            else:
                path = '/api/model-zoo/train'
            response = self._host_api(path, method='POST', payload=payload, authenticated=True)
            self._active_execution = mode
            return {**response, 'mode': mode}
        return self.safe(run)

    def training_status(self, job_id: str, mode: str = '') -> dict:
        def run():
            if not isinstance(job_id, str) or len(job_id) != 16 or not all(c in '0123456789abcdef' for c in job_id):
                raise ValueError('Invalid training job ID.')
            execution = mode or getattr(self, '_active_execution', 'multi_gpu')
            if execution not in ('single_gpu', 'multi_gpu'):
                raise ValueError('Invalid training mode.')
            route = '/api/model-zoo/status/' if execution == 'single_gpu' else '/api/distributed-zoo/status/'
            return self._host_api(route + job_id, authenticated=True)
        return self.safe(run)

    def training_history(self) -> dict:
        def run():
            if self.role != 'host' or not self.root:
                return []
            jobs = []
            for dirname, mode in (('model_zoo_runs', 'single_gpu'), ('phase19_runs', 'multi_gpu')):
                runs = self.root / '.dgpu' / dirname
                if not runs.is_dir():
                    continue
                for status_path in runs.glob('*/status.json'):
                    state = _json_read(status_path)
                    if not state:
                        continue
                    jobs.append({'id': str(state.get('id', '')), 'model': str(state.get('model', '')),
                                 'state': str(state.get('state', '')),
                                 'dataset': str(state.get('dataset', '')),
                                 'checkpoint': bool(state.get('checkpoint')),
                                 'mode': mode, 'time': int(status_path.stat().st_mtime)})
            return sorted(jobs, key=lambda item: item['time'], reverse=True)[:60]
        return self.safe(run)

    def browse_dataset(self, kind: str = 'file') -> dict:
        def run():
            if self.role != 'host' or not self.root:
                raise RuntimeError('Choose the Host computer to select a dataset.')
            data_dir = (self.root / 'data').resolve()
            if not data_dir.is_dir():
                raise RuntimeError('No data folder found in the Host project.')
            # The picker never copies personal files into a training folder without consent.
            from tkinter import Tk, filedialog
            ui = Tk()
            ui.withdraw()
            try:
                if kind == 'folder':
                    result = filedialog.askdirectory(initialdir=str(data_dir), title='Select dataset folder')
                elif kind == 'file':
                    result = filedialog.askopenfilename(initialdir=str(data_dir), title='Select dataset file')
                else:
                    raise ValueError('Unknown picker type')
            finally:
                ui.destroy()
            if not result:
                return {'cancelled': True}
            picked = Path(result).resolve()
            if not picked.is_relative_to(data_dir) or picked == data_dir or picked.is_symlink():
                raise ValueError('Choose a dataset inside the Host data folder.')
            return {'cancelled': False, 'path': picked.relative_to(self.root).as_posix()}
        return self.safe(run)

    def open_terminal(self) -> dict:
        def run():
            if sys.platform != 'win32':
                raise RuntimeError('The full terminal requires Windows.')
            if not self.root:
                raise RuntimeError('Select your project folder in Settings.')
            script = self.root / 'studio' / 'GPU-Terminal.ps1'
            if not script.is_file():
                raise RuntimeError('Install studio/GPU-Terminal.ps1 in this project first.')
            exe = Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
            # A real visible console is intentional: commands, stdin and Ctrl+C stay native.
            subprocess.Popen([str(exe), '-NoLogo', '-NoProfile', '-NoExit',
                              '-File', str(script), '-Root', str(self.root), '-HostIP', self.host,
                              '-Role', self.role], cwd=str(self.root),
                             creationflags=getattr(subprocess, 'CREATE_NEW_CONSOLE', 0))
            return {'message': 'PowerShell launch requested. It runs locally and remains open independently of this app.'}
        return self.safe(run)

    def cluster_gpu_status(self) -> dict:
        def run():
            from .gpu_metrics import gpu_snapshot
            nodes=[{'name':socket.gethostname(), 'address':'This PC', **gpu_snapshot()}]
            error=None
            if self.role == 'host':
                try:
                    workers=self._host_api('/api/cluster/workers',authenticated=True).get('workers',[])
                    token=self._get_token()
                    from concurrent.futures import ThreadPoolExecutor
                    def sample(worker):
                        node={'name':str(worker.get('name','Worker')), 'address':str(worker.get('ip','')), 'gpus':[], 'sampled_at':None, 'error':None}
                        try:
                            ip=valid_private_ip(worker['ip'])
                            if int(worker.get('port',WORKER_PORT)) != WORKER_PORT:
                                raise ValueError('Unsupported telemetry port')
                            data=api_json(f'http://{ip}:{WORKER_PORT}/api/p19/gpus',token=token,timeout=4)
                            if not isinstance(data,dict) or not isinstance(data.get('gpus'),list):
                                raise ValueError('Invalid telemetry')
                            node.update(gpus=data['gpus'],sampled_at=data.get('sampled_at'),error=data.get('error'))
                        except Exception:
                            node['error']='Metrics unavailable: check worker update, connection and credentials'
                        return node
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        nodes.extend(pool.map(sample,workers[:24]))
                except Exception:
                    error='Approved worker list unavailable; check controller and credentials'
            else:
                error='Open this panel on the Host to see approved remote GPUs'
            return {'nodes':nodes,'error':error}
        return self.safe(run)

    def start_services(self) -> dict:
        def run():
            if sys.platform != 'win32':
                raise RuntimeError('This startup helper is Windows-only.')
            task = 'DGPU-Phase20-Agent' if self.role == 'worker' else 'DGPU-AutoHeal'
            if self.role == 'host' and tcp_open(self.host, HOST_PORT):
                return {'message': 'Controller is already running.'}
            if self.role == 'worker' and tcp_open('127.0.0.1', AGENT_PORT):
                return {'message': 'Join Agent is already running.'}
            proc = subprocess.run(['schtasks.exe', '/Run', '/TN', task],
                                  capture_output=True, text=True, timeout=8,
                                  creationflags=CREATE_NO_WINDOW)
            if proc.returncode:
                raise RuntimeError(f'Could not start {task}. Install the project background task first. ({proc.stderr.strip()[:120]})')
            return {'message': 'Startup request sent to ' + task + '. Check status again shortly.'}
        return self.safe(run)
