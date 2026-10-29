"""Phase 19 coordinator: synchronous local-SGD rounds, weighted model averaging.

This is FedAvg-style distributed data parallel training, NOT pipelined model
parallelism, all-reduce DDP, FSDP or unified VRAM. It requires identical model
implementation across machines and transfers bounded dataset snapshots.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import ipaddress
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile

import torch

from studio_zoo.models import CATALOG
from studio_zoo.train import dataset_and_info
from .core import (SUPPORTED, dataset_fingerprint, model_from_info,
                   deserialize_state, local_epoch, serialize_state, weighted_average)

ROOT = Path(__file__).resolve().parents[1]
MAX_DATASET_BYTES = 48 * 1024 * 1024


def worker_url(spec: str) -> str:
    # IP literals only; prevent SSRF to DNS names/public Internet and token leak.
    spec = spec.strip().removeprefix('http://')
    if '/' in spec or '@' in spec or '?' in spec or '#' in spec:
        raise ValueError('Only private IP:port worker addresses are supported')
    host, _, port_text = spec.partition(':')
    try:
        ip = ipaddress.ip_address(host)
        port = int(port_text)
    except ValueError as exc:
        raise ValueError('Expected worker IPv4:port') from exc
    if ip.version != 4 or not (ip in ipaddress.ip_network('10.0.0.0/8') or ip in ipaddress.ip_network('172.16.0.0/12') or ip in ipaddress.ip_network('192.168.0.0/16') or ip.is_loopback) or not 1 <= port <= 65535:
        raise ValueError('Worker must be an RFC1918 or loopback IPv4 address')
    return f'http://{host}:{port}'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise RuntimeError('Worker redirects are forbidden')


_OPENER = urllib.request.build_opener(NoRedirect())


def request_json(url: str, path: str, token: str, payload=None, timeout=3600):
    body = None if payload is None else json.dumps(payload).encode('utf-8')
    headers = {'Authorization': 'Bearer ' + token}
    if body is not None:
        headers['Content-Type'] = 'application/json'
    request = urllib.request.Request(url + path, data=body, headers=headers,
                                     method='POST' if body is not None else 'GET')
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        message = exc.read(1200).decode('utf-8', errors='replace')
        raise RuntimeError(f'Worker {url} HTTP {exc.code}: {message}') from exc


def bundle_dataset(source: Path):
    """Archive an explicitly allowed local dataset; no hidden files or symlinks."""
    if not source.exists() or source.is_symlink():
        raise ValueError('Dataset missing/symlink')
    if source.is_dir():
        names = sorted(p for p in source.rglob('*') if p.is_file())
        if not names or len(names) > 10000:
            raise ValueError('Invalid number of dataset files')
        pairs = [(p, 'dataset/' + p.relative_to(source).as_posix()) for p in names]
        entry = 'dataset'
    else:
        pairs = [(source, 'dataset/' + source.name)]
        entry = 'dataset/' + source.name
    if any(p.is_symlink() for p, _ in pairs):
        raise ValueError('Dataset contains symlinks')
    if sum(p.stat().st_size for p, _ in pairs) > 90 * 1024 * 1024:
        raise ValueError('Dataset is larger than supported 90 MiB uncompressed limit')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for p, name in pairs:
            archive.write(p, name)
            if buf.tell() > MAX_DATASET_BYTES:
                raise ValueError('Compressed dataset exceeds 48 MiB limit')
    if buf.tell() > MAX_DATASET_BYTES:
        raise ValueError('Compressed dataset exceeds 48 MiB limit')
    return entry, base64.b64encode(buf.getvalue()).decode('ascii')


def save_checkpoint(output: Path, state: dict, model: str, data_info: dict,
                    epoch: int, history: list, workers: list):
    output.mkdir(parents=True, exist_ok=True)
    # Save tensor-only state plus locally generated metadata. Only load trusted checkpoints.
    checkpoint = {'model_key': model, 'state_dict': state,
                  'data_info': data_info, 'epoch': epoch, 'history': history,
                  'algorithm': 'local_sgd_weighted_average', 'workers': workers}
    temporary = output / 'model.tmp'
    torch.save(checkpoint, temporary)
    temporary.replace(output / 'model.pt')
    (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')


def run(*, model: str, dataset: Path, workers: list[str], epochs: int, batch_size: int,
        learning_rate: float, output: Path, device: str = 'cuda',
        token: str | None = None, resume: bool = False):
    if model not in SUPPORTED:
        raise ValueError('Unsupported distributed preset (LoRA and YOLO excluded)')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Controller CUDA unavailable')
    if not 1 <= epochs <= 100 or not 1 <= batch_size <= 64 or not 0 < learning_rate <= 0.05:
        raise ValueError('Invalid epochs/batch/learning rate')
    if not dataset.exists() or dataset.is_symlink():
        raise ValueError('Dataset not found')
    if not workers or len(workers) > 24:
        raise ValueError('Need 1..24 remote GPU workers')
    remotes = [worker_url(x) for x in workers]
    if len(set(remotes)) != len(remotes):
        raise ValueError('Duplicate worker URL')
    token = token or os.environ.get('DGPU_TOKEN')
    if not token:
        raise ValueError('DGPU_TOKEN missing')
    ds, info = dataset_and_info(model, dataset)
    world = len(remotes) + 1
    if len(ds) < world:
        raise ValueError(f'Dataset requires at least {world} samples')
    torch.manual_seed(42)
    random.seed(42)
    state = {k: v.cpu().detach().clone() for k, v in model_from_info(model, info).state_dict().items()}
    history = []
    start_epoch = 1
    if resume:
        saved = torch.load(output / 'model.pt', map_location='cpu', weights_only=True)
        if (saved['model_key'] != model or saved['data_info'] != info
            or saved['workers'] != remotes):
            raise ValueError('Checkpoint and requested model/data/workers mismatch')
        state = saved['state_dict']
        history = saved['history']
        start_epoch = saved['epoch'] + 1
    fingerprint = dataset_fingerprint(dataset)
    entry, archive_b64 = bundle_dataset(dataset)
    for address in remotes:
        health = request_json(address, '/api/p19/health', token, timeout=10)
        if health.get('service') != 'phase19' or (device == 'cuda' and not health.get('cuda')):
            raise RuntimeError(f'Wrong/CPU-only remote worker at {address}: {health}')
    # One immutable dataset/session per run ID. No arbitrary remote file execution.
    session_id = os.urandom(8).hex()
    for address in remotes:
        request_json(address, '/api/p19/session', token, {
            'session_id': session_id, 'model': model,
            'dataset_entry': entry, 'dataset_b64': archive_b64,
            'fingerprint': fingerprint}, timeout=300)
    print(f'P19 algorithm=local-SGD+weighted-averaging nodes={world} '
          f'samples={len(ds)} model={model} device={device}', flush=True)
    print('Nodes:', ['controller'] + remotes, flush=True)
    for epoch in range(start_epoch, epochs + 1):
        payload = serialize_state(state)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=len(remotes)) as executor:
            futures = [executor.submit(request_json, address, '/api/p19/round', token, {
                'session_id': session_id, 'model': model, 'state_b64': payload,
                'epoch': epoch, 'rank': rank, 'world_size': world,
                'batch_size': batch_size, 'learning_rate': learning_rate
            }, 3600) for rank, address in enumerate(remotes, 1)]
            local_result = local_epoch(model, dataset, state, rank=0,
                                       world_size=world, epoch=epoch,
                                       batch_size=batch_size, lr=learning_rate,
                                       device=device)
            results = [local_result]
            for rank, future in enumerate(futures, 1):
                remote_result = future.result()
                if remote_result['fingerprint'] != fingerprint:
                    raise RuntimeError('Worker dataset mismatch')
                if remote_result['rank'] != rank:
                    raise RuntimeError('Worker rank mismatch')
                if device == 'cuda' and remote_result['device'] != 'cuda':
                    raise RuntimeError('Worker fell back to CPU')
                results.append({**remote_result, 'state': deserialize_state(remote_result['state_b64'])})
        state = weighted_average(results)
        average_loss = sum(r['loss'] * r['samples'] for r in results) / len(ds)
        history.append({'epoch': epoch, 'loss': average_loss, 'samples': len(ds),
                        'participants': len(results), 'elapsed_s': round(time.time() - t0, 2),
                        'devices': [r['device'] for r in results]})
        save_checkpoint(output, state, model, info, epoch, history, remotes)
        print(f'Epoch {epoch}/{epochs} weighted_loss={average_loss:.6f} '
              f'nodes={len(results)} elapsed={history[-1]["elapsed_s"]}s', flush=True)
    print('P19 COMPLETE', output / 'model.pt', flush=True)
    return {'epochs': len(history), 'checkpoint': str(output / 'model.pt'),
            'history': history}


def main():
    parser = argparse.ArgumentParser(description='Distributed-GPU Phase19 local-SGD trainer')
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--workers', default='192.168.50.2:45420')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output', default='.dgpu/phase19_runs/manual')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(model=args.model, dataset=Path(args.dataset),
        workers=[x.strip() for x in args.workers.split(';') if x.strip()],
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
        output=Path(args.output), device=args.device, resume=args.resume)


if __name__ == '__main__':
    main()
