"""Training and serialization building blocks, shared by controller and worker."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from studio_zoo.models import CATALOG, build_model
from studio_zoo.train import dataset_and_info

# All models here use build_model and dataset_and_info: optional pretrained
# LoRA and Ultralytics use separate trainers and cannot take part in FedAvg.
SUPPORTED = frozenset(k for k, v in CATALOG.items()
                      if v.task not in ('pretrained_lm_lora', 'object_detection'))
MAX_STATE_BYTES = 90 * 1024 * 1024


def serialize_state(state: dict[str, torch.Tensor]) -> str:
    buf = io.BytesIO()
    cpu = {k: t.detach().cpu().contiguous() for k, t in state.items()}
    torch.save(cpu, buf)
    raw = buf.getvalue()
    if len(raw) > MAX_STATE_BYTES:
        raise ValueError('Model state exceeds 90 MiB LAN transport limit')
    return base64.b64encode(raw).decode('ascii')


def deserialize_state(payload: str) -> dict[str, torch.Tensor]:
    if len(payload) > MAX_STATE_BYTES * 4 // 3 + 1024 * 1024:
        raise ValueError('Model state payload is too large')
    raw = base64.b64decode(payload, validate=True)
    if len(raw) > MAX_STATE_BYTES:
        raise ValueError('Model state exceeds limit')
    result = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    if not isinstance(result, dict) or not result or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in result.items()
    ):
        raise ValueError('Expected tensor-only state_dict')
    return result


def model_from_info(key: str, info: dict):
    return build_model(key,
                       num_inputs=info.get('num_inputs', 4),
                       num_classes=info.get('num_classes', 2),
                       vocab_size=info.get('vocab_size', 128))


def local_epoch(key: str, data: Path, start_state: dict, *, rank: int,
                world_size: int, epoch: int, batch_size: int, lr: float,
                device: str = 'cuda') -> dict:
    """Train precisely one shard and one local epoch. No global gradient all-reduce.

    Each round starts from the exact same global state on each participant.
    """
    if key not in SUPPORTED:
        raise ValueError(f'Unsupported distributed model: {key}')
    if world_size < 2 or not 0 <= rank < world_size:
        raise ValueError('Invalid rank or world size')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required on this node')
    if not 1 <= batch_size <= 64 or not 0 < lr <= 0.05:
        raise ValueError('Invalid batch size or learning rate')
    torch.manual_seed(42 + epoch)
    random.seed(42 + epoch)
    dataset, info = dataset_and_info(key, data)
    indices = list(range(rank, len(dataset), world_size))
    if not indices:
        raise ValueError(f'Rank {rank} has no dataset samples: {len(dataset)} < {world_size}')
    net = model_from_info(key, info)
    net.load_state_dict(start_state, strict=True)
    net.to(device).train()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                        shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(811 + epoch + rank))
    criterion = (nn.MSELoss() if CATALOG[key].task in ('regression','forecasting')
                 else nn.BCEWithLogitsLoss() if CATALOG[key].task == 'binary_segmentation'
                 else nn.CrossEntropyLoss())
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = net(x)
        if CATALOG[key].task == 'causal_lm':
            loss = criterion(prediction.reshape(-1, prediction.shape[-1]), y.reshape(-1))
        else:
            loss = criterion(prediction, y)
        if not torch.isfinite(loss).item():
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        optimizer.step()
        total += loss.item() * len(x)
    final = {k: v.detach().to('cpu').contiguous() for k, v in net.state_dict().items()}
    # Move weights off GPU before returning to lower peak memory.
    net.cpu()
    return {'state': final, 'samples': len(indices), 'loss': total / len(indices),
            'device': device, 'rank': rank, 'info': info}


def weighted_average(results: list[dict]) -> dict[str, torch.Tensor]:
    if not results or any(r['samples'] <= 0 for r in results):
        raise ValueError('All participants must have positive shard size')
    names = list(results[0]['state'])
    base = results[0]['state']
    for r in results[1:]:
        if list(r['state']) != names:
            raise ValueError('Model parameters mismatch between nodes')
    count = sum(r['samples'] for r in results)
    averaged = {}
    for k in names:
        tensors = [r['state'][k] for r in results]
        if any(t.shape != tensors[0].shape or t.dtype != tensors[0].dtype for t in tensors):
            raise ValueError(f'Tensor mismatch: {k}')
        if torch.is_floating_point(tensors[0]):
            accumulator = torch.zeros_like(tensors[0], dtype=torch.float64, device='cpu')
            for r, v in zip(results, tensors):
                accumulator += v.to(torch.float64) * (r['samples'] / count)
            averaged[k] = accumulator.to(tensors[0].dtype)
        else:
            # BatchNorm num_batches_tracked is a counter, not a trainable weight.
            averaged[k] = torch.stack(tensors).max(dim=0).values
    return averaged


def dataset_fingerprint(data: Path) -> str:
    """Stable digest of logical dataset contents, independent of base directory."""
    h = hashlib.sha256()
    paths = sorted(p for p in data.rglob('*') if p.is_file()) if data.is_dir() else [data]
    for p in paths:
        relative = str(p.relative_to(data)) if data.is_dir() else p.name
        h.update(relative.replace('\\', '/').encode('utf-8'))
        h.update(b'\0')
        with p.open('rb') as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
    return h.hexdigest()
