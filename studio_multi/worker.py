"""Phase 19 worker. Run on each remote laptop, bound to its private Ethernet IP."""
import base64
import hashlib
import hmac
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import threading
import zipfile

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
import torch

from . import __version__
from .core import SUPPORTED, deserialize_state, local_epoch, serialize_state, dataset_fingerprint

app = FastAPI(title='DGPU Phase 19 Worker', docs_url=None, redoc_url=None)
ROOT = Path(__file__).resolve().parents[1]
STORAGE = ROOT / '.dgpu' / 'phase19_worker'
LOCK = threading.Lock()
MAX_ZIP = 48 * 1024 * 1024
MAX_EXTRACT = 90 * 1024 * 1024
ID_RE = re.compile(r'^[a-f0-9]{16}$')


def _auth(authorization: str):
    token = os.environ.get('DGPU_TOKEN', '')
    candidate = authorization[7:] if authorization.startswith('Bearer ') else ''
    if not token:
        raise HTTPException(503, 'Worker token not configured')
    if not hmac.compare_digest(token.encode(), candidate.encode()):
        raise HTTPException(401, 'Unauthorized')


class SessionRequest(BaseModel):
    session_id: str
    model: str
    dataset_entry: str
    dataset_b64: str
    fingerprint: str


class RoundRequest(BaseModel):
    session_id: str
    model: str
    state_b64: str
    epoch: int = Field(ge=1, le=100)
    rank: int = Field(ge=1, le=24)
    world_size: int = Field(ge=2, le=25)
    batch_size: int = Field(ge=1, le=64)
    learning_rate: float = Field(gt=0, le=0.05)


def _session_dir(session: str) -> Path:
    if not ID_RE.fullmatch(session):
        raise HTTPException(400, 'Bad session ID')
    return STORAGE / session


def _safe_extract(content: bytes, target: Path):
    if len(content) > MAX_ZIP:
        raise HTTPException(413, 'Dataset archive exceeds 48 MiB limit')
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        entries = archive.infolist()
        if len(entries) > 10000:
            raise HTTPException(413, 'Too many dataset files')
        if sum(x.file_size for x in entries) > MAX_EXTRACT:
            raise HTTPException(413, 'Dataset uncompressed size exceeds 90 MiB')
        for member in entries:
            name = PurePosixPath(member.filename)
            if (name.is_absolute() or not name.parts or '..' in name.parts
                or '\\' in member.filename or ':' in member.filename
                or name.parts[0] != 'dataset'):
                raise HTTPException(400, 'Invalid archive path')
            kind = (member.external_attr >> 16) & 0o170000
            if kind == stat.S_IFLNK:
                raise HTTPException(400, 'Symlink files are not allowed')
            dest = target.joinpath(*name.parts)
            if member.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as inp, dest.open('wb') as out:
                    remaining = MAX_EXTRACT
                    while chunk := inp.read(min(1024 * 1024, remaining + 1)):
                        remaining -= len(chunk)
                        if remaining < 0:
                            raise HTTPException(413, 'Dataset too large')
                        out.write(chunk)


@app.get('/api/p19/health')
def health():
    return {'service': 'phase19', 'version': __version__,
            'cuda': torch.cuda.is_available(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


@app.post('/api/p19/session')
def session(spec: SessionRequest, authorization: str = Header(default='')):
    _auth(authorization)
    with LOCK:
        if spec.model not in SUPPORTED:
            raise HTTPException(400, 'Unsupported model')
        folder = _session_dir(spec.session_id)
        # Refuse ambiguous entries instead of accepting arbitrary file names.
        rel = PurePosixPath(spec.dataset_entry)
        if (not rel.parts or rel.parts[0] != 'dataset' or '..' in rel.parts
            or rel.is_absolute() or '\\' in spec.dataset_entry):
            raise HTTPException(400, 'Bad dataset entry')
        if len(spec.dataset_b64) > MAX_ZIP * 4 // 3 + 1024:
            raise HTTPException(413, 'Archive too large')
        try:
            content = base64.b64decode(spec.dataset_b64, validate=True)
        except ValueError as exc:
            raise HTTPException(400, 'Bad dataset encoding') from exc
        if len(spec.fingerprint) != 64:
            raise HTTPException(400, 'Bad dataset fingerprint')
        if folder.exists():
            manifest_file = folder / 'manifest.json'
            if not manifest_file.exists():
                raise HTTPException(409, 'Incomplete session already exists')
            manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
            if manifest != {'model': spec.model, 'dataset_entry': spec.dataset_entry,
                            'fingerprint': spec.fingerprint}:
                raise HTTPException(409, 'Existing session has different dataset/model')
            return {'ok': True, 'cached': True}
        folder.mkdir(parents=True)
        try:
            _safe_extract(content, folder)
            dataset = folder.joinpath(*rel.parts)
            if not dataset.exists():
                raise HTTPException(400, 'Dataset missing from zip')
            if dataset_fingerprint(dataset) != spec.fingerprint:
                raise HTTPException(400, 'Dataset fingerprint mismatch')
            manifest = {'model': spec.model, 'dataset_entry': spec.dataset_entry,
                        'fingerprint': spec.fingerprint}
            (folder / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        except Exception:
            import shutil
            shutil.rmtree(folder, ignore_errors=True)
            raise
        return {'ok': True, 'cached': False}


@app.post('/api/p19/round')
def train_round(spec: RoundRequest, authorization: str = Header(default='')):
    _auth(authorization)
    if not LOCK.acquire(blocking=False):
        raise HTTPException(409, 'Worker is busy')
    try:
        folder = _session_dir(spec.session_id)
        mf = folder / 'manifest.json'
        if not mf.exists():
            raise HTTPException(404, 'Dataset session not found')
        manifest = json.loads(mf.read_text(encoding='utf-8'))
        if manifest['model'] != spec.model:
            raise HTTPException(400, 'Model mismatch')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device != 'cuda' and os.environ.get('DGPU_P19_ALLOW_CPU') != '1':
            raise HTTPException(503, 'CUDA unavailable on worker')
        state = deserialize_state(spec.state_b64)
        dataset = folder.joinpath(*PurePosixPath(manifest['dataset_entry']).parts)
        result = local_epoch(spec.model, dataset, state,
                             rank=spec.rank, world_size=spec.world_size,
                             epoch=spec.epoch, batch_size=spec.batch_size,
                             lr=spec.learning_rate, device=device)
        return {'state_b64': serialize_state(result['state']),
                'samples': result['samples'], 'loss': result['loss'],
                'device': result['device'], 'rank': result['rank'],
                'fingerprint': manifest['fingerprint']}
    except HTTPException:
        raise
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        LOCK.release()
