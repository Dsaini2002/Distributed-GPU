"""One-device training runner for model-zoo presets.

This does not use the legacy two-GPU Linear/ReLU partition engine.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .models import CATALOG, build_model


def _image_folder(path: Path, size: int):
    try:
        from torchvision import datasets, transforms
    except Exception as exc:
        raise RuntimeError('torchvision is required for image datasets') from exc
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    ds = datasets.ImageFolder(str(path), transform=transform)
    if len(ds.classes) < 2:
        raise ValueError('ImageFolder requires >=2 class directories containing images')
    return ds, {'num_classes': len(ds.classes), 'labels': ds.classes}


class SegmentationDataset(Dataset):
    def __init__(self, folder: Path, size: int):
        from PIL import Image
        self.Image = Image
        self.size = size
        self.pairs = []
        root = folder / 'images'
        masks = folder / 'masks'
        for p in sorted(root.glob('*')):
            if p.is_file() and (masks / p.name).exists():
                self.pairs.append((p, masks / p.name))
        if not self.pairs:
            raise ValueError('Expected matching image/mask filenames in images/ and masks/')
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        import numpy as np
        img, mask = self.pairs[i]
        a = self.Image.open(img).convert('RGB').resize((self.size, self.size))
        b = self.Image.open(mask).convert('L').resize((self.size, self.size), resample=self.Image.Resampling.NEAREST)
        return torch.from_numpy(np.asarray(a).copy()).permute(2,0,1).float() / 255, \
               (torch.from_numpy(np.asarray(b).copy()).float() / 255 >= 0.5).float().unsqueeze(0)


def _tabular(path: Path, classification: bool):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    if len(rows) < 2: raise ValueError('CSV needs a header and at least one numeric data row')
    cols = rows[0]
    if len(cols) < 2: raise ValueError('CSV needs feature columns plus a target column')
    xs, ys = [], []
    for row in rows[1:]:
        if not row or all(not c.strip() for c in row): continue
        if len(row) != len(cols): raise ValueError('Inconsistent number of CSV columns')
        xs.append([float(c) for c in row[:-1]])
        ys.append(int(row[-1]) if classification else float(row[-1]))
    if not xs: raise ValueError('CSV has no samples')
    x = torch.tensor(xs, dtype=torch.float32)
    if classification:
        classes = sorted(set(ys))
        if len(classes) < 2: raise ValueError('Classification needs >=2 distinct classes')
        yy = torch.tensor([classes.index(v) for v in ys], dtype=torch.long)
        return TensorDataset(x, yy), {'num_inputs': x.shape[1], 'num_classes': len(classes), 'labels': classes}
    yy = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)
    return TensorDataset(x, yy), {'num_inputs': x.shape[1]}


def _tokenize(text, max_len=128, vocab=4096):
    tokens = []
    for word in text.lower().split()[:max_len]:
        hashed = int.from_bytes(hashlib.sha256(word.encode('utf-8')).digest()[:4], 'big')
        tokens.append(2 + (hashed % (vocab - 2)))
    # Keep one nonpad id for empty text so attention mask is not entirely padded.
    tokens = tokens or [1]
    return tokens + [0] * (max_len - len(tokens))


def _text(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not {'text', 'label'}.issubset(reader.fieldnames or []):
            raise ValueError('Text CSV must contain text,label header')
        rows = [(r['text'], r['label']) for r in reader if r['text'] and r['label']]
    if not rows: raise ValueError('Text CSV has no examples')
    labels = sorted({label for _, label in rows})
    if len(labels) < 2: raise ValueError('Text classification needs >=2 different labels')
    xx = torch.tensor([_tokenize(t) for t, _ in rows], dtype=torch.long)
    yy = torch.tensor([labels.index(y) for _, y in rows], dtype=torch.long)
    return TensorDataset(xx, yy), {'num_classes': len(labels), 'labels': labels, 'tokenizer': 'sha256-hash-v1'}


class LMCorpus(Dataset):
    def __init__(self, path: Path, block=64):
        corpus = path.read_text(encoding='utf-8')
        if len(corpus) < block + 2:
            raise ValueError(f'LM text must have at least {block+2} characters')
        self.chars = sorted(set(corpus))
        self.ids = torch.tensor([self.chars.index(c) for c in corpus], dtype=torch.long)
        self.block = block
    def __len__(self): return len(self.ids) - self.block
    def __getitem__(self, i):
        return self.ids[i:i+self.block], self.ids[i+1:i+self.block+1]


def _forecast(path: Path, window=32):
    vals = []
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        for row in csv.reader(f):
            if not row or not row[0].strip(): continue
            try: vals.append(float(row[0]))
            except ValueError:
                if vals: raise ValueError('Time series CSV has nonnumeric value')
                # optional first-row header
    if len(vals) < window + 2: raise ValueError(f'Forecast CSV needs at least {window+2} samples')
    x = torch.tensor([vals[i:i+window] for i in range(len(vals)-window)], dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor([vals[i+window] for i in range(len(vals)-window)], dtype=torch.float32).unsqueeze(-1)
    return TensorDataset(x,y), {'window': window}


def dataset_and_info(key, data: Path):
    preset = CATALOG[key]
    if preset.task == 'regression': return _tabular(data, False)
    if preset.task == 'classification': return _tabular(data, True)
    if preset.task == 'image_classification': return _image_folder(data, preset.input_size)
    if preset.task == 'binary_segmentation': return SegmentationDataset(data, preset.input_size), {}
    if preset.task == 'text_classification': return _text(data)
    if preset.task == 'causal_lm':
        ds = LMCorpus(data)
        return ds, {'vocab_size': len(ds.chars), 'chars': ds.chars, 'block': ds.block}
    if preset.task == 'forecasting': return _forecast(data)
    raise ValueError(preset.task)


def train(model_key: str, data: Path, output: Path, epochs=1, batch_size=None, lr=0.001, device='auto'):
    if model_key not in CATALOG: raise ValueError('Unknown model key')
    if not data.exists(): raise FileNotFoundError(f'Dataset not found: {data}')
    if not 1 <= epochs <= 100: raise ValueError('epochs must be 1..100')
    if not 0 < lr <= 0.1: raise ValueError('learning rate must be 0 < lr <= 0.1')
    torch.manual_seed(42)
    random.seed(42)
    preset = CATALOG[model_key]
    if preset.task == 'pretrained_lm_lora':
        return _train_pretrained_lora(data, output, epochs, batch_size or 1, lr, selected)
    if preset.task == 'object_detection':
        return _train_yolo(data, output, epochs, batch_size or 2, selected)
    size = batch_size if batch_size is not None else preset.suggested_batch
    if not 1 <= size <= 256: raise ValueError('batch size must be 1..256')
    selected = 'cuda' if device == 'auto' and torch.cuda.is_available() else ('cpu' if device == 'auto' else device)
    if selected == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA is not available')
    ds, info = dataset_and_info(model_key, data)
    if not len(ds): raise ValueError('Empty dataset')
    loader = DataLoader(ds, batch_size=size, shuffle=True, num_workers=0)
    network = build_model(model_key,
                          num_inputs=info.get('num_inputs', 4),
                          num_classes=info.get('num_classes', 2),
                          vocab_size=info.get('vocab_size', 128)).to(selected)
    opt = torch.optim.AdamW(network.parameters(), lr=lr)
    if preset.task in ('regression', 'forecasting'): criterion = nn.MSELoss()
    elif preset.task == 'binary_segmentation': criterion = nn.BCEWithLogitsLoss()
    else: criterion = nn.CrossEntropyLoss()
    output.mkdir(parents=True, exist_ok=True)
    history = []
    print(f'Model={model_key} device={selected} samples={len(ds)} batch={size}', flush=True)
    for ep in range(1, epochs+1):
        network.train()
        total, count = 0.0, 0
        for x, y in loader:
            x, y = x.to(selected), y.to(selected)
            opt.zero_grad(set_to_none=True)
            out = network(x)
            if preset.task == 'causal_lm':
                loss = criterion(out.reshape(-1, out.shape[-1]), y.reshape(-1))
            else:
                loss = criterion(out, y)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
            count += len(x)
        epoch_loss = total / count
        history.append({'epoch': ep, 'loss': epoch_loss})
        print(f'Epoch {ep}/{epochs} loss={epoch_loss:.6f}', flush=True)
    # Model and optimizer metadata; only load checkpoints from trusted local runs.
    checkpoint = {'model_key': model_key, 'state_dict': network.cpu().state_dict(),
                  'data_info': info, 'history': history, 'dataset': str(data),
                  'training_device': selected}
    torch.save(checkpoint, output / 'model.pt')
    (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    print(f'COMPLETE checkpoint={output / "model.pt"}', flush=True)
    return checkpoint



def _train_pretrained_lora(data: Path, output: Path, epochs: int, batch_size: int, lr: float, device: str):
    """Experimental: true pretrained GPT-2 family LoRA adapter; local single GPU."""
    if device != 'cuda':
        raise RuntimeError('This experimental pretrained LoRA preset requires local CUDA')
    if batch_size > 2:
        raise ValueError('On 4GB cards use batch size 1 (maximum 2)')
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError('Install optional: python -m pip install transformers peft accelerate') from exc
    corpus = data.read_text(encoding='utf-8')[:250000]
    print('Downloading/loading pretrained distilgpt2 if needed...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    pretrained = AutoModelForCausalLM.from_pretrained('distilgpt2')
    pretrained.config.use_cache = False
    pretrained.gradient_checkpointing_enable()
    config = LoraConfig(r=4, lora_alpha=8, target_modules=['c_attn'],
                        lora_dropout=0.0, fan_in_fan_out=True,
                        bias='none', task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(pretrained, config).to(device)
    token_ids = tokenizer.encode(corpus, add_special_tokens=False)
    block = 64
    if len(token_ids) < block + 2:
        raise ValueError('Text corpus is too short; add 128+ tokens')
    # Small blocks, deterministic stepping, no automatic low-VRAM guarantee.
    chunks = [torch.tensor(token_ids[i:i+block], dtype=torch.long)
              for i in range(0, len(token_ids)-block+1, block)]
    loader = DataLoader(chunks, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    print('Pretrained distilgpt2 + LoRA on local GPU only', flush=True)
    for epoch in range(1, epochs+1):
        model.train()
        total = count = 0
        for ids in loader:
            ids = ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
            optimizer.step()
            total += loss.item() * ids.size(0)
            count += ids.size(0)
        mean = total / max(count, 1)
        history.append({'epoch': epoch, 'loss': mean})
        print(f'Epoch {epoch}/{epochs} loss={mean:.6f}', flush=True)
    model.save_pretrained(output / 'lora_adapter')
    tokenizer.save_pretrained(output / 'lora_adapter')
    torch.save({'model_key': 'distilgpt2_lora', 'adapter_dir': str(output / 'lora_adapter'),
                'history': history, 'base_model': 'distilgpt2'}, output / 'model.pt')
    (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    print('COMPLETE LoRA adapter in', output / 'lora_adapter', flush=True)
    return history


def _train_yolo(data: Path, output: Path, epochs: int, batch_size: int, device: str):
    """Experimental: official YOLO training workflow with YOLO-format dataset YAML."""
    if data.suffix.lower() not in ('.yaml', '.yml'):
        raise ValueError('Object detection expects a YOLO-format .yaml dataset file')
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('Install optional: python -m pip install ultralytics') from exc
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO('yolo11n.pt')
    model.train(data=str(data), epochs=epochs, imgsz=320,
                batch=batch_size, workers=0, device=(0 if device == 'cuda' else 'cpu'),
                project=str(output), name='yolo', exist_ok=True)
    best = output / 'yolo' / 'weights' / 'best.pt'
    if not best.exists():
        raise RuntimeError('Training ended without YOLO best.pt checkpoint')
    import shutil
    shutil.copy2(best, output / 'model.pt')
    print('COMPLETE YOLO checkpoint=', output / 'model.pt', flush=True)
    return {'yolo_weights': str(best)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=sorted(CATALOG))
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
    args = parser.parse_args()
    train(args.model, Path(args.dataset).resolve(), Path(args.output).resolve(),
          epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)


if __name__ == '__main__':
    main()
