"""Built-in, trainable architecture presets. No arbitrary file execution or downloads."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Preset:
    key: str
    family: str
    task: str
    dataset: str
    description: str
    input_size: int = 0
    suggested_batch: int = 4


PRESETS = [
    Preset('tabular_mlp_regression', 'Tabular', 'regression', 'CSV: numeric features, last column = numeric target', 'Fully connected regressor', suggested_batch=32),
    Preset('tabular_mlp_classification', 'Tabular', 'classification', 'CSV: numeric features, last column = integer class', 'Fully connected classifier', suggested_batch=32),
    Preset('image_cnn', 'Computer Vision', 'image_classification', 'ImageFolder: dataset/class_name/photo.jpg', 'Small CNN trained from scratch', 64, 8),
    Preset('resnet18', 'Computer Vision', 'image_classification', 'ImageFolder: dataset/class_name/photo.jpg', 'Torchvision ResNet-18 from scratch', 96, 4),
    Preset('mobilenet_v3_small', 'Computer Vision', 'image_classification', 'ImageFolder: dataset/class_name/photo.jpg', 'Torchvision MobileNetV3 Small from scratch', 96, 4),
    Preset('efficientnet_b0', 'Computer Vision', 'image_classification', 'ImageFolder: dataset/class_name/photo.jpg', 'Torchvision EfficientNet-B0 from scratch; may need low batch', 128, 2),
    Preset('tiny_vit', 'Computer Vision', 'image_classification', 'ImageFolder: dataset/class_name/photo.jpg', 'Tiny Vision Transformer', 64, 4),
    Preset('unet_segmentation', 'Computer Vision', 'binary_segmentation', 'Folder: images/file.png and masks/file.png (matching names)', 'Small binary segmentation U-Net', 64, 4),
    Preset('audio_spectrogram_cnn', 'Audio', 'image_classification', 'ImageFolder of spectrogram PNG/JPG images by class', 'CNN trained on existing spectrogram images (not WAV)', 64, 8),
    Preset('text_gru', 'NLP', 'text_classification', 'CSV: text,label columns; labels may be strings', 'GRU text classifier with hashed token vocabulary', suggested_batch=16),
    Preset('text_transformer', 'NLP', 'text_classification', 'CSV: text,label columns; labels may be strings', 'Small Transformer encoder text classifier', suggested_batch=8),
    Preset('tiny_transformer_lm', 'Language Models', 'causal_lm', 'UTF-8 .txt corpus; character-level next-token training', 'Small causal Transformer language model from scratch; NOT a pretrained LLM', suggested_batch=8),
    Preset('time_series_lstm', 'Time Series', 'forecasting', 'CSV: one numeric value per row (header optional)', 'LSTM next-step forecasting', suggested_batch=32),
    Preset('distilgpt2_lora', 'Pretrained LLM', 'pretrained_lm_lora', 'UTF-8 .txt corpus (>128 tokens); downloads distilgpt2', 'Experimental LoRA fine-tuning of pretrained distilgpt2; transformers + peft required, may OOM on 4GB', suggested_batch=1),
    Preset('yolo11n_detection', 'Object Detection', 'object_detection', 'YOLO dataset YAML inside data/ folder; downloads yolo11n.pt', 'Experimental Ultralytics YOLO11n detector fine-tuning; ultralytics required', suggested_batch=2),
]
CATALOG = {p.key: p for p in PRESETS}


class TabularMLP(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, outputs))
    def forward(self, x): return self.net(x)


class SmallCNN(nn.Module):
    def __init__(self, outputs: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(64, outputs)
    def forward(self, x): return self.head(self.features(x).flatten(1))


class TinyViT(nn.Module):
    def __init__(self, outputs: int, img_size: int = 64, patch: int = 8, embed: int = 96):
        super().__init__()
        n = (img_size // patch) ** 2
        self.patch_embed = nn.Conv2d(3, embed, patch, stride=patch)
        self.positional = nn.Parameter(torch.zeros(1, n, embed))
        layer = nn.TransformerEncoderLayer(embed, 4, dim_feedforward=embed*2, batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(embed, outputs)
    def forward(self, x):
        z = self.patch_embed(x).flatten(2).transpose(1, 2)
        z = self.encoder(z + self.positional[:, :z.size(1)])
        return self.head(z.mean(1))


class ConvBlock(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1), nn.ReLU())
    def forward(self, x): return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = ConvBlock(3, 16)
        self.b = ConvBlock(16, 32)
        self.c = ConvBlock(32, 64)
        self.up2 = ConvBlock(64+32, 32)
        self.up1 = ConvBlock(32+16, 16)
        self.head = nn.Conv2d(16, 1, 1)
    def forward(self, x):
        a = self.a(x)
        b = self.b(F.max_pool2d(a, 2))
        c = self.c(F.max_pool2d(b, 2))
        d = self.up2(torch.cat([F.interpolate(c, size=b.shape[-2:], mode='nearest'), b], dim=1))
        e = self.up1(torch.cat([F.interpolate(d, size=a.shape[-2:], mode='nearest'), a], dim=1))
        return self.head(e)


class TextGRU(nn.Module):
    def __init__(self, outputs: int, vocab=4096, dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim, padding_idx=0)
        self.gru = nn.GRU(dim, 96, batch_first=True)
        self.head = nn.Linear(96, outputs)
    def forward(self, x):
        _, h = self.gru(self.embedding(x))
        return self.head(h[-1])


class TextTransformer(nn.Module):
    def __init__(self, outputs: int, vocab=4096, dim=64, max_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim, padding_idx=0)
        self.positional = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(dim, 4, dim_feedforward=dim*2, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(dim, outputs)
    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device)
        z = self.enc(self.embedding(x) + self.positional(pos), src_key_padding_mask=(x == 0))
        valid = (x != 0).unsqueeze(-1)
        return self.head((z * valid).sum(1) / valid.sum(1).clamp(min=1))


class TinyCausalTransformer(nn.Module):
    def __init__(self, vocab: int, block=64, dim=64):
        super().__init__()
        self.token = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(block, dim)
        layer = nn.TransformerEncoderLayer(dim, 4, dim_feedforward=dim*4, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(dim, vocab)
    def forward(self, ids):
        t = ids.size(1)
        positions = torch.arange(t, device=ids.device)
        mask = torch.triu(torch.ones(t, t, device=ids.device, dtype=torch.bool), diagonal=1)
        z = self.enc(self.token(ids) + self.pos(positions), mask=mask)
        return self.head(z)


class TimeSeriesLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 48, batch_first=True)
        self.fc = nn.Linear(48, 1)
    def forward(self, x):
        z, _ = self.lstm(x)
        return self.fc(z[:, -1])


def build_model(key: str, *, num_inputs=4, num_classes=2, vocab_size=128) -> nn.Module:
    if key not in CATALOG:
        raise ValueError(f'Unknown model: {key}')
    if key.startswith('tabular_mlp'):
        return TabularMLP(num_inputs, 1 if key.endswith('regression') else num_classes)
    if key in ('image_cnn', 'audio_spectrogram_cnn'):
        return SmallCNN(num_classes)
    if key == 'tiny_vit': return TinyViT(num_classes)
    if key == 'unet_segmentation': return TinyUNet()
    if key == 'text_gru': return TextGRU(num_classes)
    if key == 'text_transformer': return TextTransformer(num_classes)
    if key == 'tiny_transformer_lm': return TinyCausalTransformer(vocab_size)
    if key == 'time_series_lstm': return TimeSeriesLSTM()
    if key in ('resnet18', 'mobilenet_v3_small', 'efficientnet_b0'):
        try:
            import torchvision.models as tvm
        except Exception as exc:
            raise RuntimeError('Install matching torchvision version for the installed torch') from exc
        if key == 'resnet18': return tvm.resnet18(weights=None, num_classes=num_classes)
        if key == 'mobilenet_v3_small': return tvm.mobilenet_v3_small(weights=None, num_classes=num_classes)
        return tvm.efficientnet_b0(weights=None, num_classes=num_classes)
    raise AssertionError(key)


def catalog_dicts() -> list[dict[str, Any]]:
    return [asdict(p) for p in PRESETS]
