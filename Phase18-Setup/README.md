# Distributed-GPU Studio — Phase 18 Model Zoo (optional extension)

**Not a universal distributed engine.** This package adds an optional Model Zoo page and multiple **real, trainable single-controller-GPU** architecture presets to the existing Windows app. It **does not** turn existing two-GPU Linear/ReLU model-parallel training into CNN/LLM distributed training. It never touches the existing `studio.train_engine` and does not alter worker agents, cluster tokens, autoheal or checkpoints.

## Models

- **Tabular:** MLP regression, MLP classification.
- **Images:** custom CNN, ResNet18, MobileNetV3 Small, EfficientNet B0, Tiny Vision Transformer.
- **Segmentation:** binary U-Net.
- **Audio:** CNN with **precomputed spectrogram images** (not WAV ingestion).
- **NLP:** GRU, Transformer Encoder, hashed word tokenizer (not pretrained embeddings).
- **Text generation:** small **character-level** causal Transformer, trained from scratch (not an LLM).
- **Time series:** next-step LSTM.
- **Optional experimental:** pretrained **DistilGPT2 LoRA** fine-tune; **YOLO11n object detector**. The optional models require additional packages and model downloads; neither is verified on the user's 4GB Windows GPUs.

Architecture names in a dropdown do not equate to universal architecture/data support. Extra models require dedicated adapters and tests.

## Important hardware and performance

The two 4GB cards are not a unified 8GB device. This extension trains on **Laptop A's local CUDA GPU only**. On GTX 1650 use small batches (usually 1-4 for pretrained/large models), 64-128px images, no concurrent training with legacy distributed engine. Larger LLMs (e.g. multi-billion parameter models) may be impossible to train within 4GB VRAM even with optimizations. Pretrained DistilGPT2 is a much smaller experimental option.

DDP across two Windows laptops, FSDP, pipeline sharding and LLM cross-machine transport remain **future integrations** and need working training reliability first.

## Installation (Laptop A/controller only)

1. Stop running training jobs. Back up your existing project.
2. Copy `DGPU-Phase18-ModelZoo.zip` to `D:\Distributed-GPU` and run PowerShell:

```powershell
cd D:\Distributed-GPU
Expand-Archive .\DGPU-Phase18-ModelZoo.zip -DestinationPath .\Phase18-Setup -Force
# Activate the Python environment with working torch and fastapi, if needed.
python -c "import torch, fastapi; print(torch.__version__, torch.cuda.is_available())"
# Matching torchvision and Pillow are needed for image presets.
# Refer to PyTorch install matrix for your Torch/CUDA version; do not blindly upgrade torch.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Phase18-Setup\Install-ModelZoo.ps1 -Root D:\Distributed-GPU
```

3. Restart Laptop A controller **with the existing cluster token**. The `.dgpu/worker_config.json` saved DPAPI token is not modified. If Phase 17 AutoHeal owns the controller, let it restart it once; don't start a second controller at the same port.
4. Open `http://192.168.50.1:45400/model-zoo`. Enter existing cluster token into the page to **authorize training** (the field is not stored in browser localStorage), pick a model and correct dataset path in project `data/`.
5. Browser jobs create `.dgpu/model_zoo_runs/<id>/train.log`, `status.json`, and `model.pt` after training. YOLO `model.pt` is Ultralytics model; DistilGPT2 `model.pt` is metadata and the actual adapter is `lora_adapter/`.

**If `studio/server.py` uses an unusual app factory**, `patch_server` intentionally refuses changes. Add the router manually after `app = FastAPI(...)` has been created, or request integration using the current server source. A backup `studio/server.py.before_model_zoo.bak` is written before modifications. Repeating the patch will not duplicate it.

**Existing web login security may differ:** Model Zoo **training endpoints require a bearer cluster token**, separate from any existing session cookie. Keep cluster on trusted private Ethernet; HTTP on port 45400 does not encrypt the token.

## Data formats and actual example paths

| Model family | Dataset path example | Format |
|---|---|---|
| Tabular regression | `data/zoo_demo/tabular_regression.csv` | Header; numeric features; numeric target final column |
| Tabular classification | `data/zoo_demo/tabular_classification.csv` | Header; numeric features; integer class final column |
| Images / CV | `data/zoo_demo/images` | `images/<class>/*.jpg` or PNG; >=2 classes |
| U-Net | `data/zoo_demo/segmentation` | `images/name.png` and `masks/name.png`, binary masks |
| Audio spectrogram | `data/zoo_demo/spectrograms` | Class-labeled spectrogram images (not WAV) |
| NLP | `data/zoo_demo/text.csv` | `text,label` header, strings allowed |
| Tiny LM | `data/zoo_demo/lm.txt` | UTF-8 plain-text corpus (>=66 chars) |
| Time series | `data/zoo_demo/timeseries.csv` | One numeric value per row; optional header |
| DistilGPT2 LoRA | `data/zoo_demo/lm.txt` | UTF-8 corpus, >128 tokenizer tokens; weights download |
| YOLO object detection | `data/my_yolo/dataset.yaml` | Standard Ultralytics train/val/image/label YOLO dataset |

The generated demo data is synthetic; its losses do not demonstrate generalization to real datasets.

## CLI smoke tests before browser integration

```powershell
python -m unittest discover -s Phase18-Setup/tests -v
python -m studio_zoo.demo_data
python -m studio_zoo.train --model tabular_mlp_regression --dataset data/zoo_demo/tabular_regression.csv --output .dgpu/zoo_smoke_tabular --epochs 1 --batch-size 8
python -m studio_zoo.train --model image_cnn --dataset data/zoo_demo/images --output .dgpu/zoo_smoke_images --epochs 1 --batch-size 4
python -m studio_zoo.train --model text_gru --dataset data/zoo_demo/text.csv --output .dgpu/zoo_smoke_text --epochs 1 --batch-size 4
python -m studio_zoo.train --model tiny_transformer_lm --dataset data/zoo_demo/lm.txt --output .dgpu/zoo_smoke_lm --epochs 1 --batch-size 4
```

These CLI commands are for testing only; browser UI launches the same local Python module from the controller.

### Optional experimental dependencies

```powershell
python -m pip install transformers peft accelerate
python -m pip install ultralytics
```

Use a separate virtual environment for optional packages when possible to protect existing CUDA PyTorch. Model downloads require Internet and trust in the model publisher. Only load saved checkpoints from a trusted source.

## Troubleshooting

- HTTP 401 in Model Zoo: its browser token is missing/mismatched; verify controller `DGPU_TOKEN`. Do not paste token into chat.
- 409: Another Model Zoo job is in progress.
- Training starts then fails: inspect `.dgpu/model_zoo_runs/<id>/train.log` for the exception.
- `CUDA out of memory`: reduce batch and/or use a smaller preset; this extension does not pool the two GPUs.
- `torchvision::nms` import errors: installed torchvision/PyTorch builds don't match; install compatible versions.
- `/model-zoo` 404: controller was not restarted or router integration was not applied.
- Regression in legacy training: remove the 4-line `DGPU_MODEL_ZOO` block from `studio/server.py`, using backup if necessary.

## Testing status

The included non-optional PyTorch architecture shape tests and selected synthetic-data CPU training smoke tests are runnable locally. **Actual user Laptop A/B deployment, browser API authorization, GPU training, optional YOLO/LoRA model downloads, memory sizing and any 2-GPU distributed training of new architectures must be tested on the user's machines.**
