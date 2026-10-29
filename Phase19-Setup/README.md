# Distributed-GPU Studio Phase 19 — Multi-Laptop Model Zoo (local SGD)

**What is implemented:** In each epoch, Laptop A trains one dataset shard on its CUDA GPU while Laptop B (and optionally more separately configured worker laptops) trains other shards. Each participant starts from the same global model weights. The controller performs **sample-weighted model averaging after each epoch** and saves a merged checkpoint. This is *local-SGD / FedAvg-style training*. It is **not** torch.distributed all-reduce DDP, not synchronized mini-batch gradients, not layer-by-layer model partition, not FSDP, and **does not combine VRAM into a single 8GB device**. This algorithm may converge differently from ordinary single-GPU training.

**Supported using the same Phase18 model and dataset loaders:** Tabular regression and classification, image CNN, ResNet18, MobileNetV3 Small, EfficientNet-B0, Tiny ViT, binary U-Net, audio spectrogram-image CNN, text GRU, text Transformer, character-level tiny Transformer LM, time-series LSTM. **Not supported:** YOLO11n and pretrained DistilGPT2-LoRA; both require architecture-specific distributed adapters. Models with BatchNorm (e.g. ResNet) have approximate buffer averaging; first test with small CNN/tabular.

## Before installing

1. Commit/backup project source. Stop all training jobs (including the old /model-zoo local trainer).
2. Controller and both existing GPU workers must be healthy. Do **not** uninstall Phase17 AutoHeal or delete old tokens. Existing Phase18 `studio_zoo/models.py` and `train.py` must match the bundled version; mismatches cause a safe install refusal.
3. Same private cluster token must be present in both `.dgpu/worker_config.json` files under the matching Windows user (DPAPI encrypted). Never paste token into chat/GitHub.
4. Both computers need a CUDA-capable `torch`, FastAPI and Uvicorn; image models also require compatible `torchvision` and Pillow. Python versions/Model Zoo source must agree. Use the Python interpreter that has CUDA; Laptop A previously used system Python 3.10, Laptop B used `.venv`.
5. Phase19 LAN traffic is HTTP plaintext, including bearer token and dataset; use dedicated, private Ethernet only. **Do not open port 45420 to Wi-Fi/public Internet**. This is experimental, not a hardened public service. Dataset upload cap: 48MiB compressed / 90MiB uncompressed; model transfer cap: 90MiB state. One training job at a time.

## Install on Laptop B (RTX 2050, private Ethernet 192.168.50.2)

Copy ZIP to `D:\Distributed-GPU`, extract, open PowerShell **with the same user account used for worker DPAPI setup**:

```powershell
cd D:\Distributed-GPU
Expand-Archive .\DGPU-Phase19-MultiGPU.zip -DestinationPath .\Phase19-Setup -Force
.\Phase19-Setup\Install-Phase19.ps1 -Role Worker -Root D:\Distributed-GPU -Python 'D:\Distributed-GPU\.venv\Scripts\python.exe'
Test-NetConnection 192.168.50.2 -Port 45420
Get-Content .\.dgpu\phase19_worker.log -Tail 20
```

If script execution is restricted, use:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Phase19-Setup\Install-Phase19.ps1 -Role Worker -Root D:\Distributed-GPU -Python 'D:\Distributed-GPU\.venv\Scripts\python.exe'
```

This creates the **separate** `DGPU-Phase19-Worker` scheduled task for port 45420. It does **not** modify existing Phase17 tasks or training on port 45410. If it cannot create task, manually run `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Start-Phase19-Worker.ps1` in a dedicated terminal (leave it open). On a future reboot, login to the same Windows account for scheduled task startup.

## Install on Laptop A (GTX 1650, private Ethernet 192.168.50.1)

Use the **CUDA working system Python** (the earlier `.venv` lacked `torch`):

```powershell
cd D:\Distributed-GPU
Expand-Archive .\DGPU-Phase19-MultiGPU.zip -DestinationPath .\Phase19-Setup -Force
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Phase19-Setup\Install-Phase19.ps1 -Role Controller -Root D:\Distributed-GPU -Python 'C:\Users\Dinesh saini\AppData\Local\Programs\Python\Python310\python.exe'
```

Controller source is safely patched with `# DGPU_PHASE19_BEGIN ... END`; backup: `studio/server.py.before_phase19.bak`. Restart **the existing AutoHeal-managed controller once**, without opening another Uvicorn instance (45400 is already owned). Check `http://192.168.50.1:45400/distributed-zoo`. The page still requires the private cluster token for training API calls.

Check remote worker from Laptop A:

```powershell
Test-NetConnection 192.168.50.2 -Port 45420
```

## First test — 2 actual GPUs

Browser: `http://192.168.50.1:45400/distributed-zoo`

- Model: `tabular_mlp_regression` (first smoke test) or `image_cnn` (after tabular works)
- Dataset: `data/zoo_demo/tabular_regression.csv` or `data/zoo_demo/images`
- Remote workers: `192.168.50.2:45420`
- Epochs 3, batch size 4 (CNN) / 8 (tabular), LR 0.001

The controller uploads a bounded dataset snapshot to Laptop B automatically; no manual dataset copy. The response prints `nodes=2`, `devices=['cuda','cuda']`, per-epoch loss, and checkpoint location under `.dgpu/phase19_runs/<job-id>/model.pt`. Synthetic losses are not a validation benchmark.

CLI from Laptop A (requires `$env:DGPU_TOKEN`, loaded privately or already present in same PowerShell environment):

```powershell
python -m studio_multi.coordinator --model image_cnn --dataset data/zoo_demo/images --workers 192.168.50.2:45420 --epochs 3 --batch-size 4 --lr 0.001 --output .dgpu/phase19_runs/cli_cnn
```

Resume interrupted runs (last successfully averaged epoch only):

```powershell
python -m studio_multi.coordinator --model image_cnn --dataset data/zoo_demo/images --workers 192.168.50.2:45420 --epochs 5 --batch-size 4 --lr 0.001 --output .dgpu/phase19_runs/cli_cnn --resume
```

The algorithm reinitializes each participant's AdamW optimizer every epoch. Checkpoints capture model state, epoch and history, not optimizer momentum. If remote GPU disappears mid-epoch, the run **fails safely** rather than claiming success or switching to single-GPU. Rerun with `--resume` after recovery; browser job resume is not yet provided.

## How many laptops?

The controller accepts a semicolon-separated list of remote **private IPv4:port** workers (`192.168.50.2:45420;192.168.50.3:45420`). Each worker needs Phase19 service, its own CUDA GPU, the same secret, and the same Phase18 architecture source. The code permits up to 24 remote workers **but only 2 physical laptops are available for verification**. Larger N is unverified and likely bandwidth-limited by HTTP and full model-state transfers.

## Source ownership and rollback

- New files live in `studio_multi/` and (on Laptop B) `Start-Phase19-Worker.ps1`.
- Existing Phase18 trainers and old Phase17 AutoHeal not replaced.
- To temporarily remove UI, delete only `# DGPU_PHASE19_BEGIN` through `# DGPU_PHASE19_END` from `studio/server.py` after backup. Stop scheduled task on B with `Stop-ScheduledTask -TaskName DGPU-Phase19-Worker`; avoid killing other workers.
- `.dgpu/` contains private config and checkpoints and MUST remain ignored by Git. Never commit tokens.
- For model-zoo package upgrades or custom models, sync architecture source exactly on all laptops and rerun model tests.

## Tests

```powershell
python -m unittest discover -s .\Phase19-Setup\tests -v
```

CLI and HTTP integration tests are CPU-only when explicitly using `--device cpu` and `DGPU_P19_ALLOW_CPU=1`; this is not proof of multi-GPU CUDA success on the two real laptops. Native Windows PyTorch `torch.distributed` still has prototype platform support and NCCL is unavailable there; this package deliberately uses a simpler HTTP local-SGD protocol instead of claiming NCCL DDP support.
