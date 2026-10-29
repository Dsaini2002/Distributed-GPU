# Phase 20 beta — Create Cluster / Join Cluster

This upgrade works on top of your already-working **Phase 19 local-SGD trainer**, without changing the training algorithm. It is a Windows **LAN-only prototype**, not a production-ready installer or a fully secure distributed system. Keep the two machines on a dedicated, trusted Ethernet network. Do **not** expose ports 45400 or 45420 to the internet.

## What works
- Host `http://192.168.50.1:45400/cluster` creates an 8-character, 5-minute join code.
- Laptop B agent provides a local `http://127.0.0.1:45430/join` page, offers UDP host discovery plus manual IP fallback.
- Worker requests approval; owner must compare **12-digit hex fingerprint** displayed on both screens.
- After approval, the existing Phase 19 cluster token is encrypted for the device's temporary RSA public key; it is never returned as plaintext over HTTP. Worker saves it using Windows user-bound DPAPI and launches its GPU worker on port 45420.
- Worker agent launches through `pythonw.exe` without a terminal window, and it restarts its own Phase 19 worker if that process exits.
- Existing Phase 17 controller and legacy worker (ports 45400/45410) are unchanged.

## Prerequisites
- Laptop A existing Controller already operates at `192.168.50.1:45400`, Python 3.10 CUDA enabled, Phase 19 installed.
- Laptop B RTX 2050 with working PyTorch CUDA, FastAPI, Uvicorn, and `studio_multi.worker`, checked in the **same Python** selected for installer.
- Python package `cryptography` installed on A and B: `python -m pip install cryptography` with each selected Python, if missing. Use the working CUDA Python; don't overwrite PyTorch.
- Host and worker share a *trusted* LAN. Password-protected GUI pages are only a basic beta; the channel itself is HTTP (not TLS).

## Laptop A (Host) — one-time
1. Extract zip inside `D:\Distributed-GPU\Phase20-Setup` (ZIP contains `studio_pairing` module and scripts).
2. In PowerShell (not the torch-less `.venv`):

```powershell
cd D:\Distributed-GPU
$py = 'C:\Users\Dinesh saini\AppData\Local\Programs\Python\Python310\python.exe'  # verify on A; or use the actual CUDA interpreter
& $py -m pip install cryptography
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Phase20-Setup\Install-Host.ps1 -Root D:\Distributed-GPU -Python $py
```
3. Restart the existing AutoHeal-owned Controller **once only**, with all training idle. Do not launch a second Uvicorn instance.
4. Open `http://192.168.50.1:45400/cluster`, enter host's private cluster token and click **Create temporary join code**.

## Laptop B (Worker) — after A pushed code and B pulled
After `git pull`, the repo should contain `studio_pairing/` and `Phase20-Setup/`.

```powershell
cd D:\Distributed-GPU
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path  # or another CUDA-enabled Python
& $py -m pip install cryptography
# This switches worker port 45420 to the Phase20 supervisor. Only do while no training runs.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Phase20-Setup\Install-Worker.ps1 -Root D:\Distributed-GPU -Python $py -TakeOverPhase19
```
Open local `http://127.0.0.1:45430/join`. Choose **Find Host Automatically**, or enter `192.168.50.1`. Enter the code from Laptop A. Compare fingerprints on both screens; then approve on A. A newly paired worker should listen on `192.168.50.2:45420`. Firewall may require adding a narrowly scoped rule via elevated PowerShell for TCP 45420, and optionally UDP 45431 for discovery.

## Verify
On A:
```powershell
Test-NetConnection 192.168.50.2 -Port 45420
```
Open `http://192.168.50.1:45400/distributed-zoo` and do the same `tabular_mlp_regression`, 3 epochs, 8 batch, 0.001 LR, dataset `data/zoo_demo/tabular_regression.csv` test. On the Host `/cluster` page, select **Train with approved GPUs** to prefill the Phase19 workers field. If the existing Phase19 UI differed from the reference version, this optional UI patch is safely skipped and users can enter the IP manually. Node scheduling is still manual (approved nodes are not auto-benchmarked).

## Tests
On A, in project root with working FastAPI/cryptography:
```
python -m unittest discover -s Phase20-Setup/tests -p 'test_phase20.py' -v
```

## Push A then pull B
Never commit `.dgpu/`, `.venv/`, `.env`, checkpoints, or real credentials. Inspect `git status --short` and `git diff --cached --name-only` before push.
```
git add studio_pairing/ studio/server.py Phase20-Setup/ .gitignore
git diff --cached --check
git diff --cached --name-only
git commit -m "Phase 20 beta: approved LAN join and worker agent"
git push origin main
```
On B:
```
git status --short
git pull --ff-only origin main
```
If Git reports local changes, preserve them; **do not** use `reset --hard` or `clean -fd`. No local token file is committed.

## Limits, safety and rollback
- Pairing approvals and displayed device list are held in the controller's memory. Restarting the controller clears these lists, but the worker's DPAPI pairing survives Windows login. The worker remains authorized until the shared token is rotated.
- **Production security not solved:** HTTP metadata and join code are visible on trusted LAN; active on-path attackers are not fully defended against. The device fingerprint comparison is required; never approve unknown devices. Future: TLS, per-device revocation/token scopes, bounded payloads & formal security review.
- **Token is still shared** across your Phase 19 workers. Host cannot revoke a single worker without changing the shared secret, yet.
- No app executable/MSI, no universal model training, no verified 3+ GPUs, no GPU pooling. This beta covers pairing + background worker control.
- To roll back B, `Stop-ScheduledTask DGPU-Phase20-Agent; Disable-ScheduledTask DGPU-Phase20-Agent`; restore the old `DGPU-Phase19-Worker` scheduled task and restart it, then test port 45420.
- To roll back Host router, restore the `studio/server.py.before_phase20_*.bak` file created by the host installer (with controller offline), or remove the marked four-line Phase20 router block and restart controller.
