# Distributed-GPU Studio — Desktop Client (Developer Notes)

This folder contains the Windows desktop shell used by Distributed-GPU Studio. The user-facing overview, verified features, safety notes and installation status are described in the [repository README](../README.md).

## Important scope

- Native desktop window via pywebview/WebView2 with local HTML, CSS and JavaScript.
- GUI calls Python bridge methods, which connect to the existing FastAPI controller, model catalog, training APIs and worker pairing agent.
- The **GPU backend must already be installed** on each machine. This is **not** yet a fresh-PC CUDA/PyTorch installer.
- Only supported models reported by the installed backend should be offered for training.
- Existing encrypted cluster credentials and checkpoints must not be changed automatically during a desktop update.
- The desktop UI should not display internal development phase numbers.

## Source layout

```text
Desktop-App/
├── desktop_app/
│   ├── assets/         # Frontend assets
│   ├── core.py         # Authenticated backend bridge and desktop state
│   └── main.py         # Desktop window entry point
├── tests/              # Python unit tests
├── Build-Windows.ps1   # Portable EXE build script
├── Install-Desktop.ps1 # Existing-machine beta installer UI
├── Run-Desktop-Dev.ps1 # Maintainer development launch script
└── DistributedGPUStudio.iss # Optional Inno Setup script
```

## Developer test commands

Run in a **Windows PowerShell** session from this folder with a supported Python installation:

```powershell
python -m pip install -r requirements-desktop.txt
python -m unittest discover -s tests -v
.\Build-Windows.ps1 -Python (Get-Command python).Source
```

Expected portable output after successful Windows build:

```
dist/DistributedGPUStudio/DistributedGPUStudio.exe
```

Optional Inno Setup wizard build (Inno Setup 6 compiler must be installed and on PATH):

```powershell
.\Build-Windows.ps1 -Python (Get-Command python).Source -Installer
```

Build scripts and generated Windows installer have not been tested on a clean public user's PC. The installer here currently packages the desktop UI shell, not the complete training services.

For safe Git staging and publication steps, see [`docs/PUBLISH_GUIDE.md`](../docs/PUBLISH_GUIDE.md).

## Project terminal and GPU details

The Terminal screen replaces the Train Model form. Open full terminal launches a
separate interactive Windows PowerShell in the selected project directory. The
project .venv is used if available. Run `gpu-status` to display all local NVIDIA
GPUs and approved remote workers: GPU name, utilization, used/free/total VRAM,
and temperature. Unsupported measurements are shown as unavailable.

Commands run locally. Using remote GPUs still requires distributed training code;
VRAM is not pooled. Terminal jobs remain independent of the Desktop App. This is
not an embedded terminal emulator or a cluster shutdown feature.

### Update an existing Laptop B

Keep the existing repository, Python environment, pairing credentials and data.
When training is idle, update the existing checkout:

```powershell
cd D:\Distributed-GPU
git pull --ff-only origin main
```

The worker must subsequently be restarted while idle to load the new
`/api/p19/gpus` endpoint. Updating source does not restart a running Python worker.
Use the existing worker launch method; do not terminate all Python processes.
If Git reports local changes, preserve them and resolve the diff before updating.
If the project folder or environment was deleted, pulling alone is insufficient:
restore the existing setup or follow the repository worker installation guide.

The endpoint uses the existing bearer authentication on port 45420. The pairing
agent remains loopback-only. Only workers approved by the current controller
session are queried. Existing trusted-LAN HTTP transport is unchanged.
A worker returning 404 needs its source updated and an idle restart.

### Open the updated Host app

```powershell
cd D:\Distributed-GPU\Desktop-App
& D:\Distributed-GPU\.venv\Scripts\python.exe -m desktop_app.main
```

An installed desktop shortcut may use a separate, older app copy. Update it with
Install-Desktop.ps1, or use the source launch above. Previously built executables
must be rebuilt to include the new interface.

Validation: 15 existing Desktop tests and 7 terminal/GPU tests passed on Windows;
JavaScript syntax, UI wiring and PowerShell startup fixture checks passed.
Two-laptop live telemetry after updating Laptop B remains to be verified.
