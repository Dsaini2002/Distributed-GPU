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
