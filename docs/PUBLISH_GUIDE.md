# Publish Distributed-GPU Studio to GitHub — maintainer guide

**Repository:** https://github.com/Dsaini2002/Distributed-GPU

This guide is for the **project maintainer**. Regular users should not have to use Git, Python or PowerShell. The current beta is not a universal fresh-PC installer.

## A. Before you push (on Laptop A)

1. Confirm training is idle and the current desktop UI is closed if replacing its source.
2. Backup any existing repository-root `README.md` and `.gitignore` if you want to preserve their contents.
3. Download/extract the GitHub-ready package. Copy these items to the **root** of `D:\Distributed-GPU` (not into a nested `DistributedGPU-GitHub-Ready` folder):
   - `README.md`
   - `SECURITY.md`
   - `docs/` (including `docs/images/desktop-distributed-training.png`)
   - `Desktop-App/` (latest desktop source)
   - `.github/workflows/windows-desktop-beta.yml` (optional manual Windows build workflow; not a verified public release)
   - `Install Distributed-GPU Studio.vbs`
   - `READ-ME-FIRST.txt`
4. If the root already has a `.gitignore`, **merge** the lines from `gitignore-RECOMMENDED.txt` into it. Otherwise, rename `gitignore-RECOMMENDED.txt` to `.gitignore` and copy it to the project root.
5. Do not copy protected `.dgpu`, virtual environments or data into the bundle. Those should remain **local** on the PCs.

If you prefer, extract the ready bundle directly in the repository folder, **after backing up existing files** and reviewing the contents to avoid overwriting custom source changes. This package does not contain a complete clone of the backend: keep your existing `studio/`, `studio_multi/`, `studio_pairing/` and `studio_zoo/` folders.

## B. Check the repository and branch

In PowerShell on **Laptop A**:

```powershell
cd D:\Distributed-GPU

git status --short
git branch --show-current
git remote -v
```

Confirm `main` is the branch you intend to publish, and `origin` points to your own repository. If there are unrelated changes, do not stage or discard them blindly.

## C. Verify private files are not tracked

```powershell
git ls-files -- .venv .dgpu data checkpoints
```

If the output is **empty**, good. If anything is listed, it is already tracked. The ignore file does **not** untrack it. Back up the affected folders and carefully de-index the tracked local directories **without deleting local content**:

```powershell
# Only use this if the preceding command listed sensitive tracked files.
git rm -r --cached --ignore-unmatch -- .venv .dgpu data checkpoints
```

Review the staged removals; these are necessary only if those files were tracked previously. If any real credentials or private datasets were pushed in **older commits**, also rotate those credentials and separately remove them from Git history.

## D. Stage only reviewed project files

```powershell
cd D:\Distributed-GPU

git add -- README.md SECURITY.md .gitignore docs/ Desktop-App/ .github/workflows/windows-desktop-beta.yml "Install Distributed-GPU Studio.vbs" READ-ME-FIRST.txt

# Stage any backend source changes you've actually made and reviewed:
foreach ($folder in @('studio','studio_zoo','studio_multi','studio_pairing')) {
    if (Test-Path $folder) { git add -A -- $folder }
}

git diff --cached --check
git diff --cached --stat
git diff --cached --name-only
```

**STOP** if the staged list contains `.dgpu`, `.venv`, personal CSV/image datasets, secrets, `*.pt` checkpoints or unwanted ZIPs. View the content before committing:

```powershell
git diff --cached
```

If you accidentally staged a private file:

```powershell
git restore --staged -- "PATH/TO/PRIVATE/FILE"
```

Replace the example path with the actual staged file path. Do not use `git add .` until your repository ignore rules and tracked-file state are audited.

## E. Commit and push

After reviewing staged changes and confirming `main` is the intended destination:

```powershell
git commit -m "docs: publish desktop studio README and beta source"
git push origin main
```

If Git says "nothing to commit", inspect the status and whether you copied files to the right directory. If a push is rejected because remote changed, do not force-push; fetch and reconcile first:

```powershell
git fetch origin
git status --short
git log --oneline --left-right --graph HEAD...origin/main
```

**Do not use `git push --force` or `git reset --hard` to fix a conflict.**

## F. Verify GitHub

Open https://github.com/Dsaini2002/Distributed-GPU and refresh. Confirm:

- Root README automatically renders on the repository home page.
- `Desktop-App/` and `docs/images/desktop-distributed-training.png` exist.
- Screenshot renders in README.
- `.dgpu`, `.venv`, checkpoints and real datasets are not present.
- New commit message and date appear on the default branch.

## G. Sync Laptop B (optional for shared source)

On Laptop B:

```powershell
cd D:\Distributed-GPU

git status --short
git fetch origin main
git pull --ff-only origin main
```

If Git complains about **dubious ownership** and this is your trusted repository at this exact path:

```powershell
git config --global --add safe.directory D:/Distributed-GPU
```

Then retry `git status` and `git pull --ff-only`. If local changes block the pull, preserve them first instead of doing a destructive reset.

## H. Releasing a Windows installer

**Source push ≠ release.** The GitHub **Code → Download ZIP** action only downloads source; it does not install PyTorch, CUDA or background GPU services.

A future public user experience should be:

1. GitHub **Releases** → download a validated `DistributedGPU-Setup.exe`.
2. Double-click Setup → Install → Open.
3. Create or Join Cluster → approve once → Train.

Before publishing any executable as a release: build on Windows, test on a **fresh PC**, virus-scan/sign if possible, verify installation/uninstall, confirm GPU/CUDA setup and auto-reconnect, and ensure no local tokens are packaged. The current desktop shell setup is **beta for already-configured PCs**, not a complete general-public installer.
