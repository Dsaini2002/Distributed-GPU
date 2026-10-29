# Build a true Windows EXE using the Python installation on a Windows PC.
# Run once on Windows (not in Linux). Does not edit D:\Distributed-GPU.
param([string]$Python='', [switch]$Installer)
$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
if (-not $IsWindows -and $PSVersionTable.PSEdition -eq 'Core') { throw 'Run this build on Windows.' }
if (-not $Python) { $Python=(Get-Command python -ErrorAction Stop).Source }
Write-Host 'Checking desktop build dependencies...' -ForegroundColor Cyan
& $Python -m pip install --default-timeout 90 --retries 5 -r .\requirements-desktop.txt
if ($LASTEXITCODE -ne 0) { throw 'Could not install desktop build dependencies.' }

Write-Host 'Building DistributedGPUStudio.exe...' -ForegroundColor Cyan
$arguments=@(
    '-m','PyInstaller', '--noconfirm','--clean','--onedir','--windowed',
    '--name','DistributedGPUStudio','--paths','.',
    '--add-data','desktop_app\assets;desktop_app\assets',
    '--collect-all','webview',
    '--icon','desktop_app\assets\app.ico',
    'desktop_app\main.py'
)
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw 'Windows EXE build failed. See PyInstaller output.' }
$exe=Join-Path $PSScriptRoot 'dist\DistributedGPUStudio\DistributedGPUStudio.exe'
if (-not (Test-Path $exe)) { throw 'Build finished but EXE was not found.' }
Write-Host "Portable app ready: $exe" -ForegroundColor Green
Write-Host 'Copy the WHOLE dist\DistributedGPUStudio folder, not only the EXE.'
if ($Installer) {
    $iscc=Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if (-not $iscc) { throw 'Inno Setup compiler ISCC.exe not found. Install Inno Setup, or use portable app folder.' }
    & $iscc.Source '.\DistributedGPUStudio.iss'
    if ($LASTEXITCODE -ne 0) { throw 'Inno Setup build failed.' }
    Write-Host 'Windows installer output in installer-output folder.' -ForegroundColor Green
}
