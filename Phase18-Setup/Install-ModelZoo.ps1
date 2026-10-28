param([string]$Root = 'D:\Distributed-GPU')
$ErrorActionPreference = 'Stop'
if (-not (Test-Path (Join-Path $Root 'studio\server.py'))) {
    throw "Not a Distributed-GPU Studio root: $Root"
}
Set-Location $Root
$python = (Get-Command python -ErrorAction Stop).Source
Write-Host "Python: $python"
& $python -c 'import torch, fastapi, pydantic; print(torch.__version__, torch.cuda.is_available())'
if ($LASTEXITCODE -ne 0) { throw 'Selected Python lacks torch/fastapi. Activate working environment first.' }
$source = Join-Path $PSScriptRoot 'studio_zoo'
$dest = Join-Path $Root 'studio_zoo'
if (Test-Path $dest) {
    Write-Host 'Model Zoo code directory already exists; leaving its files unchanged.'
} else {
    Copy-Item -Path $source -Destination $dest -Recurse
    Write-Host "Copied Model Zoo source to $dest"
}
& $python -m studio_zoo.patch_server
if ($LASTEXITCODE -ne 0) { throw 'Server patch failed. See output; existing server may remain unchanged.' }
if (-not (Test-Path (Join-Path $Root 'data\zoo_demo'))) {
    & $python -m studio_zoo.demo_data
    if ($LASTEXITCODE -ne 0) { Write-Warning 'Demo generation failed; you can try again after installing Pillow.' }
} else {
    Write-Host 'Demo data already exists; preserving it.'
}
Write-Host 'Model Zoo installed on controller. Restart controller and open /model-zoo.'
Write-Host 'Do not run legacy 2-GPU training and Model Zoo training simultaneously.'

