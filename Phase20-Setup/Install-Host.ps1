# Phase20: run once on Laptop A from the extracted Phase20-Setup folder.
param([string]$Root='D:\Distributed-GPU', [string]$Python='')
$ErrorActionPreference='Stop'
$rootPath=(Resolve-Path -LiteralPath $Root).Path
if (-not $Python) { $Python=(Get-Command python -ErrorAction Stop).Source }
if (-not (Test-Path -LiteralPath $Python)) { throw 'Python executable missing' }
& $Python -c 'import fastapi, cryptography'
if ($LASTEXITCODE -eq 0) { Write-Host 'Phase 20 dependencies OK' }
if ($LASTEXITCODE -ne 0) { throw 'Install cryptography into selected CUDA Python first: PYTHON -m pip install cryptography' }
if (-not (Test-Path (Join-Path $rootPath 'studio\server.py'))) { throw 'Cannot find studio\server.py' }
if (-not (Test-Path (Join-Path $PSScriptRoot 'studio_pairing\host.py'))) { throw 'Pairing package incomplete' }
$module=Join-Path $rootPath 'studio_pairing'
New-Item -ItemType Directory -Path $module -Force | Out-Null
Copy-Item -Path (Join-Path $PSScriptRoot 'studio_pairing\*.py') -Destination $module -Force
$server=Join-Path $rootPath 'studio\server.py'
$content=[IO.File]::ReadAllText($server)
if (-not $content.Contains('# DGPU_PHASE20_BEGIN')) {
    $stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
    Copy-Item -LiteralPath $server -Destination ($server+'.before_phase20_'+$stamp+'.bak') -Force
    $content += "`r`n# DGPU_PHASE20_BEGIN`r`nfrom studio_pairing.host import router as _dgpu_phase20_router`r`napp.include_router(_dgpu_phase20_router)`r`n# DGPU_PHASE20_END`r`n"
    [IO.File]::WriteAllText($server,$content,[Text.UTF8Encoding]::new($false))
    Write-Host 'Registered /cluster pairing route.'
} else { Write-Host 'Route already registered; no duplicate patch.' }
$cfg=Join-Path $rootPath '.dgpu\worker_config.json'
if (-not (Test-Path $cfg)) { Write-Warning 'Host worker_config.json missing; existing Controller must have DGPU_TOKEN set.' }
Push-Location $rootPath
try {
    & $Python -c 'from studio_pairing.host import router; assert len(router.routes) >= 7'
    if ($LASTEXITCODE -eq 0) { Write-Host 'Host route import OK' }
    if ($LASTEXITCODE -ne 0) { throw 'Host route import test failed' }
    & $Python -m studio_pairing.patch_phase19
    if ($LASTEXITCODE -ne 0) { throw 'Phase19 prefill patch execution failed' }
} finally { Pop-Location }
Write-Host 'HOST INSTALLED. Restart the ONE existing Controller on Laptop A (do not start a duplicate).'
Write-Host 'Then open http://192.168.50.1:45400/cluster and create a short-lived code.'
