param(
  [ValidateSet('Controller','Worker')][string]$Role,
  [string]$Root = 'D:\Distributed-GPU',
  [string]$Python = ''
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path $Root).Path
Set-Location $Root
if (-not $Python) { $Python = (Get-Command python -ErrorAction Stop).Source }
if (-not (Test-Path $Python)) { throw "Python missing: $Python" }
Write-Host "Role=$Role Python=$Python"
& $Python -c 'import torch,fastapi,uvicorn;print(torch.__version__,torch.cuda.is_available())'
if ($LASTEXITCODE -ne 0) { throw 'Selected Python lacks torch/fastapi/uvicorn' }
& $Python -c 'import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)'
if ($LASTEXITCODE -ne 0) { throw 'CUDA required; select the correct Python interpreter' }
if (-not (Test-Path '.dgpu\worker_config.json')) { throw 'Saved worker token config missing; finish Phase16 first' }
if ($Role -eq 'Controller' -and -not (Test-Path 'studio\server.py')) { throw 'studio/server.py missing' }
$origin = Join-Path $PSScriptRoot 'studio_zoo'
$target = Join-Path $Root 'studio_zoo'
if (Test-Path $target) {
    foreach ($name in @('models.py','train.py')) {
        $expected = (Get-FileHash (Join-Path $origin $name) -Algorithm SHA256).Hash
        $existing = (Get-FileHash (Join-Path $target $name) -Algorithm SHA256).Hash
        if ($expected -ne $existing) { throw "Model Zoo mismatch: $name. Keep the same version on all machines. No changes made." }
    }
} else {
    Copy-Item $origin -Destination $target -Recurse
    Write-Host 'Installed matching Model Zoo training/model files'
}
$moduleDest = Join-Path $Root 'studio_multi'
if (-not (Test-Path $moduleDest)) { New-Item -ItemType Directory -Path $moduleDest | Out-Null }
Copy-Item (Join-Path $PSScriptRoot 'studio_multi\*') -Destination $moduleDest -Recurse -Force
& $Python -c "import studio_multi.core; import studio_multi.worker; import studio_multi.coordinator"
if ($LASTEXITCODE -ne 0) { throw 'Phase19 module imports failed' }
if ($Role -eq 'Controller') {
    & $Python -m studio_multi.patch_server
    if ($LASTEXITCODE -ne 0) { throw 'Controller server patch failed' }
    Write-Host 'CONTROLLER READY: restart existing AutoHeal-managed controller ONCE; use /distributed-zoo.'
} else {
    # Save interpreter path only; the DPAPI-protected token remains in the existing worker config.
    $Python | Set-Content -Path '.dgpu\phase19_python.txt' -Encoding UTF8
    Copy-Item (Join-Path $PSScriptRoot 'Start-Phase19-Worker.ps1') (Join-Path $Root 'Start-Phase19-Worker.ps1') -Force
    $taskName = 'DGPU-Phase19-Worker'
    $script = Join-Path $Root 'Start-Phase19-Worker.ps1'
    $arguments = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + $script + '" -Root "' + $Root + '"'
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arguments -WorkingDirectory $Root
    $id = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $id
    $principal = New-ScheduledTaskPrincipal -UserId $id -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
    try {
        Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description 'DGPU Phase19 worker (private Ethernet only)' -Force | Out-Null
        $existingListener = Get-NetTCPConnection -LocalPort 45420 -State Listen -ErrorAction SilentlyContinue
        if (-not $existingListener) { Start-ScheduledTask -TaskName $taskName }
        Write-Host "WORKER SCHEDULED: $taskName; private LAN port 45420"
    } catch {
        Write-Warning "Task registration failed: $($_.Exception.Message)"
        Write-Warning 'Launch manually from PowerShell: .\Start-Phase19-Worker.ps1'
    }
}
Write-Host 'Training files and cluster tokens were not deleted or replaced.'
