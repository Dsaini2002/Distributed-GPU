# Phase20: run once on Laptop B; same user account as existing Phase19 setup.
param([string]$Root='D:\Distributed-GPU', [string]$Python='', [switch]$TakeOverPhase19)
$ErrorActionPreference='Stop'
$rootPath=(Resolve-Path -LiteralPath $Root).Path
if (-not $Python) { $Python=(Get-Command python -ErrorAction Stop).Source }
if (-not (Test-Path -LiteralPath $Python)) { throw 'Python executable missing' }
Push-Location $rootPath
try {
    & $Python -c 'import torch, fastapi, uvicorn, cryptography, studio_multi.worker; assert torch.cuda.is_available()'
    if ($LASTEXITCODE -eq 0) { Write-Host 'Worker CUDA and pairing dependencies OK' }
    if ($LASTEXITCODE -ne 0) { throw 'Worker needs CUDA PyTorch + fastapi + uvicorn + cryptography in selected Python.' }
} finally { Pop-Location }
$source=Join-Path $PSScriptRoot 'studio_pairing'
if (-not (Test-Path (Join-Path $source 'agent.py'))) { throw 'Pairing package incomplete' }
$target=Join-Path $rootPath 'studio_pairing'
New-Item -ItemType Directory -Force -Path $target | Out-Null
Copy-Item (Join-Path $source '*.py') -Destination $target -Force
if ($TakeOverPhase19) {
    $old=Get-ScheduledTask -TaskName 'DGPU-Phase19-Worker' -ErrorAction SilentlyContinue
    if ($old) {
        $backup=Join-Path $rootPath '.dgpu\phase19_task_before_phase20.xml'
        Export-ScheduledTask -TaskName 'DGPU-Phase19-Worker' | Set-Content -Path $backup -Encoding UTF8
        Stop-ScheduledTask -TaskName 'DGPU-Phase19-Worker' -ErrorAction SilentlyContinue
        Disable-ScheduledTask -TaskName 'DGPU-Phase19-Worker' -ErrorAction Stop | Out-Null
        Write-Host 'Disabled old Phase19 scheduled task; XML backed up. Phase20 will manage port 45420.'
    }
} else {
    Write-Warning 'If old DGPU-Phase19-Worker task is enabled, new service may see port 45420 occupied. To hand over, rerun with -TakeOverPhase19 (idle training only).'
}
# Windows no-console Python: pythonw is in the same Python install / venv.
$dir=Split-Path -Parent $Python
$pythonw=Join-Path $dir 'pythonw.exe'
if (-not (Test-Path $pythonw)) { throw 'pythonw.exe missing. Install Python with Windows launcher support or use a working virtualenv.' }
$taskName='DGPU-Phase20-Agent'
$action=New-ScheduledTaskAction -Execute $pythonw -Argument '-m studio_pairing.agent' -WorkingDirectory $rootPath
$identity=[Security.Principal.WindowsIdentity]::GetCurrent().Name
$trigger=New-ScheduledTaskTrigger -AtLogOn -User $identity
$principal=New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
$settings=New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description 'DGPU Phase20 local join agent (hidden Python, localhost UI only)' -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Start-Sleep -Seconds 3
try {
    $response=Invoke-WebRequest -Uri 'http://127.0.0.1:45430/api/agent/status' -UseBasicParsing -TimeoutSec 4
    if ($response.StatusCode -eq 200) { Write-Host 'AGENT READY: http://127.0.0.1:45430/join' }
} catch { Write-Warning 'Agent not responding. Inspect D:\Distributed-GPU\.dgpu\phase20_agent.log and task status.' }
Write-Host 'Open http://127.0.0.1:45430/join on Laptop B.'
Write-Host 'No permanent token must be typed on Laptop B; Host must approve the matching fingerprint.'
