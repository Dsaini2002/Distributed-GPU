# One-time install per Windows laptop. Copy this entire package to each laptop.
# Run under SAME Windows account that configured .dgpu\worker_config.json.
param(
    [string]$Root = 'D:\Distributed-GPU',
    [ValidateSet('Auto','Controller','Worker')][string]$Role = 'Auto'
)
$ErrorActionPreference = 'Stop'
$rootPath = (Resolve-Path -LiteralPath $Root -ErrorAction Stop).Path
$studio = Join-Path $rootPath 'studio'
$dgpu = Join-Path $rootPath '.dgpu'
$cfgPath = Join-Path $dgpu 'worker_config.json'
if (-not (Test-Path -LiteralPath $cfgPath)) { throw "Missing $cfgPath. Run setup_worker.ps1 first." }
if (-not (Test-Path -LiteralPath (Join-Path $studio 'run_worker_saved.ps1'))) {
    throw "Missing studio\run_worker_saved.ps1 in $rootPath"
}

$localIPs = @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue | ForEach-Object { $_.IPAddress })
if ($Role -eq 'Auto') {
    if ($localIPs -contains '192.168.50.1') { $Role = 'Controller' }
    elseif ($localIPs -contains '192.168.50.2') { $Role = 'Worker' }
    else { throw 'Cannot detect Ethernet IP 192.168.50.1 or 192.168.50.2. Set the LAN IP and try again.' }
}
Write-Host "Installing DGPU Phase 17 Reliability for $env:COMPUTERNAME ($Role)..." -ForegroundColor Cyan
Write-Host 'IMPORTANT: Install only when no training job is running.' -ForegroundColor Yellow

$package = Split-Path -Parent $MyInvocation.MyCommand.Path
foreach ($name in @('DGPU-AutoHeal.ps1','DGPU-ControllerRunner.ps1','DGPU-Keeper.ps1')) {
    $source = Join-Path $package $name
    $target = Join-Path $studio $name
    if (-not (Test-Path -LiteralPath $source)) { throw "Package file missing: $source" }
    if (-not [string]::Equals($source, $target, [StringComparison]::OrdinalIgnoreCase)) {
        Copy-Item -LiteralPath $source -Destination $target -Force
    }
}

# Controller may have used a different Python than the worker's GPU venv.
$config = Get-Content -LiteralPath $cfgPath -Raw | ConvertFrom-Json
$controllerPython = [string]$config.python
if ($Role -eq 'Controller') {
    $candidates = @($controllerPython)
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand) { $candidates += [string]$pythonCommand.Source }
    $valid = $false
    foreach ($candidate in $candidates | Select-Object -Unique) {
        if (-not $candidate) { continue }
        try {
            & $candidate -c 'import fastapi, uvicorn' 2>$null
            if ($LASTEXITCODE -eq 0) {
                $controllerPython = $candidate
                $valid = $true
                break
            }
        } catch { }
    }
    if (-not $valid) {
        throw 'Cannot find Python with fastapi and uvicorn. Fix controller Python dependencies before installing.'
    }
}
@{ role = $Role; controller_python = $controllerPython } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $dgpu 'autoheal_settings.json') -Encoding UTF8

# Clean up prior autoheal task if this is a reinstall.
foreach ($name in @('DGPU-AutoHeal','DGPU-AutoHeal-Keeper')) {
    $existing = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if ($null -ne $existing) {
        Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction Stop
    }
}

# Disable the old worker autorun, or it will spawn competing workers on logon.
$legacy = Get-ScheduledTask -TaskName 'DistributedGPUWorker' -ErrorAction SilentlyContinue
if ($null -ne $legacy) {
    Stop-ScheduledTask -TaskName 'DistributedGPUWorker' -ErrorAction SilentlyContinue
    Disable-ScheduledTask -TaskName 'DistributedGPUWorker' -ErrorAction Stop | Out-Null
    Write-Host 'Old DistributedGPUWorker task disabled to prevent duplicate workers.'
}

# Stop ONLY confirmed distributed-GPU worker Python processes, and dedicated
# PowerShell launchers. The manually running controller is deliberately kept.
$oldWorkers = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
        ($_.Name -match '(?i)^pythonw?\.exe$' -and $_.CommandLine -match '(?i)(?:-m\s+)studio\.worker_run(?:\s|$)') -or
        ($_.Name -match '(?i)^powershell\.exe$' -and $_.CommandLine -match '(?i)-File\s+"?[^\"]*run_worker_saved\.ps1')
    })
foreach ($w in $oldWorkers) {
    Write-Host "Stopping old confirmed worker process PID $($w.ProcessId)"
    Stop-Process -Id $w.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 2

$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$exe = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable -MultipleInstances IgnoreNew `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
$logon = New-ScheduledTaskTrigger -AtLogOn -User $user
$supervisorPath = Join-Path $studio 'DGPU-AutoHeal.ps1'
$supervisorArgs = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}" -Root "{1}" -Role {2}' -f $supervisorPath,$rootPath,$Role
$action = New-ScheduledTaskAction -Execute $exe -Argument $supervisorArgs -WorkingDirectory $rootPath
Register-ScheduledTask -TaskName 'DGPU-AutoHeal' -Action $action -Trigger $logon `
    -Settings $settings -Principal $principal -Description 'Self-healing Distributed GPU worker and controller' -Force | Out-Null

# The keeper restarts the supervisor if Windows or an unexpected error stops it.
# 1-minute periodic check; 365-day task trigger (reinstall once annually).
$keeperPath = Join-Path $studio 'DGPU-Keeper.ps1'
$keeperArgs = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}"' -f $keeperPath
$keeperAction = New-ScheduledTaskAction -Execute $exe -Argument $keeperArgs -WorkingDirectory $rootPath
$repeat = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 1) -RepetitionDuration (New-TimeSpan -Days 365)
$keeperSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 1) -StartWhenAvailable -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName 'DGPU-AutoHeal-Keeper' -Action $keeperAction -Trigger $repeat `
    -Settings $keeperSettings -Principal $principal -Description 'Restart DGPU AutoHeal supervisor if stopped' -Force | Out-Null

Start-ScheduledTask -TaskName 'DGPU-AutoHeal'
Start-Sleep -Seconds 6
Write-Host "`n=== TASKS ===" -ForegroundColor Cyan
Get-ScheduledTask -TaskName 'DGPU-AutoHeal','DGPU-AutoHeal-Keeper' |
    Select-Object TaskName,State | Format-Table -AutoSize
Write-Host "`n=== AUTOHEAL LOG ===" -ForegroundColor Cyan
Get-Content -LiteralPath (Join-Path $dgpu 'autoheal.log') -Tail 12 -ErrorAction SilentlyContinue
Write-Host "`nSetup complete on $env:COMPUTERNAME. It may take a few seconds to reconnect." -ForegroundColor Green
Write-Host 'Do not start the old DistributedGPUWorker task; AutoHeal now manages the worker.'
Write-Host 'Phase 17 safety: heartbeat log staleness alone will NOT terminate an active worker.'
