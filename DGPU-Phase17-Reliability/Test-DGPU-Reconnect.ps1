# OPTIONAL controlled failure-injection test, use only when NO training is running.
# Run this ON the laptop whose worker you want to test. Does not touch controller.
param(
    [switch]$ConfirmNoActiveTraining,
    [ValidateRange(30,180)][int]$TimeoutSeconds = 90
)
$ErrorActionPreference = 'Stop'
if (-not $ConfirmNoActiveTraining) {
    throw 'Refusing to kill worker without -ConfirmNoActiveTraining. Stop training first.'
}
$supervisor = Get-ScheduledTask -TaskName 'DGPU-AutoHeal' -ErrorAction Stop
if ($supervisor.State -ne 'Running') { throw "Supervisor is not running: $($supervisor.State)" }
$listeners = @(Get-NetTCPConnection -LocalPort 45410 -State Listen -ErrorAction SilentlyContinue)
if ($listeners.Count -ne 1) { throw "Expected exactly one TCP listener on port 45410, got $($listeners.Count)." }
$pidToStop = [int]$listeners[0].OwningProcess
$proc = Get-CimInstance Win32_Process -Filter "ProcessId=$pidToStop" -ErrorAction Stop
if (-not ($proc.CommandLine -match '(?i)(?:-m\s+)studio\.worker_run(?:\s|$)')) {
    throw "Listener PID $pidToStop is not a verified Distributed-GPU worker. Refusing to stop."
}
Write-Host "Simulating worker crash: PID $pidToStop. AutoHeal should restore it." -ForegroundColor Yellow
Stop-Process -Id $pidToStop -Force
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$recovered = $false
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 3
    $newListeners = @(Get-NetTCPConnection -LocalPort 45410 -State Listen -ErrorAction SilentlyContinue)
    foreach ($listener in $newListeners) {
        if ([int]$listener.OwningProcess -ne $pidToStop) {
            $replacement = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)" -ErrorAction SilentlyContinue
            if ($replacement -and ($replacement.CommandLine -match '(?i)(?:-m\s+)studio\.worker_run(?:\s|$)')) {
                Write-Host "PASS: Worker restarted, replacement PID=$($listener.OwningProcess)" -ForegroundColor Green
                $recovered=$true
                break
            }
        }
    }
    if ($recovered) { break }
}
if (-not $recovered) {
    Write-Warning "FAIL: Worker did not recover within $TimeoutSeconds seconds. Inspect .dgpu\autoheal.log and .dgpu\worker.log."
    exit 1
}
Write-Host 'Verify dashboard heartbeat ONLINE before starting training.'
