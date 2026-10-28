# Phase 17 end-to-end LAN stability check. Run on Laptop A.
# Does NOT stop, restart, or change any services.
param(
    [ValidateRange(1,30)][int]$Minutes = 3,
    [ValidateRange(3,60)][int]$IntervalSeconds = 10,
    [string]$Root = 'D:\Distributed-GPU'
)
$ErrorActionPreference = 'Stop'
$rootPath = (Resolve-Path -LiteralPath $Root).Path
$saveDir = Join-Path $rootPath '.dgpu'
if (-not (Test-Path -LiteralPath $saveDir)) { throw "Missing $saveDir" }

function Test-Tcp([string]$Address, [int]$Port, [int]$TimeoutMs=1500) {
    $client = New-Object System.Net.Sockets.TcpClient
    $async = $null
    try {
        $async = $client.BeginConnect($Address,$Port,$null,$null)
        if (-not $async.AsyncWaitHandle.WaitOne($TimeoutMs,$false)) { return $false }
        $client.EndConnect($async)
        return $true
    } catch { return $false }
    finally {
        if ($null -ne $async) { $async.AsyncWaitHandle.Close() }
        $client.Dispose()
    }
}

Write-Host "Phase 17 stability test: $Minutes minute(s), every $IntervalSeconds seconds." -ForegroundColor Cyan
Write-Host 'Do not start or stop training during this test.' -ForegroundColor Yellow
$end = (Get-Date).AddMinutes($Minutes)
$samples = New-Object 'System.Collections.Generic.List[object]'
while ((Get-Date) -lt $end) {
    $controller = Test-Tcp '192.168.50.1' 45400
    $gpuA = Test-Tcp '192.168.50.1' 45410
    $gpuB = Test-Tcp '192.168.50.2' 45410
    $record = [pscustomobject]@{
        time=(Get-Date).ToString('s')
        controller=$controller
        gtx1650=$gpuA
        rtx2050=$gpuB
        all_ports_ok=($controller -and $gpuA -and $gpuB)
    }
    $samples.Add($record)
    $status = if ($record.all_ports_ok) { 'OK' } else { 'FAILED' }
    $line = '{0} {1} CTRL={2} GTX1650={3} RTX2050={4}' -f $record.time,$status,$controller,$gpuA,$gpuB
    Write-Host $line -ForegroundColor $(if ($record.all_ports_ok) {'Green'} else {'Red'})
    if ((Get-Date).AddSeconds($IntervalSeconds) -lt $end) { Start-Sleep -Seconds $IntervalSeconds }
    else { break }
}

$failed = @($samples | Where-Object { -not $_.all_ports_ok }).Count
$report = [ordered]@{
    test='Phase17 TCP endpoint stability'
    tested_at=(Get-Date).ToString('s')
    duration_minutes=$Minutes
    interval_seconds=$IntervalSeconds
    samples=$samples.Count
    failures=$failed
    passed=($samples.Count -gt 0 -and $failed -eq 0)
    note='Port checks do not prove active heartbeat, GPU execution, or checkpoint recovery. Confirm /api/nodes and run a short 2-stage training job separately.'
    data=$samples.ToArray()
}
$path = Join-Path $saveDir 'phase17_report.json'
$report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $path -Encoding UTF8
Write-Host "Saved: $path" -ForegroundColor Cyan
if ($report.passed) {
    Write-Host "PASS: $($samples.Count) samples without a port outage." -ForegroundColor Green
} else {
    Write-Warning "FAIL: $failed/$($samples.Count) samples had disconnected endpoint(s)."
    exit 1
}

