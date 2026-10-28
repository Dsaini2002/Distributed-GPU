# Read-only current state. Can be run from either laptop.
param([string]$Root = 'D:\Distributed-GPU')
$rootPath = (Resolve-Path -LiteralPath $Root).Path
Write-Host "COMPUTER: $env:COMPUTERNAME" -ForegroundColor Cyan
foreach ($taskName in @('DGPU-AutoHeal','DGPU-AutoHeal-Keeper','DistributedGPUWorker')) {
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($task) {
        $info = Get-ScheduledTaskInfo -TaskName $taskName -ErrorAction SilentlyContinue
        Write-Host ("TASK {0}: {1}; LastTaskResult={2}" -f $taskName,$task.State,$info.LastTaskResult)
    } else { Write-Host "TASK $taskName missing" }
}
foreach ($port in @(45400,45410)) {
    $listeners = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
    if ($listeners.Count -eq 0) { Write-Host "PORT ${port}: not listening" }
    else { foreach ($listener in $listeners) { Write-Host ("PORT {0}: {1}, PID={2}" -f $port,$listener.LocalAddress,$listener.OwningProcess) } }
}
foreach ($name in @('autoheal.log','worker.log','controller-autoheal.log')) {
    $path = Join-Path $rootPath ('.dgpu\' + $name)
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    if (Test-Path -LiteralPath $path) { Get-Content -LiteralPath $path -Tail 12 }
    else { Write-Host 'Not found' }
}
