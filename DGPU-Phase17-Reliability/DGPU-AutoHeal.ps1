# Distributed-GPU Studio: persistent single-instance supervisor.
# Windows PowerShell 5.1. Does NOT store or print the cluster token.
param(
    [string]$Root = 'D:\Distributed-GPU',
    [ValidateSet('Controller','Worker')][string]$Role = 'Worker'
)

$ErrorActionPreference = 'Stop'
$rootPath = (Resolve-Path -LiteralPath $Root).Path
$dgpuDir = Join-Path $rootPath '.dgpu'
$logPath = Join-Path $dgpuDir 'autoheal.log'
$workerLog = Join-Path $dgpuDir 'worker.log'
$workerRunner = Join-Path $rootPath 'studio\run_worker_saved.ps1'
$controllerRunner = Join-Path $rootPath 'studio\DGPU-ControllerRunner.ps1'
$hostIP = if ($Role -eq 'Controller') { '192.168.50.1' } else { '192.168.50.2' }
$psExe = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'

function Write-DgpuLog([string]$Message) {
    $line = ('{0:yyyy-MM-dd HH:mm:ss} [{1}] {2}' -f (Get-Date), $env:COMPUTERNAME, $Message)
    try { Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8 } catch { }
}

function Get-Listeners([int]$Port, [string]$IP) {
    @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
      Where-Object { $_.LocalAddress -in @($IP, '0.0.0.0', '::') })
}

function Get-WorkerProcesses {
    @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue |
      Where-Object { $_.CommandLine -match '(?i)(?:-m\s+)studio\.worker_run(?:\s|$)' })
}

function Start-DetachedScript([string]$Path) {
    $cmdline = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}"' -f $Path
    Start-Process -FilePath $psExe -ArgumentList $cmdline -WorkingDirectory $rootPath -WindowStyle Hidden -PassThru
}

if (-not (Test-Path -LiteralPath $workerRunner)) { throw "Missing worker runner: $workerRunner" }
if (-not (Test-Path -LiteralPath (Join-Path $dgpuDir 'worker_config.json'))) {
    throw 'Missing .dgpu\worker_config.json. Configure your worker before installing autoheal.'
}

$mutexName = 'Local\DGPU-AutoHeal-' + $env:COMPUTERNAME
$mutex = New-Object System.Threading.Mutex($false, $mutexName)
$ownsMutex = $false
try {
    try { $ownsMutex = $mutex.WaitOne(0) }
    catch [System.Threading.AbandonedMutexException] { $ownsMutex = $true }
    if (-not $ownsMutex) {
        Write-DgpuLog 'Another autoheal supervisor is already active; avoiding duplicate.'
        exit 0
    }

    Write-DgpuLog "Phase17 supervisor started. Role=$Role; worker address=$($hostIP):45410. Active worker protected from stale-log kills."
    $lastWorkerLaunch = [datetime]::MinValue
    $lastControllerLaunch = [datetime]::MinValue
    $lastMessage = ''

    while ($true) {
        try {
            if ($Role -eq 'Controller') {
                $controllerListening = @(Get-Listeners 45400 '192.168.50.1')
                if ($controllerListening.Count -eq 0 -and
                    ((Get-Date) - $lastControllerLaunch).TotalSeconds -ge 30) {
                    $lastControllerLaunch = Get-Date
                    $p = Start-DetachedScript -Path $controllerRunner
                    Write-DgpuLog "Controller not listening: launched controller runner PID $($p.Id)."
                }
            }

            $listeners = @(Get-Listeners 45410 $hostIP)
            $workers = @(Get-WorkerProcesses)

            if ($listeners.Count -gt 0) {
                $ownerIds = @($listeners | ForEach-Object { [int]$_.OwningProcess })
                $isOurWorker = @($workers | Where-Object { $ownerIds -contains [int]$_.ProcessId }).Count -gt 0
                if (-not $isOurWorker) {
                    if ($lastMessage -ne 'foreign-port') {
                        Write-DgpuLog "Port 45410 is occupied by an unrecognized process (PID(s): $($ownerIds -join ',')). Not killing it."
                        $lastMessage = 'foreign-port'
                    }
                } else {
                    $lastMessage = 'online-port'
                    # If the API remains up but its heartbeat thread dies, recover after 150s.
                    # Do not restart a worker when the controller is unreachable.
                    $controllerUp = @(Get-Listeners 45400 '192.168.50.1').Count -gt 0
                    if (-not $controllerUp -and $Role -eq 'Worker') {
                        # Remote connectivity (no HTTP request/secret required).
                        $client = New-Object System.Net.Sockets.TcpClient
                        try {
                            $result = $client.BeginConnect('192.168.50.1', 45400, $null, $null)
                            $controllerUp = $result.AsyncWaitHandle.WaitOne(1200, $false)
                            if ($controllerUp) { $client.EndConnect($result) }
                        } catch { $controllerUp = $false }
                        finally { $client.Dispose() }
                    }
                    # Safety for model-parallel training: do not kill a worker merely
                    # because the log has stopped updating. A live worker may be
                    # executing a long CUDA operation or an active training job.
                    # The supervisor still restarts the worker if its listener exits.
                    if ($controllerUp -and (Test-Path -LiteralPath $workerLog)) {
                        $age = ((Get-Date) - (Get-Item -LiteralPath $workerLog).LastWriteTime).TotalSeconds
                        if ($age -gt 150 -and $lastMessage -ne 'heartbeat-stale') {
                            Write-DgpuLog "WARNING: Worker log stale for $([int]$age)s while port 45410 listens. Preserving worker to avoid aborting training; inspect heartbeat/controller."
                            $lastMessage = 'heartbeat-stale'
                        } elseif ($age -le 150) {
                            $lastMessage = 'online-port'
                        }
                    }
                }
            } elseif (((Get-Date) - $lastWorkerLaunch).TotalSeconds -ge 30) {
                # Only terminate our own stale worker-run processes. Never kill arbitrary Python.
                foreach ($w in $workers) {
                    $aliveSeconds = 60
                    try {
                        $proc = Get-Process -Id $w.ProcessId -ErrorAction Stop
                        $aliveSeconds = ((Get-Date) - $proc.StartTime).TotalSeconds
                    } catch { }
                    if ($aliveSeconds -ge 30) {
                        Write-DgpuLog "Stale worker PID $($w.ProcessId) has no listening socket; stopping it."
                        Stop-Process -Id $w.ProcessId -Force -ErrorAction SilentlyContinue
                    }
                }
                $lastWorkerLaunch = Get-Date
                $p = Start-DetachedScript -Path $workerRunner
                Write-DgpuLog "Worker was not listening: launched saved worker runner PID $($p.Id)."
                $lastMessage = 'started-worker'
            }
        } catch {
            Write-DgpuLog "Supervisor check error: $($_.Exception.Message)"
        }
        Start-Sleep -Seconds 15
    }
} finally {
    if ($ownsMutex) { try { $mutex.ReleaseMutex() } catch { } }
    $mutex.Dispose()
}
