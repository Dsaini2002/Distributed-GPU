param(
    [Parameter(Mandatory=$true)]
    [string]$BindIP,

    [string]$Controller = "http://192.168.50.1:45400"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$state = Join-Path $root ".dgpu"

New-Item -ItemType Directory -Force $state | Out-Null

$pythonPath = (& python -c "import sys; print(sys.executable)").Trim()

& $pythonPath -c "import torch, uvicorn; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"

if ($LASTEXITCODE -ne 0) {
    throw "Activate the CUDA-enabled Python environment first"
}

$secret = Read-Host "Enter cluster token ONCE" -AsSecureString

$config = @{
    python = $pythonPath
    bind_ip = $BindIP
    controller = $Controller
    protected_token = ConvertFrom-SecureString -SecureString $secret
}

$configPath = Join-Path $state "worker_config.json"

$config | ConvertTo-Json |
    Set-Content -Path $configPath -Encoding UTF8

$runner = Join-Path $PSScriptRoot "run_worker_saved.ps1"

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$runner`"" `
    -WorkingDirectory $root

$trigger = New-ScheduledTaskTrigger `
    -AtLogOn `
    -User "$env:USERDOMAIN\$env:USERNAME"

Register-ScheduledTask `
    -TaskName "DistributedGPUWorker" `
    -Action $action `
    -Trigger $trigger `
    -Description "Start DGPU worker when this user signs in" `
    -Force | Out-Null

Write-Host "WORKER AUTO-START CONFIGURED" -ForegroundColor Green
Write-Host "Python: $pythonPath"
Write-Host "Bind IP: $BindIP"
Write-Host "Use Start-ScheduledTask -TaskName DistributedGPUWorker to test"
