$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $root ".dgpu\worker_config.json"

if (-not (Test-Path $configPath)) {
    throw "Worker not configured. Run setup_worker.ps1"
}

$config = Get-Content $configPath -Raw | ConvertFrom-Json

# Windows DPAPI: only original Windows user can decrypt it
$secure = ConvertTo-SecureString $config.protected_token

$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)

try {
    $env:DGPU_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
}

$env:DGPU_CONTROLLER = $config.controller
$env:DGPU_WORKER_BIND_IP = $config.bind_ip

Set-Location $root

$log = Join-Path $root ".dgpu\worker.log"

& $config.python -u -m studio.worker_run *>> $log
