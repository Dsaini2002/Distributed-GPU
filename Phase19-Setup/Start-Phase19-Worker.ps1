param([string]$Root = 'D:\Distributed-GPU', [string]$BindIP = '192.168.50.2')
$ErrorActionPreference = 'Stop'
Set-Location $Root
$python = (Get-Content '.dgpu\phase19_python.txt' -Raw).Trim()
if (-not (Test-Path $python)) { throw "Python path does not exist: $python" }
$config = Get-Content '.dgpu\worker_config.json' -Raw | ConvertFrom-Json
$secure = ConvertTo-SecureString $config.protected_token
$ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try {
    $env:DGPU_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
}
Remove-Variable -Name secure -ErrorAction SilentlyContinue
$log = Join-Path $Root '.dgpu\phase19_worker.log'
Add-Content $log "===== $(Get-Date -Format o) P19 worker launching on $BindIP`:45420 ====="
& $python -u -m uvicorn studio_multi.worker:app --host $BindIP --port 45420 *>> $log
