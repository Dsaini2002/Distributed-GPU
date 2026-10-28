# Controller launcher using Laptop A's DPAPI protected saved worker token.
# Requires the same Windows account used during worker setup.
param([string]$Root = 'D:\Distributed-GPU')

$rootPath = (Resolve-Path -LiteralPath $Root).Path
Set-Location -LiteralPath $rootPath
$configPath = Join-Path $rootPath '.dgpu\worker_config.json'
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$log = Join-Path $rootPath '.dgpu\controller-autoheal.log'

try {
    $secure = ConvertTo-SecureString $config.protected_token
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try { $env:DGPU_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }

    $settings = Get-Content -LiteralPath (Join-Path $rootPath '.dgpu\autoheal_settings.json') -Raw | ConvertFrom-Json
    $python = [string]$settings.controller_python
    "$(Get-Date -Format s) Controller launcher starting (no token logged)" | Out-File -FilePath $log -Append -Encoding UTF8
    & $python -u -m uvicorn studio.server:app --host 192.168.50.1 --port 45400 *>> $log
} catch {
    "$(Get-Date -Format s) Controller launcher error: $($_.Exception.Message)" | Out-File -FilePath $log -Append -Encoding UTF8
    throw
} finally {
    Remove-Item Env:\DGPU_TOKEN -ErrorAction SilentlyContinue
}
