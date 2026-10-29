# Full interactive terminal. No service restarts, process kills, or token changes.
param(
    [Parameter(Mandatory=$true)][string]$Root,
    [string]$HostIP='192.168.50.1',
    [ValidateSet('host','worker')][string]$Role='host'
)
$ErrorActionPreference='Stop'
$resolvedRoot=(Resolve-Path -LiteralPath $Root).Path
Set-Location -LiteralPath $resolvedRoot
$global:DgpuTerminalPython=Join-Path $resolvedRoot '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $global:DgpuTerminalPython) {
    $env:VIRTUAL_ENV=Join-Path $resolvedRoot '.venv'
    $env:Path=(Join-Path $resolvedRoot '.venv\Scripts')+';'+$env:Path
} else {
    $global:DgpuTerminalPython=(Get-Command python -ErrorAction Stop).Source
}
$global:DgpuTerminalRoot=$resolvedRoot
$global:DgpuTerminalHost=$HostIP
$global:DgpuTerminalRole=$Role
$global:DgpuGpuConsole=Join-Path $resolvedRoot 'Desktop-App\desktop_app\gpu_console.py'
function global:gpu-status {
    & $global:DgpuTerminalPython -B $global:DgpuGpuConsole --root $global:DgpuTerminalRoot --host-ip $global:DgpuTerminalHost --role $global:DgpuTerminalRole
}
Write-Host 'Distributed-GPU | Full local PowerShell terminal' -ForegroundColor Cyan
Write-Host 'gpu-status   Show local + approved remote GPU details'
Write-Host 'python       Uses the project .venv when available'
Write-Host 'Remote GPUs require distributed training code; VRAM is not pooled.'
Write-Host 'The terminal and its jobs remain independent of the Desktop App.'
gpu-status
