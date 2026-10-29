# Developer shortcut only; uses existing Python. Does not touch GPU runtimes or tokens.
param([string]$Python='')
$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
if (-not $Python) {
    $cmd=Get-Command python -ErrorAction Stop
    $Python=$cmd.Source
}
& $Python -c 'import webview'
if ($LASTEXITCODE -ne 0) { throw 'pywebview is missing. Run: python -m pip install pywebview' }
& $Python -m desktop_app.main
if ($LASTEXITCODE -ne 0) { throw 'Desktop application did not start. Check the errors above.' }
