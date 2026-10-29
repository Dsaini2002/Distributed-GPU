# One-click Windows beta desktop shell setup for EXISTING DGPU machines.
# No token mutation, no git, no GPU driver modifications, no backend restarts.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

function Show-ErrorDialog([string] $Message) {
    [System.Windows.Forms.MessageBox]::Show($Message, 'Distributed-GPU Studio Setup',
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Error) | Out-Null
}

$ui = New-Object System.Windows.Forms.Form
$ui.Text = 'Distributed-GPU Studio Setup'
$ui.Size = [System.Drawing.Size]::new(610,275)
$ui.StartPosition = 'CenterScreen'
$ui.FormBorderStyle = 'FixedDialog'
$ui.MaximizeBox = $false
$ui.MinimizeBox = $false
$ui.TopMost = $true
$ui.BackColor = [System.Drawing.Color]::FromArgb(13,24,42)
$ui.ForeColor = [System.Drawing.Color]::White
$ui.Font = [System.Drawing.Font]::new('Segoe UI',10)

$title = New-Object System.Windows.Forms.Label
$title.Text = 'Distributed-GPU Studio'
$title.Font = [System.Drawing.Font]::new('Segoe UI',18,[System.Drawing.FontStyle]::Bold)
$title.Location = [System.Drawing.Point]::new(25,18)
$title.Size = [System.Drawing.Size]::new(550,40)
$ui.Controls.Add($title)
$description = New-Object System.Windows.Forms.Label
$description.Text = 'Install your GPU Studio desktop app. No commands needed on this computer.'
$description.Location = [System.Drawing.Point]::new(27,66)
$description.Size = [System.Drawing.Size]::new(535,43)
$ui.Controls.Add($description)
$status = New-Object System.Windows.Forms.Label
$status.Text = 'Ready to update your desktop application.'
$status.Location = [System.Drawing.Point]::new(27,117)
$status.Size = [System.Drawing.Size]::new(535,41)
$ui.Controls.Add($status)
$button = New-Object System.Windows.Forms.Button
$button.Text = 'Install && Open App'
$button.BackColor = [System.Drawing.Color]::FromArgb(40,124,224)
$button.ForeColor = [System.Drawing.Color]::White
$button.FlatStyle = 'Flat'
$button.Location = [System.Drawing.Point]::new(25,180)
$button.Size = [System.Drawing.Size]::new(225,37)
$ui.Controls.Add($button)
$close = New-Object System.Windows.Forms.Button
$close.Text = 'Cancel'
$close.Location = [System.Drawing.Point]::new(268,180)
$close.Size = [System.Drawing.Size]::new(110,37)
$close.Add_Click({$ui.Close()})
$ui.Controls.Add($close)

function Update-Status([string]$Text) {
    $status.Text = $Text
    $ui.Refresh()
    [System.Windows.Forms.Application]::DoEvents()
}

function Find-WorkingPython([string] $Root) {
    $candidates = @()
    $candidates += (Join-Path $Root '.venv\Scripts\python.exe')
    foreach ($version in @('Python313','Python312','Python311','Python310')) {
        $candidates += (Join-Path $env:LOCALAPPDATA "Programs\Python\$version\python.exe")
    }
    foreach ($name in @('python.exe','py.exe')) {
        $found = Get-Command $name -ErrorAction SilentlyContinue
        if ($found) { $candidates += $found.Source }
    }
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $candidate)) { continue }
        try {
            $ver = & $candidate -c 'import sys; print(sys.version_info.major, sys.version_info.minor)' 2>$null
            if ($LASTEXITCODE -eq 0 -and "$ver" -match '^3 (1[0-9]|9)$') { return $candidate }
        } catch { }
    }
    return $null
}

$button.Add_Click({
    $button.Enabled = $false
    $close.Enabled = $false
    try {
        # The ZIP can be extracted anywhere (Downloads, Desktop, D:\...).
        # Never mistake the extraction folder for the already-installed GPU backend.
        $sourceDir = $PSScriptRoot
        $projectRoot = $null
        $candidates = @(
            $env:DGPU_PROJECT_ROOT,
            'D:\Distributed-GPU',
            (Split-Path $sourceDir -Parent),
            (Join-Path $env:USERPROFILE 'Distributed-GPU')
        )
        foreach ($candidate in $candidates) {
            if ([string]::IsNullOrWhiteSpace($candidate)) { continue }
            $backendExists = (Test-Path (Join-Path $candidate 'studio\server.py')) -or
                (Test-Path (Join-Path $candidate 'studio_pairing\agent.py'))
            if ($backendExists) {
                $projectRoot = (Resolve-Path -LiteralPath $candidate).Path
                break
            }
        }
        if (-not $projectRoot) {
            Update-Status 'Please locate your existing Distributed-GPU project folder...'
            $folderPicker = New-Object System.Windows.Forms.FolderBrowserDialog
            $folderPicker.Description = 'Select your existing Distributed-GPU project folder'
            $folderPicker.ShowNewFolderButton = $false
            if ($folderPicker.ShowDialog($ui) -ne [System.Windows.Forms.DialogResult]::OK) {
                throw 'An existing Distributed-GPU backend is required for this beta. No files were changed.'
            }
            $candidate = $folderPicker.SelectedPath
            $backendExists = (Test-Path (Join-Path $candidate 'studio\server.py')) -or
                (Test-Path (Join-Path $candidate 'studio_pairing\agent.py'))
            if (-not $backendExists) {
                throw 'That folder does not contain the Distributed-GPU backend. Pick the folder containing studio or studio_pairing. A fresh-PC installer is not ready yet.'
            }
            $projectRoot = $candidate
        }
        Update-Status 'Finding the existing Python runtime...'
        $python = Find-WorkingPython $projectRoot
        if (-not $python) {
            throw 'No working Python 3.10+ was found. This test installer needs your current GPU backend.'
        }
        Update-Status 'Checking desktop dependencies...'
        $checkLog = Join-Path $env:TEMP 'dgpu-desktop-check-out.log'
        $checkErr = Join-Path $env:TEMP 'dgpu-desktop-check-error.log'
        $check = Start-Process -FilePath $python -ArgumentList '-c "import webview"' -Wait -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput $checkLog -RedirectStandardError $checkErr
        if ($check.ExitCode -ne 0) {
            Update-Status 'Installing desktop UI dependency. Internet required. This may take a little time...'
            $outLog = Join-Path $env:TEMP 'dgpu-desktop-pip-out.log'
            $errLog = Join-Path $env:TEMP 'dgpu-desktop-pip-error.log'
            $args = '-m pip install --disable-pip-version-check --default-timeout 90 --retries 5 "pywebview>=5,<7"'
            $install = Start-Process -FilePath $python -ArgumentList $args -PassThru -WindowStyle Hidden `
                -RedirectStandardOutput $outLog -RedirectStandardError $errLog
            while (-not $install.HasExited) {
                [System.Windows.Forms.Application]::DoEvents()
                Start-Sleep -Milliseconds 250
                $install.Refresh()
            }
            if ($install.ExitCode -ne 0) {
                throw "Could not download the desktop UI dependency. Check the internet connection and retry. Error log: $errLog"
            }
        }
        Update-Status 'Installing application files to your user profile...'
        # Copy only desktop shell files; do not touch existing GPU source, tokens,
        # checkpoints, task scheduler jobs or training services.
        $appDir = Join-Path $env:LOCALAPPDATA 'Programs\Distributed-GPU Studio'
        if (-not [string]::Equals(
                [IO.Path]::GetFullPath($sourceDir).TrimEnd('\'),
                [IO.Path]::GetFullPath($appDir).TrimEnd('\'),
                [StringComparison]::OrdinalIgnoreCase)) {
            if (-not (Test-Path -LiteralPath $appDir)) {
                New-Item -ItemType Directory -Path $appDir -Force | Out-Null
            }
            foreach ($item in (Get-ChildItem -LiteralPath $sourceDir -Force)) {
                if ($item.Name -eq 'tests' -or $item.Name -eq '.git') { continue }
                Copy-Item -LiteralPath $item.FullName -Destination $appDir -Force -Recurse
            }
        }
        Update-Status 'Creating your desktop and Start Menu shortcuts...'
        $pyw = Join-Path (Split-Path $python -Parent) 'pythonw.exe'
        if (-not (Test-Path -LiteralPath $pyw)) {
            throw 'pythonw.exe is missing from the existing Python installation.'
        }
        $launcher = Join-Path $appDir 'Launch-Desktop.pyw'
        $icon = Join-Path $appDir 'desktop_app\assets\app.ico'
        $shell = New-Object -ComObject WScript.Shell
        foreach ($path in @(
            (Join-Path ([Environment]::GetFolderPath('Desktop')) 'Distributed-GPU Studio.lnk'),
            (Join-Path ([Environment]::GetFolderPath('Programs')) 'Distributed-GPU Studio.lnk')
        )) {
            $shortcut = $shell.CreateShortcut($path)
            $shortcut.TargetPath = $pyw
            $shortcut.Arguments = '"' + $launcher + '"'
            $shortcut.WorkingDirectory = $appDir
            if (Test-Path $icon) { $shortcut.IconLocation = $icon }
            $shortcut.Description = 'Distributed-GPU Studio desktop application'
            $shortcut.Save()
        }
        Update-Status 'Installed. Opening your app...'
        Start-Process -FilePath $pyw -ArgumentList ('"' + $launcher + '"') -WorkingDirectory $appDir
        [System.Windows.Forms.MessageBox]::Show(
            'App updated. Open Distributed-GPU Studio from your desktop. Your existing GPU services should be running.',
            'Setup complete', [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Information) | Out-Null
        $ui.Close()
    } catch {
        Update-Status 'Setup could not finish.'
        Show-ErrorDialog $_.Exception.Message
        $button.Enabled = $true
        $close.Enabled = $true
    }
})
[void] $ui.ShowDialog()
