# Recurring fallback: if Windows stops the long-running supervisor, restart it.
$taskName = 'DGPU-AutoHeal'
try {
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
    if ($task.State -ne 'Running') {
        Start-ScheduledTask -TaskName $taskName -ErrorAction Stop
    }
} catch {
    $log = 'D:\Distributed-GPU\.dgpu\autoheal.log'
    "$(Get-Date -Format s) Keeper error: $($_.Exception.Message)" | Out-File -FilePath $log -Append -Encoding UTF8
}
