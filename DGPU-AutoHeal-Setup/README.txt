DISTRIBUTED-GPU STUDIO - PHASE 16B AUTOHEAL

What this package does
- Reuses existing DPAPI encrypted cluster token; no new token creation.
- Keeps one GPU worker per laptop running; prevents duplicate startup from old worker task.
- Monitors GPU worker TCP port 45410 every ~15 seconds.
- Restarts worker if it exits or stops updating its log for >150s while controller is reachable.
- On Laptop A, starts/restores Controller TCP port 45400 using saved token.
- Creates a Windows Task Scheduler supervisor (at user logon) and once/minute keeper.
- Allows battery power and removes scheduled execution timeout for the supervisor.
- Logs: D:\Distributed-GPU\.dgpu\autoheal.log and controller-autoheal.log.
- Does not change training/model code or delete checkpoints.

INSTALL: EXTRACT THIS ZIP ON BOTH LAPTOPS (to Downloads or D:\Distributed-GPU).
Run under the SAME Windows user that ran studio\setup_worker.ps1.
Use a PowerShell terminal AS ADMINISTRATOR if Task Scheduler access is denied.

Laptop A (192.168.50.1 / GTX 1650):
cd <folder containing this package>
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Install-DGPU-AutoHeal.ps1 -Root D:\Distributed-GPU -Role Controller

Laptop B (192.168.50.2 / RTX 2050):
cd <folder containing this package>
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Install-DGPU-AutoHeal.ps1 -Root D:\Distributed-GPU -Role Worker

Check on EACH laptop:
Get-ScheduledTask -TaskName 'DGPU-AutoHeal','DGPU-AutoHeal-Keeper' | Select-Object TaskName,State
Get-Content D:\Distributed-GPU\.dgpu\autoheal.log -Tail 20
Get-NetTCPConnection -LocalPort 45410 -State Listen -ErrorAction SilentlyContinue

Check from Laptop A:
Test-NetConnection 192.168.50.1 -Port 45400
Test-NetConnection 192.168.50.1 -Port 45410
Test-NetConnection 192.168.50.2 -Port 45410

Open: http://192.168.50.1:45400/
Verify 2 computers / 2 GPUs online over a few minutes BEFORE retraining.

IMPORTANT
- This is a recovery mechanism, not a verified fix for every crash.
- If Python itself repeatedly fails to start, inspect worker.log and autoheal.log.
- If the controller is manually run with a DIFFERENT token, authentication can still fail.
- The task starts at WINDOWS USER LOGON, not before login.
- The keeper trigger is configured for 365 days; reinstall/refresh annually.
- Keep this local cluster isolated on your trusted Ethernet; HTTP is not encrypted.
- Do not publish saved token or worker_config.json; rotate any previously shared token.
- With two physical GPUs online, model partitioning still depends on planner/model capabilities.
