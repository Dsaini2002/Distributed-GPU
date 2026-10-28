DISTRIBUTED-GPU STUDIO -- PHASE 17 RELIABILITY UPDATE

This update upgrades the existing Phase 16B AutoHeal supervisor without touching
model code, existing checkpoints, or saved tokens.

CHANGES
1. Keeps exactly one supervisor per laptop, old task remains disabled.
2. Controller (Laptop A) and GPU worker restart if their listening process exits.
3. Prevents the previous unsafe auto-restart rule: a stale heartbeat/log ALONE
   will never kill a listening worker (which might be in the middle of training).
4. Installs Hidden-window Task Scheduler actions for AutoHeal and Keeper.
5. Adds a 3-minute TCP-stability test, read-only diagnostics, and a deliberate
   failure-injection test (only when not training).

IMPORTANT LIMITATIONS
- This package has NOT been exercised on your laptops; validate in your environment.
- TCP availability != successful model training or active heartbeat.
- It does NOT implement full job-level model-parallel resume/recovery.
- Existing Phase 15 checkpoint-resume must be verified separately and must not
  be assumed to work automatically when a GPU worker crashes during a batch.
- If a worker keeps port 45410 open but stops heartbeating, AutoHeal reports
  stale logs but does not forcibly terminate live training.
- Controller/worker auto-start still needs that Windows user logged in.
- Run only on trusted private Ethernet; HTTP and socket traffic are not encrypted.
- Run installer only when NO training is in progress.

INSTALLATION (ON BOTH LAPTOPS)
1. Download ZIP to D:\Distributed-GPU\DGPU-Phase17-Reliability.zip
2. Extract with:
     Expand-Archive .\DGPU-Phase17-Reliability.zip .\DGPU-Phase17-Reliability -Force
3. Under the SAME Windows user that created the DPAPI encrypted token:
     powershell -NoProfile -ExecutionPolicy Bypass -File .\DGPU-Phase17-Reliability\Install-DGPU-AutoHeal.ps1 -Root D:\Distributed-GPU -Role Auto
   Install again on Laptop B.
4. Inspect diagnostics ON BOTH laptops:
     powershell -NoProfile -ExecutionPolicy Bypass -File .\DGPU-Phase17-Reliability\Get-DGPU-Phase17-Diagnostics.ps1
5. On Laptop A test both worker ports/controller for 3 min:
     powershell -NoProfile -ExecutionPolicy Bypass -File .\DGPU-Phase17-Reliability\Test-DGPU-Phase17.ps1 -Minutes 3
6. OPTIONAL and only while NO training is running, run on each laptop:
     powershell -NoProfile -ExecutionPolicy Bypass -File .\DGPU-Phase17-Reliability\Test-DGPU-Reconnect.ps1 -ConfirmNoActiveTraining
7. Open http://192.168.50.1:45400/ and confirm 2 computers / 2 GPUs ONLINE.
8. Then run a short two-GPU training in http://192.168.50.1:45400/train
   and check that both stages are used, all epochs complete and a new checkpoint
   is produced. This is required before calling Phase 17 completed.

If something fails, share diagnostics and .dgpu\phase17_report.json
but never share .dgpu\worker_config.json or cluster token.
