"""GUI launcher for the desktop beta. No console window is required."""
import ctypes
import os
import sys
import traceback
from pathlib import Path

root = Path(__file__).resolve().parent
os.chdir(root)
sys.path.insert(0, str(root))
try:
    from desktop_app.main import main
    main()
except Exception as exc:
    log_folder = Path(os.getenv('LOCALAPPDATA', str(Path.home()))) / 'DistributedGPUStudio'
    message = 'Unable to start Distributed-GPU Studio. See the local launch log for details.'
    try:
        log_folder.mkdir(parents=True, exist_ok=True)
        (log_folder / 'desktop-launch-error.log').write_text(traceback.format_exc(), encoding='utf-8')
        message += '\n\nLog: ' + str(log_folder / 'desktop-launch-error.log')
    except OSError:
        pass
    try:
        ctypes.windll.user32.MessageBoxW(0, message, 'Distributed-GPU Studio', 0x10)
    except Exception:
        raise
