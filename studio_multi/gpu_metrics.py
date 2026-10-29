import csv, io, subprocess
from datetime import datetime, timezone

def gpu_snapshot():
    result={'gpus':[], 'sampled_at':datetime.now(timezone.utc).isoformat(), 'error':None}
    try:
        p=subprocess.run(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.free,memory.total,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if p.returncode: raise RuntimeError()
        def number(v):
            try: return float(v.strip())
            except ValueError: return None
        for row in csv.reader(io.StringIO(p.stdout)):
            if len(row)==7:
                result['gpus'].append(dict(zip(('id','name','util','used','free','total','temperature'),[row[0].strip(),row[1].strip()]+[number(v) for v in row[2:]])))
        if not result['gpus']: result['error']='No NVIDIA GPUs reported'
    except (OSError, subprocess.TimeoutExpired, RuntimeError):
        result['error']='NVIDIA telemetry unavailable; check driver and nvidia-smi'
    return result
