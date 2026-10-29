"""Optional Phase19 UI convenience: prefill worker addresses from an approved cluster URL.

Never changes coordinator/training APIs. Back up and patch only an exact known source line.
"""
from pathlib import Path
import sys
import time


def patch(root=None):
    base = Path(root) if root else Path(__file__).resolve().parents[1]
    source = base / 'studio_multi' / 'api.py'
    if not source.is_file():
        return 'Phase19 UI not found; skipping optional worker address prefill'
    content = source.read_text(encoding='utf-8')
    marker = '/* DGPU_P20_WORKER_PREFILL */'
    if marker in content: return 'Worker address prefill already installed'
    needle = "const $=id=>document.getElementById(id);let presets=[],jobId=null;"
    if content.count(needle) != 1:
        return 'Phase19 UI layout changed; safely skipped optional worker prefill'
    replacement = needle + '''\n/* DGPU_P20_WORKER_PREFILL */
const p20Workers = new URLSearchParams(location.search).get('workers');
if(p20Workers && /^[0-9.;:]+$/.test(p20Workers)) $('workers').value=p20Workers;
'''
    backup = source.with_name(source.name + '.before_phase20_' + str(int(time.time())) + '.bak')
    backup.write_text(content, encoding='utf-8')
    source.write_text(content.replace(needle,replacement),encoding='utf-8')
    return 'Phase19 UI worker address prefill installed (backup: '+str(backup)+')'

if __name__ == '__main__': print(patch())
