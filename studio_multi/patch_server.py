"""Conservative, idempotent controller integration; creates one backup."""
import ast
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
path = root / 'studio' / 'server.py'
if not path.is_file():
    raise SystemExit('studio/server.py missing. Use controller role on Laptop A only')
source = path.read_text(encoding='utf-8-sig')
if 'DGPU_PHASE19_BEGIN' in source:
    print('Phase19 controller router is already installed')
    raise SystemExit(0)
parsed = ast.parse(source)
if not any(isinstance(node, (ast.Assign,ast.AnnAssign)) and
           any(isinstance(t,ast.Name) and t.id=='app'
               for t in (node.targets if isinstance(node, ast.Assign) else [node.target]))
           for node in parsed.body):
    raise SystemExit('Cannot find module-level FastAPI app variable; refusing patch')
addition = ('\n# DGPU_PHASE19_BEGIN\n'
            'from studio_multi.api import router as _dgpu_phase19_router\n'
            'app.include_router(_dgpu_phase19_router)\n'
            '# DGPU_PHASE19_END\n')
ast.parse(source + addition)
backup = path.with_name('server.py.before_phase19.bak')
if not backup.exists():
    shutil.copy2(path, backup)
path.write_text(source + addition, encoding='utf-8')
print('Phase19 routes added. Backup:', backup)
