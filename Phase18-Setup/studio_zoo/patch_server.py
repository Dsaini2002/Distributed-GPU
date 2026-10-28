"""Conservative opt-in integration with studio.server; backup first."""
from pathlib import Path
import ast
import shutil

ROOT = Path(__file__).resolve().parents[1]
server = ROOT / 'studio' / 'server.py'
if not server.exists():
    raise SystemExit('studio/server.py not found; put extension in D:\\Distributed-GPU first')
source = server.read_text(encoding='utf-8-sig')
if 'DGPU_MODEL_ZOO_BEGIN' in source:
    print('Model Zoo already installed. Nothing changed.')
    raise SystemExit(0)
tree = ast.parse(source, filename=str(server))
found = False
for node in tree.body:
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(t, ast.Name) and t.id == 'app' for t in targets):
            found = True
if not found:
    raise SystemExit('Expected module-level app assignment in studio/server.py; no modifications made')
addition = '\n# DGPU_MODEL_ZOO_BEGIN\nfrom studio_zoo.api import router as _dgpu_zoo_router\napp.include_router(_dgpu_zoo_router)\n# DGPU_MODEL_ZOO_END\n'
# Ensure the existing server remains valid Python before writing changes.
ast.parse(source + addition, filename=str(server))
backup = server.with_name('server.py.before_model_zoo.bak')
if backup.exists():
    raise SystemExit(f'Existing backup {backup} found; no modifications made')
shutil.copy2(server, backup)
server.write_text(source + addition, encoding='utf-8')
print('Added Model Zoo router to studio/server.py')
print('Backup:', backup)
print('Restart controller after installing Python dependencies.')
