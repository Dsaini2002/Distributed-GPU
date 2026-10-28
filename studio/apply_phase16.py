import re
from pathlib import Path

server = Path("studio/server.py")
content = server.read_text(encoding="utf-8-sig")

if "install_session_auth(app)" not in content:
    content += """

from .session_auth import install as install_session_auth
install_session_auth(app)
"""
    server.write_text(content, encoding="utf-8")

# Existing JS can still reference a hidden token input.
# Session middleware supplies authentication internally.
for filename in [
    "train_ui.py",
    "upload_ui.py",
    "monitor_ui.py",
    "history_ui.py"
]:
    path = Path("studio") / filename
    if not path.exists():
        continue

    content = path.read_text(encoding="utf-8-sig")

    content = re.sub(
        r'<input\s+id="token"\s+type="password"[^>]*>',
        '<input id="token" type="hidden" value="">',
        content
    )

    content = content.replace(
        "Cluster Authentication Token",
        "Authenticated with your login session"
    ).replace(
        "Cluster authentication token",
        "Authenticated with your login session"
    ).replace(
        "Cluster token",
        "Authenticated with your login session"
    ).replace(
        "Enter token and load history.",
        "Click Load History."
    )

    path.write_text(content, encoding="utf-8")

# Add explicit offline-worker reporting to monitor API.
path = Path("studio/monitor_api.py")
content = path.read_text(encoding="utf-8-sig")

if "offline_workers" not in content:
    old = 'status = get_status()'
    new = '''status = get_status()

    from .server import snapshot
    online_names = {
        n["hostname"] for n in snapshot()
        if n["online"]
    }

    offline_workers = []

    if status.get("status") == "running":
        offline_workers = sorted(set(
            stage["computer"]
            for stage in status.get("stages", [])
            if stage["computer"] not in online_names
        ))'''

    if old not in content:
        raise RuntimeError("Monitor API patch location missing")

    content = content.replace(old, new, 1)

    marker = '"error": status.get("error")'

    if marker not in content:
        raise RuntimeError("Monitor response patch location missing")

    content = content.replace(
        marker,
        marker + ',\n        "offline_workers": offline_workers',
        1
    )

    path.write_text(content, encoding="utf-8")

print("PHASE 16A INTEGRATION COMPLETE")
