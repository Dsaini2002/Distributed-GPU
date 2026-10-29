"""Native desktop launcher (Windows + Edge WebView2, via pywebview)."""
from __future__ import annotations
import sys
from pathlib import Path


def resource_path() -> Path:
    if hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS) / 'desktop_app' / 'assets' / 'index.html'
    return Path(__file__).resolve().parent / 'assets' / 'index.html'


def main() -> None:
    if sys.platform != 'win32':
        raise SystemExit('Distributed-GPU Studio desktop beta currently supports Windows only.')
    try:
        import webview
    except ImportError as exc:
        raise SystemExit('Desktop dependency missing: install pywebview. See README.md.') from exc

    from desktop_app.core import DesktopBridge
    page = resource_path()
    if not page.is_file():
        raise SystemExit('App assets missing: ' + str(page))
    bridge = DesktopBridge()
    webview.create_window(
        'Distributed-GPU Studio', url=page.as_uri(), js_api=bridge,
        width=1420, height=920, min_size=(880, 650),
        background_color='#08111f', text_select=True,
    )
    webview.start(debug=False)


if __name__ == '__main__':
    main()
