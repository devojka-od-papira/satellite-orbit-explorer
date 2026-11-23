"""The 3D viewer site: build it as pages.yml does, serve it, open a browser.

Shared by the viewer tests (tests/viewer) and the animation recorder
(tools/record_viewer.py), so the two can't drift apart.
"""

import contextlib
import functools
import json
import shutil
import threading
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"
# Cesium needs WebGL; SwiftShader gives headless Chromium a software GPU.
CHROMIUM_ARGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def build_site(folder: Path, czml: list[dict] | None = None) -> None:
    """_site as pages.yml assembles it: web/ plus orbits.czml (built from the
    current data unless given)."""
    if czml is None:
        from orbit_explorer.czml import build_czml

        czml = build_czml()
    shutil.copytree(WEB_DIR, folder, dirs_exist_ok=True)
    (folder / "orbits.czml").write_text(json.dumps(czml, separators=(",", ":")))


@contextlib.contextmanager
def serve(folder: Path) -> Iterator[str]:
    """Serve a folder on a free local port; yields its URL."""
    handler = functools.partial(QuietHandler, directory=str(folder))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()


def launch_chromium(playwright):
    """Headless Chromium with a software GPU; an installed Google Chrome if
    Playwright's own Chromium isn't there. Raises Playwright's error if
    neither starts."""
    from playwright.sync_api import Error

    try:
        return playwright.chromium.launch(args=CHROMIUM_ARGS)
    except Error:
        return playwright.chromium.launch(channel="chrome", args=CHROMIUM_ARGS)
