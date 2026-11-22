"""Record a short animation of the 3D viewer for the README (WebP or GIF).

Builds the site as pages.yml does (web/ plus orbits.czml), serves it locally,
opens it in headless Chromium (Playwright, as the viewer tests do), steps the
clock through the day with window.explorer.setTime and saves the frames as a
looping animated WebP (or GIF).

    pip install -e ".[viewer]" && playwright install chromium
    python tools/record_viewer.py                      # docs/images/viewer.webp
    python tools/record_viewer.py -o out.gif           # a GIF instead
"""

import argparse
import functools
import io
import json
import shutil
import tempfile
import threading
from datetime import timedelta
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from orbit_explorer.czml import build_czml
from orbit_explorer.kepler import START

ROOT = Path(__file__).resolve().parents[1]
# Cesium needs WebGL; SwiftShader gives headless Chromium a software GPU.
CHROMIUM_ARGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
VIEWPORT = {"width": 1280, "height": 720}

# Wait until the frame for the new time is drawn and the globe's tiles are in.
RENDER_SETTLED = """
() => new Promise((resolve) => {
  const scene = explorer.viewer.scene;
  let frames = 0;
  const done = scene.postRender.addEventListener(() => {
    frames += 1;
    if (frames >= 2 && scene.globe.tilesLoaded) { done(); resolve(); }
  });
  scene.requestRender();
})
"""


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def build_site(folder: Path) -> None:
    shutil.copytree(ROOT / "web", folder, dirs_exist_ok=True)
    (folder / "orbits.czml").write_text(json.dumps(build_czml(), separators=(",", ":")))


def record(
    frames: int, hours: float, inertial: bool, zoom: float = 0.0
) -> list[Image.Image]:
    with tempfile.TemporaryDirectory() as tmp:
        site = Path(tmp)
        build_site(site)
        handler = functools.partial(QuietHandler, directory=str(site))
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with sync_playwright() as p:
                try:
                    browser = p.chromium.launch(args=CHROMIUM_ARGS)
                except PlaywrightError:  # no bundled Chromium: an installed Chrome
                    browser = p.chromium.launch(channel="chrome", args=CHROMIUM_ARGS)
                page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
                page.goto(f"http://127.0.0.1:{server.server_port}/")
                page.wait_for_function(
                    "document.getElementById('loading').hidden", timeout=60_000
                )
                # a plain black sky: stars are noise that animation formats
                # compress badly, and they say nothing about the orbits
                page.evaluate(
                    "() => { const s = explorer.viewer.scene;"
                    " s.skyBox.show = false; s.sun.show = false; s.moon.show = false; }"
                )
                if zoom:  # closer than the default whole-system view
                    page.evaluate(
                        "f => { const c = explorer.viewer.camera;"
                        " c.zoomIn(c.positionCartographic.height * f); }",
                        zoom,
                    )
                if inertial:
                    page.click("#frame-inertial")
                    page.wait_for_timeout(2500)  # the camera's fly-to
                images = []
                for k in range(frames):
                    t = START + timedelta(hours=hours * k / frames)
                    page.evaluate(
                        "iso => explorer.setTime(iso)", f"{t:%Y-%m-%dT%H:%M:%S}Z"
                    )
                    page.evaluate(RENDER_SETTLED)
                    images.append(
                        Image.open(io.BytesIO(page.screenshot())).convert("RGB")
                    )
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
    return images


def save_gif(
    images: list[Image.Image], path: Path, width: int, colors: int, fps: float
):
    """Scale, share one palette across frames (no flicker) and loop forever."""
    height = round(images[0].height * width / images[0].width)
    small = [im.resize((width, height), Image.LANCZOS) for im in images]
    # one palette from a strip of sample frames, applied to all
    sample = Image.new("RGB", (width, height * min(4, len(small))))
    for i, im in enumerate(small[:: max(1, len(small) // 4)][:4]):
        sample.paste(im, (0, i * height))
    palette = sample.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    frames = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in small]
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
        disposal=1,
    )


def save_webp(
    images: list[Image.Image], path: Path, width: int, quality: int, fps: float
):
    """Animated WebP: full colour, far smaller than a GIF; GitHub shows it."""
    height = round(images[0].height * width / images[0].width)
    small = [im.resize((width, height), Image.LANCZOS) for im in images]
    path.parent.mkdir(parents=True, exist_ok=True)
    small[0].save(
        path,
        save_all=True,
        append_images=small[1:],
        duration=round(1000 / fps),
        loop=0,
        quality=quality,
        method=6,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-o", "--output", type=Path, default=ROOT / "docs/images/viewer.webp"
    )
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument(
        "--hours",
        type=float,
        default=4,
        help="span shown (4 h: PULSAR-0 moves smoothly)",
    )
    parser.add_argument(
        "--zoom", type=float, default=0.35, help="move the camera this share closer"
    )
    parser.add_argument("--width", type=int, default=720, help="GIF width in pixels")
    parser.add_argument("--colors", type=int, default=96, help="GIF palette size")
    parser.add_argument("--quality", type=int, default=55, help="WebP quality")
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument(
        "--inertial", action="store_true", help="inertial frame: the Earth turns"
    )
    args = parser.parse_args(argv)
    images = record(args.frames, args.hours, args.inertial, args.zoom)
    if args.output.suffix == ".webp":
        save_webp(images, args.output, args.width, args.quality, args.fps)
    else:
        save_gif(images, args.output, args.width, args.colors, args.fps)
    kb = args.output.stat().st_size / 1024
    print(
        f"Written {args.output}: {len(images)} frames, {args.width} px wide, {kb:.0f} KB"
    )


if __name__ == "__main__":
    main()
