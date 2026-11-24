"""Record a short animation of the 3D viewer for the README (WebP, or GIF).

Builds the site as pages.yml does (web/ plus orbits.czml), serves it locally,
opens it in headless Chromium (orbit_explorer.site, as the viewer tests do),
steps the clock with window.explorer.setTime and saves the frames as a looping
animated WebP.

By default it records the day the README figures are pinned to, 2026-11-17:
the CZML is built from the GP data of that day in data/inputs/precise/, not
from the current data that the data refresh replaces, so re-running it gives
the same animation. ``--current-data`` records the current day instead.

    pip install -e ".[viewer]" && playwright install chromium
    python tools/record_viewer.py                    # docs/images/viewer.webp
    python tools/record_viewer.py -o out.gif         # a GIF (needs ffmpeg)
"""

import argparse
import io
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

from orbit_explorer.site import ROOT, build_site, launch_chromium, serve

PRECISE_DIR = ROOT / "data/inputs/precise"
VIEWPORT = {"width": 1280, "height": 720}
SETTLE_TIMEOUT_MS = 5000

# Resolve once the new time is drawn and the globe's tiles are in, or after
# SETTLE_TIMEOUT_MS anyway (with false), so a tile that never loads can't hang
# the recording.
RENDER_SETTLED = """
timeout => new Promise((resolve) => {
  const scene = explorer.viewer.scene;
  let frames = 0;
  const finish = (settled) => { stop(); clearTimeout(timer); resolve(settled); };
  const stop = scene.postRender.addEventListener(() => {
    frames += 1;
    if (frames >= 2 && scene.globe.tilesLoaded) finish(true);
  });
  const timer = setTimeout(() => finish(false), timeout);
  scene.requestRender();
})
"""


def pinned_day() -> tuple[list[dict], datetime]:
    """CZML and window start for the README's day, 2026-11-17.

    czml.py reads data/inputs/ relative to the working directory, so it runs
    in a scratch folder holding that day's GP data instead of the current one.
    """
    manifest = json.loads((PRECISE_DIR / "manifest.json").read_text())
    gp = PRECISE_DIR / manifest["gp_file"]
    gp_manifest = PRECISE_DIR / manifest["gp_manifest"]
    with tempfile.TemporaryDirectory() as tmp:
        inputs = Path(tmp) / "data/inputs"
        inputs.mkdir(parents=True)
        shutil.copy(gp, inputs / "orbital_elements.json")
        shutil.copy(gp_manifest, inputs / "data_manifest.json")
        subprocess.run(
            [sys.executable, "-m", "orbit_explorer.czml"],
            cwd=tmp,
            check=True,
            capture_output=True,
        )
        czml = json.loads((Path(tmp) / "outputs/orbits.czml").read_text())
    start = json.loads(gp_manifest.read_text())["reference_start_utc"]
    return czml, datetime.fromisoformat(start)


def current_day() -> tuple[list[dict], datetime]:
    from orbit_explorer.czml import build_czml
    from orbit_explorer.kepler import START

    return build_czml(), START


def record(
    frames: int,
    hours: float,
    inertial: bool = False,
    zoom: float = 0.0,
    current_data: bool = False,
    viewport: dict | None = None,
) -> list[Image.Image]:
    """Screenshots of the viewer at ``frames`` times over ``hours`` from the
    day's start. Warns when frames were taken before the globe had settled."""
    czml, start = current_day() if current_data else pinned_day()
    unsettled = 0
    with tempfile.TemporaryDirectory() as tmp:
        site = Path(tmp)
        build_site(site, czml)
        with serve(site) as url, sync_playwright() as p:
            browser = launch_chromium(p)
            page = browser.new_page(
                viewport=viewport or VIEWPORT, device_scale_factor=1
            )
            page.goto(url)
            page.wait_for_function(
                "document.getElementById('loading').hidden", timeout=60_000
            )
            # a plain black sky, only for the recording: stars are noise that
            # animation formats compress badly, and say nothing about orbits
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
                t = start + timedelta(hours=hours * k / frames)
                page.evaluate("iso => explorer.setTime(iso)", f"{t:%Y-%m-%dT%H:%M:%S}Z")
                if not page.evaluate(RENDER_SETTLED, SETTLE_TIMEOUT_MS):
                    unsettled += 1
                images.append(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))
            browser.close()
    if unsettled:
        print(
            f"Warning: {unsettled} of {frames} frames were taken after "
            f"{SETTLE_TIMEOUT_MS / 1000:.0f} s without the globe fully loaded",
            file=sys.stderr,
        )
    return images


def scaled(images: list[Image.Image], width: int) -> list[Image.Image]:
    height = round(images[0].height * width / images[0].width)
    return [im.resize((width, height), Image.LANCZOS) for im in images]


def save_webp(
    images: list[Image.Image], path: Path, width: int, quality: int, fps: float
):
    """Animated WebP: full colour, far smaller than a GIF; GitHub shows it."""
    small = scaled(images, width)
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


def save_gif(images: list[Image.Image], path: Path, width: int, fps: float):
    """GIF through ffmpeg's palettegen/paletteuse (Pillow's single shared
    palette washed the colours out). About twice the size of the WebP."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("A GIF needs ffmpeg; use a .webp output instead")
    with tempfile.TemporaryDirectory() as tmp:
        for k, im in enumerate(images):
            im.save(Path(tmp) / f"f{k:03d}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(Path(tmp) / "f%03d.png"),
                "-vf",
                (
                    f"scale={width}:-1:flags=lanczos,split[a][b];"
                    "[a]palettegen=max_colors=96:stats_mode=diff[p];"
                    "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
                ),
                "-loop",
                "0",
                str(path),
            ],
            check=True,
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
    parser.add_argument("--width", type=int, default=720, help="output width in pixels")
    parser.add_argument("--quality", type=int, default=55, help="WebP quality")
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument(
        "--inertial", action="store_true", help="inertial frame: the Earth turns"
    )
    parser.add_argument(
        "--current-data",
        action="store_true",
        help="record the current GP data's day instead of 2026-11-17",
    )
    args = parser.parse_args(argv)
    images = record(
        args.frames, args.hours, args.inertial, args.zoom, args.current_data
    )
    if args.output.suffix == ".gif":
        save_gif(images, args.output, args.width, args.fps)
    else:
        save_webp(images, args.output, args.width, args.quality, args.fps)
    kb = args.output.stat().st_size / 1024
    print(
        f"Written {args.output}: {len(images)} frames, {args.width} px wide, {kb:.0f} KB"
    )


if __name__ == "__main__":
    main()
