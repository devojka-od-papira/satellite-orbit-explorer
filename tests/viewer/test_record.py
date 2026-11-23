"""The animation recorder (tools/record_viewer.py) still drives the viewer.

It depends on window.explorer.setTime, the #loading element and the scene, all
of which a viewer change could break without anyone re-recording; two small
frames are enough to notice.
"""

import os
import sys
from pathlib import Path

import pytest

try:
    import playwright.sync_api  # noqa: F401
except ImportError:
    if os.environ.get("VIEWER_TESTS_REQUIRED"):
        raise
    pytest.skip('needs pip install -e ".[viewer]"', allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import record_viewer

pytestmark = pytest.mark.viewer


def test_records_frames_of_the_pinned_day(tmp_path, capsys):
    viewport = {"width": 480, "height": 270}
    images = record_viewer.record(frames=2, hours=0.5, viewport=viewport)
    assert [im.size for im in images] == [(480, 270)] * 2
    assert images[0].tobytes() != images[1].tobytes()  # the clock moved
    out = tmp_path / "v.webp"
    record_viewer.save_webp(images, out, width=240, quality=50, fps=10)
    assert out.stat().st_size > 0
    assert "Warning" not in capsys.readouterr().err  # every frame settled


def test_pinned_day_is_the_readme_day():
    czml, start = record_viewer.pinned_day()
    assert f"{start:%Y-%m-%d}" == "2026-11-17"
    assert czml[0]["clock"]["currentTime"].startswith("2026-11-17")
