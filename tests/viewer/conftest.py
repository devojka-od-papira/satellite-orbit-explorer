"""Fixtures for the 3D viewer tests: build the site as Pages does, serve it
locally and open it in headless Chromium (Playwright).

These tests need ``pip install -e ".[viewer]"`` and a browser:
``playwright install chromium`` (or a local Google Chrome, used as a fallback).
Without them they are skipped, so a plain ``pytest`` stays fast, except when
VIEWER_TESTS_REQUIRED is set (the CI viewer job): then a missing Playwright or
browser fails the run instead of skipping it into a green check.
"""

import json
import os

import pytest

from orbit_explorer.site import build_site, launch_chromium, serve

try:
    from playwright import sync_api
except ImportError:  # the test module skips itself; see test_viewer.py
    sync_api = None

REQUIRED = bool(os.environ.get("VIEWER_TESTS_REQUIRED"))


def unavailable(reason: str):
    """Skip locally; fail where the viewer tests are required."""
    if REQUIRED:
        pytest.fail(f"{reason} (VIEWER_TESTS_REQUIRED is set)", pytrace=False)
    pytest.skip(reason)


@pytest.fixture(scope="session")
def site_dir(tmp_path_factory):
    """_site as pages.yml assembles it: web/ plus orbits.czml."""
    site = tmp_path_factory.mktemp("site")
    build_site(site)
    return site


@pytest.fixture(scope="session")
def site_czml(site_dir):
    """The CZML packets the page loads (built once per session)."""
    return json.loads((site_dir / "orbits.czml").read_text())


@pytest.fixture(scope="session")
def site_url(site_dir):
    with serve(site_dir) as url:
        yield url


@pytest.fixture(scope="session")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = launch_chromium(p)
        except sync_api.Error:
            unavailable("no browser: run `playwright install chromium`")
        yield b
        b.close()


class Viewer:
    """An open viewer page, with the console and page errors it produced.

    Talks to the page only through ``window.explorer``, the small test surface
    main.js exposes, not through its internal names.
    """

    def __init__(self, page):
        self.page = page
        self.errors = []
        page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        page.on("console", self._console)

    def _console(self, message):
        # The software GPU (SwiftShader) logs performance notes as warnings;
        # they come from the driver, not from the page.
        if (
            message.type in ("error", "warning")
            and "GL Driver Message" not in message.text
        ):
            self.errors.append(f"{message.type}: {message.text}")

    def open(self, url):
        self.page.goto(url)
        self.page.wait_for_function(
            "document.getElementById('loading').hidden", timeout=60_000
        )
        return self

    def js(self, expression, arg=None):
        return self.page.evaluate(expression, arg)

    def set_time(self, iso):
        """Pause the clock at a UTC time and let the panel catch up."""
        self.js("iso => explorer.setTime(iso)", iso)

    def pick_station(self, name):
        self.page.click(f"#station-picker button:text-is('{name}')")


@pytest.fixture
def open_viewer(browser, site_url):
    pages = []

    def _open(fragment="", width=1400, height=900):
        page = browser.new_page(viewport={"width": width, "height": height})
        pages.append(page)
        return Viewer(page).open(site_url + fragment)

    yield _open
    for page in pages:
        page.close()
