"""The 3D viewer in a real browser: loading, the panel, stations and passes."""

import json
import re

import numpy as np
import pytest

pytest.importorskip("playwright.sync_api", reason='needs pip install -e ".[viewer]"')

from orbit_explorer.czml import build_czml
from orbit_explorer.elements import load_elements
from orbit_explorer.kepler import START
from orbit_explorer.style import short_name
from orbit_explorer.visibility import BELGRADE, STATIONS, satellite_look_angles

pytestmark = pytest.mark.viewer

SATS = load_elements()


@pytest.fixture(scope="module")
def czml_passes():
    czml = build_czml()
    return {
        p["id"][len("station-") :]: json.loads(p["properties"]["passes"])
        for p in czml
        if p["id"].startswith("station-")
    }


def test_loads_without_errors(open_viewer):
    v = open_viewer()
    names = v.js("[...document.querySelectorAll('.sat-name')].map(e => e.textContent)")
    assert names == [short_name(s) for s in SATS]
    buttons = v.js(
        "[...document.querySelectorAll('#station-picker button')]"
        ".map(b => b.textContent)"
    )
    assert buttons == [s.name for s in STATIONS]
    assert v.js("document.getElementById('window-date').textContent") == (
        f", {START.day} {START:%b %Y}"
    )
    assert v.errors == []


def test_pass_list_matches_the_czml(open_viewer, czml_passes):
    v = open_viewer()
    for station in STATIONS:
        v.page.click(f"#station-picker button:text-is('{station.name}')")
        rows = v.js(
            "[...document.querySelectorAll('.pass .pass-name')].map(e => e.textContent)"
        )
        assert rows == [p["name"] for p in czml_passes[station.slug]]
        assert v.js("document.getElementById('station-title').textContent") == (
            f"Passes over {station.name}"
        )
    assert v.errors == []


@pytest.mark.parametrize("station", STATIONS, ids=lambda s: s.slug)
def test_live_look_angles_match_python(open_viewer, station):
    """The panel's elevation and azimuth agree with visibility.py to 1°."""
    v = open_viewer(f"#{station.slug}")
    t_s = 12 * 3600 + 90  # 12:01:30 UTC
    v.set_time(f"{START:%Y-%m-%d}T12:01:30Z")
    lines = v.js("satellites.map(s => s.sky.textContent)")
    for sat, line in zip(SATS, lines):
        az, el, _ = satellite_look_angles(sat, np.array([float(t_s)]), station)
        shown_el = float(re.search(r": (−?\d+)°", line).group(1).replace("−", "-"))
        assert shown_el == pytest.approx(el[0], abs=1), line
        if "azimuth" in line:  # only shown while the satellite is up
            shown_az = float(re.search(r"azimuth (\d+)°", line).group(1))
            assert shown_az == pytest.approx(az[0], abs=1), line


def test_hash_opens_and_follows_a_station(open_viewer):
    v = open_viewer("#tokyo")
    assert v.js("station.slug") == "tokyo"
    qzs = v.js("[...document.querySelectorAll('.pass-name')].map(e => e.textContent)")
    assert "QZS-2" in qzs  # the reason Tokyo is offered
    v.js("location.hash = '#quito'")
    v.page.wait_for_function("station.slug === 'quito'")
    v.page.click("#station-picker button:text-is('Belgrade')")
    assert v.js("location.hash") == ""  # the default station has a clean URL


def test_only_the_chosen_station_is_drawn(open_viewer):
    v = open_viewer("#sydney")
    shown = v.js(
        "Object.fromEntries(stations.map(s => [s.slug,"
        " [s.entity.show, ...s.rings.map(e => e.show), ...s.links.map(e => e.show)]]))"
    )
    for slug, flags in shown.items():
        assert all(flags) if slug == "sydney" else not any(flags), slug
    v.page.click("#model-kepler")  # links end at the hidden SGP4 marker
    assert not any(v.js("station.links.map(e => e.show)"))
    assert all(v.js("station.rings.map(e => e.show)"))
    v.page.click("label.check:has(#show-station)")
    assert not v.js("station.entity.show")


def test_short_pass_plays_slower_then_speeds_up(open_viewer, czml_passes):
    v = open_viewer()
    default = v.js("viewer.clock.multiplier")
    leo = next(
        i for i, p in enumerate(czml_passes[BELGRADE.slug]) if p["name"] == "PULSAR-0"
    )
    v.page.locator(".pass").nth(leo).click()
    assert v.js("viewer.clock.multiplier") == 30
    v.set_time(czml_passes[BELGRADE.slug][leo]["set"].replace("Z", "") + ".5Z")
    v.js(
        "updatePassRows(Cesium.JulianDate.addSeconds(viewer.clock.currentTime, 5,"
        " new Cesium.JulianDate()))"
    )
    assert v.js("viewer.clock.multiplier") == default


def test_phone_layout(open_viewer):
    v = open_viewer(width=390, height=844)
    box = v.page.locator("#panel").bounding_box()
    assert box["width"] <= 390 - 16 + 1  # 8 px margins on both sides
    assert box["y"] > 844 / 3  # a sheet at the bottom, not a sidebar
    assert v.errors == []
