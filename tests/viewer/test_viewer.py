"""The 3D viewer in a real browser: loading, the panel, stations and passes."""

import json
import os
import re
from datetime import datetime

import numpy as np
import pytest

try:
    import playwright.sync_api  # noqa: F401
except ImportError:
    if os.environ.get("VIEWER_TESTS_REQUIRED"):
        raise  # the CI viewer job must run these, never skip them
    pytest.skip('needs pip install -e ".[viewer]"', allow_module_level=True)

from orbit_explorer.elements import load_elements
from orbit_explorer.kepler import START
from orbit_explorer.style import short_name
from orbit_explorer.visibility import STATIONS, satellite_look_angles

pytestmark = pytest.mark.viewer

SATS = load_elements()
SHORT_PASS_S = 15 * 60  # main.js plays passes shorter than this at 30x


@pytest.fixture(scope="module")
def czml_passes(site_czml):
    """Passes per station slug, from the same CZML the page loads."""
    return {
        p["id"][len("station-") :]: json.loads(p["properties"]["passes"])
        for p in site_czml
        if p["id"].startswith("station-")
    }


def parse_deg(text: str) -> float:
    return float(text.replace("−", "-"))


def test_loads_without_errors(open_viewer):
    v = open_viewer()
    names = v.js("[...document.querySelectorAll('.sat-name')].map(e => e.textContent)")
    assert names == [short_name(s) for s in SATS]
    buttons = v.js(
        "[...document.querySelectorAll('#station-picker button')]"
        ".map(b => b.textContent)"
    )
    assert buttons == [s.name for s in STATIONS]
    # The browser's short month isn't always Python's %b ("Sept" vs "Sep"),
    # so compare day, month prefix and year.
    date = v.js("document.getElementById('window-date').textContent")
    m = re.fullmatch(r", (\d+) (\w+) (\d{4})", date)
    assert m, date
    assert int(m[1]) == START.day and int(m[3]) == START.year
    assert m[2][:3] == f"{START:%b}", date
    assert v.errors == []


def test_pass_list_matches_the_czml(open_viewer, czml_passes):
    v = open_viewer()
    for station in STATIONS:
        v.pick_station(station.name)
        rows = v.js(
            "[...document.querySelectorAll('.pass .pass-name')].map(e => e.textContent)"
        )
        assert rows == [p["name"] for p in czml_passes[station.slug]]
        assert v.js("document.getElementById('station-title').textContent") == (
            f"Passes over {station.name}"
        )
    assert v.errors == []


def test_live_look_angles_match_python(open_viewer):
    """The panel's elevation and azimuth agree with visibility.py to 1°,
    for every satellite and every station (one page, switching stations)."""
    v = open_viewer()
    t_s = np.array([12 * 3600 + 90.0])  # 12:01:30 UTC
    for station in STATIONS:
        v.pick_station(station.name)
        v.set_time(f"{START:%Y-%m-%d}T12:01:30Z")
        lines = v.js("explorer.satellites.map(s => s.sky.textContent)")
        assert len(lines) == len(SATS), lines  # nothing silently missing
        for sat, line in zip(SATS, lines, strict=True):
            az, el, _ = satellite_look_angles(sat, t_s, station)
            where = f"{station.name}, {short_name(sat)}: {line!r}"
            assert line.startswith(f"{station.name}: "), where
            shown_el = parse_deg(re.search(r": (−?\d+)°", line)[1])
            assert shown_el == pytest.approx(el[0], abs=1), where
            if "azimuth" in line:  # only shown while the satellite is up
                shown_az = parse_deg(re.search(r"azimuth (\d+)°", line)[1])
                gap = (shown_az - az[0] + 180) % 360 - 180  # 360° and 0° agree
                assert abs(gap) <= 1, where


def test_hash_opens_and_follows_a_station(open_viewer):
    v = open_viewer("#tokyo")
    assert v.js("explorer.station.slug") == "tokyo"
    names = v.js("[...document.querySelectorAll('.pass-name')].map(e => e.textContent)")
    assert "QZS-2" in names  # the reason Tokyo is offered
    v.js("location.hash = '#quito'")
    v.page.wait_for_function("explorer.station.slug === 'quito'")
    v.pick_station("Belgrade")
    assert v.js("location.hash") == ""  # the default station has a clean URL


def test_only_the_chosen_station_is_drawn(open_viewer):
    v = open_viewer("#sydney")
    shown = v.js(
        "Object.fromEntries(explorer.stations.map(s => [s.slug,"
        " [s.entity.show, ...s.rings.map(e => e.show), ...s.links.map(e => e.show)]]))"
    )
    assert set(shown) == {s.slug for s in STATIONS}
    for slug, flags in shown.items():
        assert all(flags) if slug == "sydney" else not any(flags), slug
    v.page.click("#model-kepler")  # links end at the hidden SGP4 marker
    assert not any(v.js("explorer.station.links.map(e => e.show)"))
    assert all(v.js("explorer.station.rings.map(e => e.show)"))
    v.page.click("label.check:has(#show-station)")
    assert not v.js("explorer.station.entity.show")


def short_complete_pass(czml_passes):
    """(station, index, pass) of the first pass that rises and sets inside the
    day and is short enough to be slowed down. Any day's data has one (LEO
    passes are minutes long), but the station and satellite vary."""
    for station in STATIONS:
        for i, p in enumerate(czml_passes[station.slug]):
            rise, set_ = (datetime.fromisoformat(p[k]) for k in ("rise", "set"))
            if (
                p["risesInWindow"]
                and p["setsInWindow"]
                and (set_ - rise).total_seconds() < SHORT_PASS_S
            ):
                return station, i, p
    return None


def test_short_pass_plays_slower_then_speeds_up(open_viewer, czml_passes):
    found = short_complete_pass(czml_passes)
    if found is None:
        pytest.skip("no short pass in today's data")
    station, i, p = found
    v = open_viewer()
    v.pick_station(station.name)
    default = v.js("explorer.viewer.clock.multiplier")
    v.page.locator(".pass").nth(i).click()
    assert v.js("explorer.viewer.clock.multiplier") == 30
    v.set_time(p["set"])  # the last second of the pass
    assert v.js("explorer.viewer.clock.multiplier") == 30  # still inside
    end = datetime.fromisoformat(p["set"]).timestamp() + 2
    v.set_time(f"{datetime.fromtimestamp(end, START.tzinfo):%Y-%m-%dT%H:%M:%S}Z")
    assert v.js("explorer.viewer.clock.multiplier") == default  # pass over


def test_phone_layout(open_viewer):
    v = open_viewer(width=390, height=844)
    box = v.page.locator("#panel").bounding_box()
    assert box["width"] <= 390 - 16 + 1  # 8 px margins on both sides
    assert box["y"] > 844 / 3  # a sheet at the bottom, not a sidebar
    assert v.errors == []
