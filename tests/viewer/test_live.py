"""The viewer's live mode: SGP4 in the browser (satellite.js), checked against
the Python sgp4 library and visibility.py, which the rest of the project uses."""

import os
import re
import time
from datetime import datetime, timedelta

import numpy as np
import pytest

try:
    import playwright.sync_api  # noqa: F401
except ImportError:
    if os.environ.get("VIEWER_TESTS_REQUIRED"):
        raise  # the CI viewer job must run these, never skip them
    pytest.skip('needs pip install -e ".[viewer]"', allow_module_level=True)

from sgp4.api import jday
from test_viewer import short_complete_pass

from orbit_explorer.elements import load_elements
from orbit_explorer.frames import eci_to_ecef
from orbit_explorer.kepler import START
from orbit_explorer.sgp4_compare import propagate_sgp4, satrec
from orbit_explorer.style import short_name
from orbit_explorer.validation import NO_EOP, gmst_ut1
from orbit_explorer.visibility import STATIONS, find_passes, satellite_look_angles

pytestmark = pytest.mark.viewer

SATS = load_elements()
# From the start of the GP data's day to three days on: around the epochs and
# a little beyond, where the live mode is used. One time is between two
# milliseconds (0.4 ms is about 3 m along a LEO orbit): a JS Date would lose it.
OFFSETS_S = [0, 3 * 3600 + 17.5004, 12 * 3600, 3 * 86400 + 600]


def iso(seconds_after_start: float) -> str:
    """ISO 8601 UTC to the microsecond (Cesium keeps all of it)."""
    t = START + timedelta(seconds=seconds_after_start)
    return f"{t:%Y-%m-%dT%H:%M:%S.%f}Z"


def parse_deg(text: str) -> float:
    return float(text.replace("−", "-"))


@pytest.fixture
def live(open_viewer):
    v = open_viewer("?now")
    assert v.wait_for_live_mode() == "ready"
    return v


def test_now_link_opens_the_live_mode(live):
    v = live
    assert v.js("explorer.mode") == "now"
    assert v.js("document.getElementById('mode-now').getAttribute('aria-pressed')") == (
        "true"
    )
    assert not v.js("document.getElementById('subtitle-now').hidden")
    assert v.js("document.getElementById('subtitle-day').hidden")
    # the clock runs at real time, from now
    clock_ms = v.js(
        "Cesium.JulianDate.toDate(explorer.viewer.clock.currentTime).getTime()"
    )
    assert abs(clock_ms / 1000 - time.time()) < 60
    assert v.js("explorer.viewer.clock.multiplier") == 1
    # live markers instead of the CZML ones; no day pass list
    shown = v.js(
        "explorer.satellites.map(s => [s.liveEntity.show, s.entity.show, s.sgp4.show])"
    )
    assert shown == [[True, False, False]] * len(SATS)
    assert v.js("document.getElementById('pass-list').hidden")
    assert v.js("document.getElementById('station-title').textContent") == (
        "Over Belgrade now"
    )
    assert v.errors == []


def test_switching_back_restores_the_day(live):
    v = live
    v.page.click("#mode-day")
    assert v.js("explorer.mode") == "day"
    assert v.js("location.search") == ""
    assert v.js("document.getElementById('station-title').textContent") == (
        "Passes over Belgrade"
    )
    assert not v.js("document.getElementById('pass-list').hidden")
    start = v.js("Cesium.JulianDate.toIso8601(explorer.viewer.clock.startTime, 0)")
    assert start == f"{START:%Y-%m-%dT%H:%M:%S}Z"  # the CZML day again
    assert v.js("explorer.satellites.every(s => !s.liveEntity.show)")
    v.pick_station("Tokyo")
    v.page.click("#mode-now")
    assert v.js("location.search + location.hash") == "?now#tokyo"
    assert v.errors == []


def test_back_from_now_the_day_plays_as_it_did(open_viewer, czml_passes):
    """The day's clock comes back as it was: paused if it was paused, and at
    the day's speed even if Now was chosen during a slowed-down short pass."""
    v = open_viewer()
    assert v.wait_for_live_mode() == "ready"
    default = v.js("explorer.viewer.clock.multiplier")
    found = short_complete_pass(czml_passes)
    if found is not None:
        station, i, _ = found
        v.pick_station(station.name)
        v.page.locator(".pass").nth(i).click()
        assert v.js("explorer.viewer.clock.multiplier") == 30
        v.page.click("#mode-now")
        v.page.click("#mode-day")
        assert v.js("explorer.viewer.clock.multiplier") == default
        assert v.js("explorer.viewer.clock.shouldAnimate")
    v.set_time(iso(6 * 3600))  # pauses the clock
    v.page.click("#mode-now")
    assert v.js("explorer.viewer.clock.shouldAnimate")  # live: it runs
    v.page.click("#mode-day")
    assert not v.js("explorer.viewer.clock.shouldAnimate")
    assert v.js(
        "Cesium.JulianDate.toIso8601(explorer.viewer.clock.currentTime, 0)"
    ) == (f"{START + timedelta(hours=6):%Y-%m-%dT%H:%M:%S}Z")
    assert v.errors == []


def test_sgp4_in_the_browser_matches_python(live):
    """satellite.js and Python's sgp4 implement the same Vallado code: the
    same GP elements give the same TEME position, here to under a metre."""
    worst = 0.0
    for sat in SATS:
        rec = satrec(sat)
        for t_s in OFFSETS_S:
            js = np.array(
                v_teme := live.js(
                    "([n, t]) => explorer.liveTemeKm(n, t)",
                    [sat["NORAD_CAT_ID"], iso(t_s)],
                )
            )
            assert v_teme is not None
            when = START + timedelta(seconds=t_s)
            jd, fr = jday(
                when.year,
                when.month,
                when.day,
                when.hour,
                when.minute,
                when.second + when.microsecond / 1e6,
            )
            err, r, _ = rec.sgp4(jd, fr)
            assert err == 0
            gap_m = np.linalg.norm(js - np.array(r)) * 1000
            worst = max(worst, gap_m)
            assert gap_m < 1.0, f"{short_name(sat)} at {iso(t_s)}: {gap_m:.3f} m"
    print(f"largest browser-Python gap: {worst * 1000:.2f} mm")


def test_live_marker_is_in_the_earth_fixed_frame(live):
    """The marker is TEME turned by GMST (IAU 1982, UT1 = UTC), what Cesium
    and validation.gmst_ut1 compute. Agreement to 1 m checks the rotation."""
    t_s = np.array([12 * 3600.0])
    theta = gmst_ut1(t_s, START, NO_EOP)
    for sat in SATS:
        pos, _ = propagate_sgp4(sat, t_s, START)
        expected = eci_to_ecef(pos, theta)[0] * 1000
        js = np.array(
            live.js(
                "([n, t]) => explorer.liveFixedM(n, t)",
                [sat["NORAD_CAT_ID"], iso(12 * 3600)],
            )
        )
        gap = np.linalg.norm(js - expected)
        assert gap < 1.0, f"{short_name(sat)}: {gap:.2f} m"


def test_live_look_angles_match_python(live):
    """As in the day view, the panel's elevation and azimuth agree with
    visibility.py to 1°, for every satellite and station."""
    t_s = np.array([12 * 3600 + 90.0])
    for station in STATIONS:
        live.pick_station(station.name)
        live.set_time(iso(t_s[0]))
        lines = live.js("explorer.satellites.map(s => s.sky.textContent)")
        assert len(lines) == len(SATS), lines
        for sat, line in zip(SATS, lines, strict=True):
            az, el, _ = satellite_look_angles(sat, t_s, station)
            where = f"{station.name}, {short_name(sat)}: {line!r}"
            assert line.startswith(f"{station.name}: "), where
            shown_el = parse_deg(re.search(r": (−?\d+)°", line)[1])
            assert shown_el == pytest.approx(el[0], abs=1), where
            if "azimuth" in line:
                shown_az = parse_deg(re.search(r"azimuth (\d+)°", line)[1])
                assert abs((shown_az - az[0] + 180) % 360 - 180) <= 1, where
            assert re.search(
                r"· (rises in|sets in|up for the next 24 h|doesn't rise)", line
            ), where


def test_next_pass_matches_visibility_py(live):
    """From 00:00 the next 24 hours are the day visibility.py covers: the next
    rise (or set, if already up) agrees to a minute, the maximum to 1°."""
    t = np.arange(0, 86400 + 1, 10.0)
    compared = 0
    for station in STATIONS:
        live.pick_station(station.name)
        live.set_time(iso(0))
        found = {p["norad"]: p for p in live.js("explorer.nextPasses()")}
        for sat in SATS:
            az, el, _ = satellite_look_angles(sat, t, station)
            passes = find_passes(t, el, az)  # the mask the viewer uses too
            ours = found[sat["NORAD_CAT_ID"]]
            where = f"{station.name}, {short_name(sat)}: {ours}"
            if not passes:
                assert ours["rise"] is None and not ours["upNow"], where
                continue
            first = passes[0]
            assert ours["upNow"] == (not first.rises_in_window), where
            if first.rises_in_window:
                rise = START + timedelta(seconds=first.rise_s)
                assert abs(_seconds(ours["rise"]) - rise.timestamp()) < 60, where
                compared += 1
            if first.sets_in_window:
                set_ = START + timedelta(seconds=first.set_s)
                assert abs(_seconds(ours["set"]) - set_.timestamp()) < 60, where
            if first.rises_in_window and first.sets_in_window:
                assert ours["maxEl"] == pytest.approx(first.max_el_deg, abs=1), where
    assert compared > 0  # some satellite rose somewhere


def _seconds(iso_text: str) -> float:
    return datetime.fromisoformat(iso_text).timestamp()


EPOCHS = [datetime.fromisoformat(s["EPOCH"] + "+00:00") for s in SATS]


@pytest.mark.parametrize(
    "at",
    [min(EPOCHS) - timedelta(hours=6), max(EPOCHS) + timedelta(hours=12)],
    ids=["before-the-epochs", "after"],
)
def test_panel_shows_element_age(live, at):
    """Each line ends with how old the elements are at the clock's time, or
    how much later they are, when the clock is before their epoch."""
    live.set_time(iso((at - START).total_seconds()))
    lines = live.js("explorer.satellites.map(s => s.live.textContent)")
    for epoch, line in zip(EPOCHS, lines, strict=True):
        age_h = (at - epoch).total_seconds() / 3600
        m = re.search(r"elements (?:from )?(?:(\d+) h|([\d.]+) d) (old|later)$", line)
        assert m, line
        shown_h = float(m[1]) if m[1] else float(m[2]) * 24
        assert m[3] == ("old" if age_h >= 0 else "later"), line
        assert shown_h == pytest.approx(abs(age_h), abs=0.5 if m[1] else 2.4), line


@pytest.mark.parametrize(
    "blocked", ["**/satellite.js@*/**", "**/gp.json"], ids=["library", "elements"]
)
def test_without_the_library_or_elements_the_day_view_still_works(open_viewer, blocked):
    v = open_viewer("?now", setup=lambda page: page.route(blocked, lambda r: r.abort()))
    assert v.wait_for_live_mode() == "unavailable"
    assert v.js("explorer.mode") == "day"
    assert v.js("document.getElementById('mode-now').disabled")
    assert "unavailable" in v.js("document.getElementById('mode-hint').textContent")
    assert v.js("document.querySelectorAll('.pass').length") > 0
    # the blocked request and the one expected message, nothing else
    assert v.errors, "the failure should be logged"
    assert all(
        "Live mode unavailable" in e or "Failed to load resource" in e for e in v.errors
    ), v.errors
