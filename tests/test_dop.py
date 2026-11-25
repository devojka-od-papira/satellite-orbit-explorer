"""DOP: the formula on geometries with a known answer, what it means (Monte
Carlo), and the committed GPS/Galileo snapshot against orbital facts."""

import re

import numpy as np
import pytest

from orbit_explorer import dop, gnss
from orbit_explorer.kepler import START
from orbit_explorer.visibility import BELGRADE, STATIONS, satellite_look_angles

RNG = np.random.default_rng(2026)


def one_epoch(az_deg, el_deg, system=None) -> dop.DOP:
    """DOP of one epoch from satellite azimuths and elevations."""
    u = dop.unit_vectors(np.asarray(az_deg, float), np.asarray(el_deg, float))
    u = u[:, None, :]
    return dop.dop(u, np.ones(u.shape[:2], bool), system)


def random_sky(n: int) -> tuple[np.ndarray, np.ndarray]:
    """n satellites spread over the sky above 10° (uniform on the dome)."""
    az = RNG.uniform(0, 360, n)
    el = np.degrees(np.arcsin(RNG.uniform(np.sin(np.radians(10)), 1, n)))
    return az, el


# ---------- The formula ----------
@pytest.mark.parametrize("el_deg", [0.0, 15.0, 30.0, 45.0])
def test_zenith_and_three_around_has_the_closed_form(el_deg):
    """One satellite at the zenith, three 120° apart at elevation e. With
    s = sin e, G^T G separates into horizontal and vertical/clock blocks:
    HDOP^2 = 4 / (3 cos^2 e),  VDOP^2 = 4 / (3 (1 - s)^2),
    TDOP^2 = (1 + 3 s^2) / (3 (1 - s)^2)."""
    s, c = np.sin(np.radians(el_deg)), np.cos(np.radians(el_deg))
    d = one_epoch([0, 0, 120, 240], [90, el_deg, el_deg, el_deg])
    hdop2, vdop2 = 4 / (3 * c**2), 4 / (3 * (1 - s) ** 2)
    tdop2 = (1 + 3 * s**2) / (3 * (1 - s) ** 2)
    assert d.hdop[0] == pytest.approx(np.sqrt(hdop2), rel=1e-12)
    assert d.vdop[0] == pytest.approx(np.sqrt(vdop2), rel=1e-12)
    assert d.pdop[0] == pytest.approx(np.sqrt(hdop2 + vdop2), rel=1e-12)
    assert d.gdop[0] == pytest.approx(np.sqrt(hdop2 + vdop2 + tdop2), rel=1e-12)


def test_turning_the_sky_changes_nothing():
    """DOP depends on the geometry, not on where north is."""
    az, el = random_sky(9)
    a, b = one_epoch(az, el), one_epoch((az + 77) % 360, el)
    for name in ("gdop", "pdop", "hdop", "vdop"):
        assert getattr(a, name)[0] == pytest.approx(getattr(b, name)[0], rel=1e-10)


def test_another_satellite_never_makes_it_worse():
    for _ in range(200):
        az, el = random_sky(RNG.integers(4, 12))
        more_az, more_el = random_sky(1)
        before = one_epoch(az, el)
        after = one_epoch(np.append(az, more_az), np.append(el, more_el))
        assert after.pdop[0] <= before.pdop[0] * (1 + 1e-12)
        assert after.gdop[0] <= before.gdop[0] * (1 + 1e-12)


def test_too_few_satellites_or_a_cone_cannot_be_solved():
    assert np.isnan(one_epoch([0, 90, 180], [30, 40, 50]).pdop[0])
    # all at one elevation: the up component is the same for every
    # satellite, so height and clock can't be told apart
    assert np.isnan(one_epoch([0, 90, 180, 270], [35, 35, 35, 35]).pdop[0])
    assert np.isfinite(one_epoch([0, 90, 180, 270], [35, 35, 35, 60]).pdop[0])


def test_dop_is_the_position_error_per_unit_ranging_error():
    """What DOP means: solve the position by least squares from pseudoranges
    with unit Gaussian noise; the RMS 3D error is PDOP (horizontal HDOP)."""
    az, el = random_sky(8)
    u = dop.unit_vectors(az, el)
    g = np.column_stack([-u, np.ones(len(u))])
    noise = RNG.standard_normal((len(u), 40_000))
    x = np.linalg.lstsq(g, noise, rcond=None)[0]  # the error of each solution
    d = one_epoch(az, el)
    assert np.sqrt(np.mean(np.sum(x[:3] ** 2, axis=0))) == pytest.approx(
        d.pdop[0], rel=0.02
    )
    assert np.sqrt(np.mean(np.sum(x[:2] ** 2, axis=0))) == pytest.approx(
        d.hdop[0], rel=0.02
    )


# ---------- Two systems, two clocks ----------
def test_a_second_system_with_one_satellite_adds_nothing():
    """Its clock offset is a new unknown, which that one satellite uses up."""
    az, el = random_sky(6)
    alone = one_epoch(az, el)
    extra_az, extra_el = random_sky(1)
    both = one_epoch(np.append(az, extra_az), np.append(el, extra_el), [0] * 6 + [1])
    assert both.pdop[0] == pytest.approx(alone.pdop[0], rel=1e-9)
    assert both.gdop[0] > alone.gdop[0]  # GDOP counts the new clock


def test_an_unseen_system_drops_out():
    """A system with no satellite in view leaves the other's DOP as it is."""
    az, el = random_sky(10)
    u = dop.unit_vectors(az, el)[:, None, :]
    visible = np.ones((10, 1), bool)
    visible[6:] = False
    two = dop.dop(u, visible, [0] * 6 + [1] * 4)
    one = dop.dop(u[:6], visible[:6])
    for name in ("gdop", "pdop", "hdop", "vdop"):
        assert getattr(two, name)[0] == pytest.approx(getattr(one, name)[0], rel=1e-9)
    assert two.n[0] == 6


# ---------- The committed snapshot ----------
@pytest.fixture(scope="module")
def snapshot():
    positions, t, start = dop.load_positions()
    return positions, t, start


@pytest.fixture(scope="module")
def at_stations(snapshot):
    positions, _, _ = snapshot
    return {s.name: dop.station_dop(positions, s) for s in STATIONS}


def test_snapshot_is_both_whole_constellations():
    systems, _start = gnss.load_snapshot()
    assert set(systems) == {"GPS", "Galileo"}
    for sats in systems.values():
        assert len(sats) >= gnss.MIN_SATELLITES
    # no satellite in both lists (they'd be counted twice)
    ids = [s["NORAD_CAT_ID"] for sats in systems.values() for s in sats]
    assert len(ids) == len(set(ids))


def test_lines_of_sight_match_visibility(snapshot):
    """dop's look angles are visibility.py's: GALILEO 29 is in the snapshot,
    propagated here over the current GP data's day for the comparison."""
    systems, _ = gnss.load_snapshot()
    (sat,) = [s for s in systems["Galileo"] if s["NORAD_CAT_ID"] == 59598]
    t = np.arange(0, 86400, 600.0)
    pos = dop.ecef_positions([sat], t, START)
    _, el = dop.lines_of_sight(
        pos, BELGRADE.lat_deg, BELGRADE.lon_deg, BELGRADE.height_km
    )
    _, expected, _ = satellite_look_angles(sat, t, BELGRADE)
    np.testing.assert_allclose(el[0], expected, atol=1e-9)


def test_gps_sky_repeats_a_little_before_a_sidereal_day(snapshot):
    """GPS orbits twice per sidereal day, so each satellite comes back to the
    same place in the sky about 23 h 56 min later. J2 turns the orbit planes
    west, so the ground track repeats some seconds sooner: the shift that
    best brings the sky back lies 5-15 s short of the sidereal day, at every
    station, and there every satellite is within 0.1° of elevation."""
    _, _, start = snapshot
    systems, _, _ = gnss.usable()
    gps = systems["GPS"]
    t = np.arange(0, 6 * 3600, 60.0)
    today = dop.ecef_positions(gps, t, start)
    shifts = np.arange(dop.SIDEREAL_DAY_S - 30, dop.SIDEREAL_DAY_S + 10, 1.0)
    later = [dop.ecef_positions(gps, t + s, start) for s in shifts]
    for station in STATIONS:
        args = (station.lat_deg, station.lon_deg, station.height_km)
        _, el = dop.lines_of_sight(today, *args)
        gaps = [np.max(np.abs(dop.lines_of_sight(p, *args)[1] - el)) for p in later]
        best = int(np.argmin(gaps))
        short_s = dop.SIDEREAL_DAY_S - shifts[best]
        assert 5 <= short_s <= 15, f"{station.name}: {short_s:.0f} s"
        assert gaps[best] < 0.1, f"{station.name}: {gaps[best]:.3f}°"
        sidereal = int(np.argmin(np.abs(shifts - dop.SIDEREAL_DAY_S)))
        assert gaps[best] < gaps[sidereal]  # better than a whole sidereal day


def test_galileo_sky_does_not_repeat_daily(snapshot):
    """Galileo orbits 17 times in 10 days: after one sidereal day its sky,
    and so its DOP, is a different one."""
    _, _, start = snapshot
    systems, _, _ = gnss.usable()
    t = np.arange(0, 6 * 3600, 60.0)
    today = dop.ecef_positions(systems["Galileo"], t, start)
    later = dop.ecef_positions(systems["Galileo"], t + dop.SIDEREAL_DAY_S, start)
    for station in STATIONS:
        args = (station.lat_deg, station.lon_deg, station.height_km)
        (u1, e1), (u2, e2) = (
            dop.lines_of_sight(today, *args),
            dop.lines_of_sight(later, *args),
        )
        gap = np.median(np.abs(dop.dop(u1, e1 >= 10).pdop - dop.dop(u2, e2 >= 10).pdop))
        assert gap > 0.05, f"{station.name}: {gap:.4f}"


def test_two_systems_beat_either_alone(at_stations):
    for station, result in at_stations.items():
        both = result[dop.BOTH]
        for name in ("GPS", "Galileo"):
            assert np.all(both.pdop <= result[name].pdop * (1 + 1e-9)), station
            np.testing.assert_array_equal(both.n, result["GPS"].n + result["Galileo"].n)


def test_snapshot_values_are_what_gnss_gives(at_stations):
    """Broad sanity bounds from GNSS practice with a 10° mask: 6-12 GPS
    satellites in view, PDOP around 1.5-2.5 for one system, near 1.2 for two."""
    for station, result in at_stations.items():
        for name in ("GPS", "Galileo"):
            d = result[name]
            assert d.n.min() >= 4 and np.all(np.isfinite(d.pdop)), station
            assert 6 <= d.n.mean() <= 12, (station, name, d.n.mean())
            assert 1.4 <= np.mean(d.pdop) <= 2.5, (station, name)
        both = result[dop.BOTH]
        assert 1.0 <= np.mean(both.pdop) <= 1.5, station
        assert np.max(both.pdop) < 3, station


def test_world_grid_matches_one_site_at_a_time(snapshot):
    positions, _, _ = snapshot
    sampled = {n: p[:, ::60] for n, p in positions.items()}
    lats, lons, p95 = dop.world_pdop(sampled, grid_deg=30)
    i, j = 4, 7
    for name, (pos, system) in dop.combinations(sampled).items():
        u, el = dop.lines_of_sight(pos, lats[i], lons[j])
        expected = np.nanpercentile(dop.dop(u, el >= 10, system).pdop, 95)
        assert p95[name][i, j] == pytest.approx(expected, rel=1e-12)


# ---------- Command line ----------
def test_cli_writes_the_summary_and_figures(tmp_path, monkeypatch):
    monkeypatch.setattr(dop, "MAP_GRID_DEG", 30.0)  # a coarse, fast map
    paths = dop.main(["--station", "tokyo"], output_dir=tmp_path)
    assert [p.name for p in paths.values()] == [
        "dop_tokyo.txt",
        "dop_tokyo.png",
        "dop_world.png",
    ]
    text = paths["summary"].read_text()
    assert "TOKYO" in text and dop.BOTH in text
    for path in paths.values():
        assert path.stat().st_size > 0


def test_readme_table_is_the_snapshots(at_stations):
    """The Belgrade table in the README is what the snapshot gives."""
    from pathlib import Path

    text = Path(__file__).parents[1].joinpath("README.md").read_text()
    for name, d in at_stations["Belgrade"].items():
        row = re.search(rf"^\| {re.escape(name)} \|(.*)\|$", text, re.MULTILINE)
        assert row, name
        cells = [c.strip().strip("*") for c in row[1].split("|")]
        assert cells == [
            f"{d.n.min():.0f}–{d.n.max():.0f}",
            f"{np.nanmean(d.pdop):.2f}",
            f"{np.nanpercentile(d.pdop, 95):.2f}",
            f"{np.nanmean(d.hdop):.2f}",
            f"{np.nanmean(d.vdop):.2f}",
        ], name
