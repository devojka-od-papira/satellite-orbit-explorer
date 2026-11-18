import numpy as np
import pytest

from orbit_explorer.elements import load_elements
from orbit_explorer.frames import geodetic_to_ecef
from orbit_explorer.kepler import time_grid_s
from orbit_explorer.visibility import (
    BELGRADE,
    ELEVATION_MASK_DEG,
    Station,
    build_summary,
    clock,
    enu_rotation,
    find_passes,
    look_angles,
    parse_station,
    satellite_look_angles,
    with_mask_crossings,
)

T = time_grid_s()


def find_sat(name):
    return next(s for s in load_elements() if name.lower() in s["OBJECT_NAME"].lower())


# ---------- Look angles ----------
def test_enu_rotation_is_orthonormal():
    R = enu_rotation(BELGRADE.lat_deg, BELGRADE.lon_deg)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_point_straight_up_is_at_zenith():
    st = BELGRADE
    above = geodetic_to_ecef(st.lat_deg, st.lon_deg, st.height_km + 500)
    _, el, rng = look_angles(above, st)
    assert el[0] == pytest.approx(90, abs=1e-6)
    assert rng[0] == pytest.approx(500, abs=1e-6)


@pytest.mark.parametrize(
    "row, expected_az", [(1, 0.0), (0, 90.0)], ids=["north", "east"]
)
def test_local_horizontal_directions(row, expected_az):
    """A point 100 km along the local north/east axis: that azimuth, elevation 0."""
    st = Station("equator", 0.0, 0.0)
    site = geodetic_to_ecef(st.lat_deg, st.lon_deg, st.height_km)
    target = site + 100 * enu_rotation(st.lat_deg, st.lon_deg)[row]
    az, el, _ = look_angles(target, st)
    assert az[0] == pytest.approx(expected_az, abs=1e-6)
    assert el[0] == pytest.approx(0, abs=1e-6)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        satellite_look_angles(load_elements()[0], T[:3], BELGRADE, model="sgp8")


# ---------- Pass detection ----------
def test_find_passes_interpolates_crossings():
    t = np.arange(0, 1001, 10.0)
    el = 30 * np.sin(2 * np.pi * t / 1000)  # above 10° between ~54 s and ~446 s
    passes = find_passes(t, el, np.zeros_like(t), mask=10)
    assert len(passes) == 1
    expected_rise = np.arcsin(1 / 3) / (2 * np.pi) * 1000
    assert passes[0].rise_s == pytest.approx(expected_rise, abs=1)
    assert passes[0].set_s == pytest.approx(500 - expected_rise, abs=1)
    assert passes[0].max_el_deg == pytest.approx(30, abs=0.1)
    assert passes[0].rises_in_window and passes[0].sets_in_window


def test_find_passes_at_window_edges():
    t = np.arange(0, 101, 1.0)
    el = np.where((t < 20) | (t > 80), 45.0, -5.0)  # up at both ends
    passes = find_passes(t, el, np.zeros_like(t), mask=10)
    assert len(passes) == 2
    assert not passes[0].rises_in_window and passes[0].rise_s == 0
    assert not passes[1].sets_in_window and passes[1].set_s == 100


def test_never_above_mask_gives_no_passes():
    t = np.arange(0, 100, 1.0)
    assert find_passes(t, np.full_like(t, 5.0), np.zeros_like(t), mask=10) == []


# ---------- The four satellites from Belgrade ----------
def test_geostationary_stays_put_in_the_southern_sky():
    az, el, _ = satellite_look_angles(find_sat("SES-5"), T, BELGRADE)
    assert np.ptp(el) < 1 and np.ptp(az) < 1
    assert el.min() > ELEVATION_MASK_DEG
    assert 180 < az.mean() < 225  # south, a little west (SES-5 is at 5° E)


def test_leo_passes_are_short():
    az, el, _ = satellite_look_angles(find_sat("PULSAR"), T, BELGRADE)
    passes = find_passes(T, el, az)
    assert len(passes) >= 2
    assert all(p.set_s - p.rise_s < 15 * 60 for p in passes)


def test_qzss_is_regional():
    """QZS-2 serves Asia-Oceania: from Belgrade it never clears the mask."""
    _, el, _ = satellite_look_angles(find_sat("QZS-2"), T, BELGRADE)
    assert el.max() < ELEVATION_MASK_DEG


def test_qzss_is_high_over_tokyo():
    _, el, _ = satellite_look_angles(
        find_sat("QZS-2"), T, Station("Tokyo", 35.68, 139.77)
    )
    assert el.max() > 70


def test_clock_shows_window_end_as_24h():
    assert clock(0) == "00:00:00"
    assert clock(3661) == "01:01:01"
    assert clock(86400) == "24:00:00"


def test_summary_and_cli():
    station = parse_station(["--name", "Novi Sad", "--lat", "45.25", "--lon", "19.84"])
    assert station == Station("Novi Sad", 45.25, 19.84, BELGRADE.height_km)
    summary = build_summary(station)
    assert "NOVI SAD" in summary
    for sat in load_elements():
        assert sat["OBJECT_NAME"] in summary


def test_mask_crossings_are_inserted_on_the_line():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    el = np.array([0.0, 20.0, 20.0, 0.0])
    x_c, el_c = with_mask_crossings(x, el, 10)
    np.testing.assert_allclose(x_c, [0, 0.5, 1, 2, 2.5, 3])
    np.testing.assert_allclose(el_c, [0, 10, 20, 20, 10, 0])


def test_peak_just_below_mask_gets_no_visible_points():
    """The figure must agree with find_passes: 9.7° is not a pass."""
    x = np.arange(5.0)
    el = np.array([0, 5, 9.7, 5, 0])
    x_c, el_c = with_mask_crossings(x, el, 10)
    assert len(x_c) == len(x)
    assert not (el_c >= 10).any()
    assert find_passes(x, el, np.zeros_like(x), mask=10) == []
