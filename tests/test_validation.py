"""Validation against precise orbits: frames, Earth orientation and results.

Everything here is for 2026-11-17, the day of the stored SP3 file and GP
elements in data/inputs/precise/, so the results are fixed numbers.
"""

import numpy as np
import pytest
from sgp4.api import jday
from sgp4.propagation import gstime

from orbit_explorer.elements import MU_EARTH_KM3_S2
from orbit_explorer.frames import eci_to_ecef, gmst_rad
from orbit_explorer.sp3 import PRECISE_ORBITS, SP3_IDS, read_sp3
from orbit_explorer.validation import (
    EARTH_ROTATION_RAD_S,
    EOP_2026_11_17,
    MODELS,
    NO_EOP,
    PRECISE_START,
    build_summary,
    frame_effect_km,
    gmst_ut1,
    itrf_to_teme,
    polar_motion,
    precise_state,
    teme_to_itrf,
    time_grid,
    validate,
)

GALILEO, QZS2 = 59598, 42738
T = time_grid()


@pytest.fixture(scope="module")
def results():
    return validate()


# ---------- Frames and Earth orientation ----------
def test_gmst_matches_the_sgp4_library():
    """Vectorised IAU 1982 GMST, the angle TEME is defined with."""
    t = np.array([0.0, 3600.0, 43200.0, 86000.0])
    jd, fr = jday(2026, 11, 17, 0, 0, 0)
    expected = [gstime(jd + fr + x / 86400) for x in t]
    np.testing.assert_allclose(gmst_ut1(t, PRECISE_START, NO_EOP), expected, atol=1e-8)


def test_ut1_shifts_the_earth_angle_by_dut1():
    t = np.array([0.0, 50000.0])
    shifted = gmst_ut1(t + EOP_2026_11_17.ut1_minus_utc_s, PRECISE_START, NO_EOP)
    np.testing.assert_allclose(
        gmst_ut1(t, PRECISE_START, EOP_2026_11_17), shifted, atol=1e-12
    )


def test_polar_motion_is_a_small_rotation():
    w = polar_motion(EOP_2026_11_17)
    np.testing.assert_allclose(w @ w.T, np.eye(3), atol=1e-15)
    np.testing.assert_array_equal(polar_motion(NO_EOP), np.eye(3))
    # 0.12" and 0.33" move a point at Galileo's radius by tens of metres
    r = 29600.0 * np.array([0.6, 0.6, np.sqrt(0.28)])
    shift = np.linalg.norm(w @ r - r)
    assert 0.01 < shift < 0.06


def test_without_eop_it_matches_the_simple_frames_module():
    """frames.py's linear GMST (no T² term, about 0.1" off in 2026) and no
    polar motion: the same to about 15 m at GNSS radius."""
    t = np.array([0.0, 21600.0, 86000.0])
    r = np.array([[26000.0, 12000.0, 9000.0]] * 3)
    simple = eci_to_ecef(r, gmst_rad(t, PRECISE_START))
    precise = teme_to_itrf(r, t, eop=NO_EOP)
    assert np.linalg.norm(simple - precise, axis=1).max() < 0.02  # km


def test_teme_itrf_round_trip():
    t = np.array([0.0, 7200.0, 80000.0])
    r = np.array([[26000.0, 12000.0, 9000.0], [-30000, 5000, -20000], [1, 2, 42000]])
    back, _ = itrf_to_teme(teme_to_itrf(r, t), np.zeros((3, 3)), t)
    np.testing.assert_allclose(back, r, atol=1e-9)


def test_a_point_fixed_on_the_earth_moves_with_its_rotation():
    """Velocity zero in ITRF is omega x r in TEME (about 3 km/s at GEO radius)."""
    t = np.array([0.0])
    r_itrf = np.array([[42164.0, 0.0, 0.0]])
    r, v = itrf_to_teme(r_itrf, np.zeros((1, 3)), t, eop=NO_EOP)
    np.testing.assert_allclose(
        np.linalg.norm(v), EARTH_ROTATION_RAD_S * 42164, rtol=1e-12
    )
    assert abs(np.dot(r[0], v[0])) < 1e-9  # perpendicular to the radius


@pytest.mark.parametrize(
    "sat_id, a_km", [("E29", 29600), ("J02", 42164)], ids=["E29", "J02"]
)
def test_precise_inertial_state_has_the_right_orbit(sat_id, a_km):
    """The SP3 state turned inertial gives the satellite's semi-major axis
    (vis-viva): the derivative and the rotation are both right."""
    sp3 = read_sp3(PRECISE_ORBITS)
    t = T[::720]
    r, v = itrf_to_teme(*precise_state(sp3, sat_id, t), t)
    rn, v2 = np.linalg.norm(r, axis=1), np.sum(v * v, axis=1)
    a = 1 / (2 / rn - v2 / MU_EARTH_KM3_S2)
    assert np.abs(a - a_km).max() < 150  # km (J2 short-period terms, QZS e)


def test_grid_stays_inside_the_sp3_day():
    sp3 = read_sp3(PRECISE_ORBITS)
    assert T[0] == 0 and T[-1] + 1 <= sp3.seconds_since(PRECISE_START)[-1]
    assert np.all(np.diff(T) == 5)


# ---------- Results for 2026-11-17 ----------
def test_no_missing_values(results):
    for res in results.values():
        for key, _ in MODELS:
            assert np.isfinite(res[key]["total"]).all()


@pytest.mark.parametrize("norad", list(SP3_IDS), ids=lambda n: SP3_IDS[n])
def test_sgp4_is_within_kilometres_and_kepler_far_off(results, norad):
    res = results[norad]

    def rms(key):
        return np.sqrt(np.mean(res[key]["total"] ** 2))

    assert rms("sgp4") < 10  # km: GALILEO 29 0.75, QZS-2 4.4
    assert rms("kepler") > 4 * rms("sgp4")  # mean elements read as osculating


@pytest.mark.parametrize("norad", list(SP3_IDS), ids=lambda n: SP3_IDS[n])
def test_j2_from_the_precise_state_starts_on_the_truth_and_drifts(results, norad):
    """Its error is the force model alone: tiny at first, kilometres by the end
    of the day without the Moon and the Sun."""
    total = results[norad]["j2_precise"]["total"]
    assert total[0] < 1e-3  # km: interpolation and frames, under a metre
    assert total[T <= 3600].max() < 0.1  # 100 m after an hour
    assert 1 < total.max() < 20  # km after a day


def test_frame_effect_of_earth_orientation(results):
    """UT1-UTC (-0.072 s) and polar motion are worth 0.1-0.3 km here."""
    effect = frame_effect_km()
    assert effect.keys() == SP3_IDS.keys()
    for km in effect.values():
        assert 0.05 < km < 0.5


def test_summary(results):
    summary = build_summary(results, {n: 0.1 for n in SP3_IDS})
    for res in results.values():
        assert res["sat"]["OBJECT_NAME"] in summary
    for _, label in MODELS:
        assert label in summary
    assert EOP_2026_11_17.source in summary
