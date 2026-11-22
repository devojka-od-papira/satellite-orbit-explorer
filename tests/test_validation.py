"""Validation against precise orbits: frames, Earth orientation and results.

Everything here is for 2026-11-17, the day of the stored SP3 file and GP
elements in data/inputs/precise/, so the results are fixed numbers.
"""

import json

import numpy as np
import pytest
from sgp4.api import jday
from sgp4.propagation import gstime

from orbit_explorer.elements import MU_EARTH_KM3_S2, load_elements
from orbit_explorer.frames import eci_to_ecef, gmst_rad
from orbit_explorer.sgp4_compare import propagate_sgp4
from orbit_explorer.sp3 import PRECISE_MANIFEST, PRECISE_ORBITS, SP3_IDS, read_sp3
from orbit_explorer.validation import (
    EARTH_ROTATION_RAD_S,
    MODELS,
    NO_EOP,
    PRECISE_DAY,
    PRECISE_EOP,
    PRECISE_GP,
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
    shifted = gmst_ut1(t + PRECISE_EOP.ut1_minus_utc_s, PRECISE_START, NO_EOP)
    np.testing.assert_allclose(
        gmst_ut1(t, PRECISE_START, PRECISE_EOP), shifted, atol=1e-12
    )


def test_polar_motion_is_a_small_rotation():
    w = polar_motion(PRECISE_EOP)
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


@pytest.mark.parametrize("norad", list(SP3_IDS), ids=lambda n: SP3_IDS[n])
def test_precise_inertial_state_matches_sgp4s_orbit(norad):
    """The SP3 state turned inertial has the same osculating semi-major axis
    as SGP4 to within a fraction of a kilometre (vis-viva): the derivative and
    ω x r are both right. A velocity error of 3 m/s would show as 50-90 km."""
    sp3 = read_sp3(PRECISE_ORBITS)
    t = T[::360]
    r, v = itrf_to_teme(*precise_state(sp3, SP3_IDS[norad], t), t)
    sat = next(s for s in load_elements(PRECISE_GP) if s["NORAD_CAT_ID"] == norad)
    r_sgp4, v_sgp4 = propagate_sgp4(sat, t, PRECISE_START)

    def semi_major_axis(r, v):
        rn, v2 = np.linalg.norm(r, axis=1), np.sum(v * v, axis=1)
        return 1 / (2 / rn - v2 / MU_EARTH_KM3_S2)

    gap = semi_major_axis(r, v) - semi_major_axis(r_sgp4, v_sgp4)
    assert np.abs(gap).max() < 2  # km (0.2 for E29, 0.35 for J02)


# ---------- One source for the precise day ----------
def test_the_day_agrees_with_the_sp3_file_and_the_gp_manifest():
    """The manifest names the day; the SP3 file and the GP data must be of it."""
    sp3 = read_sp3(PRECISE_ORBITS)
    assert sp3.epochs[0].date() == PRECISE_START.date()  # GPS-time midnight
    gp_manifest = json.loads(PRECISE_DAY.gp_manifest_path.read_text())
    assert gp_manifest["reference_start_utc"] == f"{PRECISE_START:%Y-%m-%dT%H:%M:%SZ}"
    gp_epochs = [s["EPOCH"][:10] for s in load_elements(PRECISE_GP)]
    assert max(gp_epochs) <= f"{PRECISE_START:%Y-%m-%d}"  # none from a later day


def test_earth_orientation_comes_from_the_manifest():
    e = json.loads(PRECISE_MANIFEST.read_text())["earth_orientation"]
    assert (PRECISE_EOP.xp_arcsec, PRECISE_EOP.yp_arcsec) == (
        e["xp_arcsec"],
        e["yp_arcsec"],
    )
    assert PRECISE_EOP.ut1_minus_utc_s == e["ut1_minus_utc_s"]
    assert str(e["mjd"]) in PRECISE_EOP.source


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


@pytest.mark.parametrize("norad", list(SP3_IDS), ids=lambda n: SP3_IDS[n])
def test_the_moon_and_the_sun_are_most_of_j2s_error(results, norad):
    """Adding them as point masses cuts the force model's error several times
    (GALILEO 29: 1.5 -> 0.14 km RMS; QZS-2: 3.3 -> 0.36 km) and brings it
    below SGP4's."""
    res = results[norad]

    def rms(key):
        return np.sqrt(np.mean(res[key]["total"] ** 2))

    assert rms("j2_moon_sun_precise") < rms("j2_precise") / 4
    assert rms("j2_moon_sun_precise") < rms("sgp4")
    assert res["j2_moon_sun_precise"]["total"].max() < 2  # km over the day


@pytest.mark.parametrize("norad", list(SP3_IDS), ids=lambda n: SP3_IDS[n])
def test_what_is_left_is_the_size_of_radiation_pressure(results, norad):
    """The remaining error is the same order as the estimated drift from solar
    radiation pressure (within a factor of 10 either way), not much bigger."""
    from orbit_explorer.ephemeris import drift_km
    from orbit_explorer.validation import leftover_estimates

    srp_km = drift_km(
        dict(leftover_estimates())["Solar radiation pressure (A/m 0.02 m²/kg)"]
    )
    left = np.sqrt(np.mean(results[norad]["j2_moon_sun_precise"]["total"] ** 2))
    assert srp_km / 10 < left < srp_km * 10


@pytest.mark.parametrize("norad", list(SP3_IDS), ids=lambda n: SP3_IDS[n])
def test_the_starting_state_matters_more_than_the_force_model(results, norad):
    """From SGP4's state the full force model starts with SGP4's error and ends
    near it (GALILEO 29: 0.86 vs 0.75 km RMS; QZS-2 2.5 vs 4.4); from the
    precise state the same model is several times closer."""
    res = results[norad]

    def rms(key):
        return np.sqrt(np.mean(res[key]["total"] ** 2))

    assert res["j2_moon_sun_sgp4"]["total"][0] == pytest.approx(
        res["sgp4"]["total"][0], abs=1e-9
    )
    assert rms("j2_moon_sun_sgp4") < rms("j2_sgp4")  # Moon and Sun help either way
    assert rms("j2_moon_sun_precise") < rms("j2_moon_sun_sgp4") / 4
    assert 0.5 < rms("j2_moon_sun_sgp4") / rms("sgp4") < 2  # same league as SGP4


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
    assert PRECISE_EOP.source in summary
