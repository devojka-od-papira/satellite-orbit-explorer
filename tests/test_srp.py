"""Solar radiation pressure: the Earth's shadow, the cannonball and DYB models."""

import numpy as np
import pytest

from orbit_explorer import srp
from orbit_explorer.ephemeris import AU_KM, sun_position
from orbit_explorer.sp3 import PRECISE_ORBITS, SP3_IDS, read_sp3
from orbit_explorer.validation import PRECISE_START, itrf_to_teme, precise_state

SUN = np.array([AU_KM, 0.0, 0.0])
R = 29600.0


def test_shadow_sunlit_umbra_and_penumbra():
    sunlit = np.array([[R, 0, 0], [0, R, 0], [-R, 0, 8000.0]])
    np.testing.assert_array_equal(srp.shadow_fraction(sunlit, SUN), 1.0)
    assert srp.shadow_fraction(np.array([-R, 0, 0]), SUN) == 0.0  # behind the Earth
    # crossing the shadow's edge, the Sun fades smoothly from 1 to 0
    z = np.linspace(6000, 7000, 201)
    edge = np.column_stack([np.full_like(z, -R), np.zeros_like(z), z])
    nu = srp.shadow_fraction(edge, SUN)
    assert nu[0] == 0.0 and nu[-1] == 1.0
    assert np.all(np.diff(nu) >= 0)
    assert np.abs(np.diff(nu)).max() < 0.05
    assert ((nu > 0) & (nu < 1)).sum() > 20  # a penumbra, not a step


def test_penumbra_is_as_wide_as_the_suns_disc():
    """At GNSS radius the penumbra spans the Sun's 0.53° seen from there."""
    z = np.linspace(5000, 8000, 30001)
    edge = np.column_stack([np.full_like(z, -R), np.zeros_like(z), z])
    nu = srp.shadow_fraction(edge, SUN)
    width_km = np.ptp(z[(nu > 0) & (nu < 1)])
    expected = 2 * srp.R_SUN_KM / AU_KM * R  # the disc's width projected at R
    assert width_km == pytest.approx(expected, rel=0.05)


def test_cannonball_pushes_away_from_the_sun():
    r = np.array([[0.0, R, 0.0]])
    a = srp.acceleration(r, SUN + r, np.array([[0.1]]), "cannonball")  # 1 AU away
    np.testing.assert_allclose(a, [[-0.1e-9, 0, 0]], atol=1e-24)
    far = srp.acceleration(r, r + 2 * SUN, np.array([[0.1]]), "cannonball")
    np.testing.assert_allclose(far, a / 4, rtol=1e-12)  # inverse square
    behind = np.array([[-R, 0, 0]])
    assert not srp.acceleration(behind, SUN, np.array([[0.1]]), "cannonball").any()


def test_dyb_frame_and_its_d_term_is_the_cannonball():
    r = np.array([[R * 0.6, R * 0.8, 3000.0]])
    e_d, e_y, e_b = srp.sun_frame(r, SUN)
    frame = np.vstack([e_d, e_y, e_b])
    np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-14)
    assert abs(e_y @ r[0]) < 1e-9  # the panel axis is square to the Earth too
    np.testing.assert_allclose(
        srp.acceleration(r, SUN, np.array([[0.1, 0, 0]]), "dyb"),
        srp.acceleration(r, SUN, np.array([[0.1]]), "cannonball"),
        rtol=1e-14,
    )
    y = srp.acceleration(r, SUN, np.array([[0, 0.02, 0]]), "dyb")[0]
    np.testing.assert_allclose(y / np.linalg.norm(y), e_y[0], atol=1e-12)


def test_the_parameters_must_fit_the_model():
    with pytest.raises(ValueError, match="dyb takes 3"):
        srp.acceleration(np.array([[R, 0, 0]]), SUN, np.array([[0.1]]), "dyb")
    assert not srp.acceleration(
        np.array([[R, 0, 0]]), SUN, np.zeros((1, 0)), "none"
    ).any()


def test_no_eclipse_on_the_precise_day():
    """Neither satellite enters the shadow on 2026-11-17: the fit there can't
    tell how well the shadow is modelled."""
    sp3 = read_sp3(PRECISE_ORBITS)
    t = np.arange(0.0, 86000.0, 300.0)
    sun = sun_position(t, PRECISE_START)
    for sat_id in SP3_IDS.values():
        r, _ = itrf_to_teme(*precise_state(sp3, sat_id, t), t)
        assert srp.shadow_fraction(r, sun).min() == 1.0
