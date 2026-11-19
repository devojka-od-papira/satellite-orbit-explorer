"""The J2 propagator: force model, integrator accuracy and the physics it shows."""

import numpy as np
import pytest

from orbit_explorer.elements import MU_EARTH_KM3_S2, R_EARTH_KM, load_elements
from orbit_explorer.j2 import (
    J2,
    SUN_SYNC_DEG_DAY,
    acceleration,
    build_summary,
    compare,
    energy,
    j2_secular_rates,
    node_drift,
    osculating_elements,
    propagate,
)
from orbit_explorer.kepler import time_grid_s

SATS = {s["NORAD_CAT_ID"]: s for s in load_elements()}
PULSAR, GALILEO, SES5, QZS2 = 64541, 59598, 38652, 42738
T = time_grid_s()


@pytest.fixture(scope="module")
def results():
    """One day of two-body and J2 propagation for every satellite (~5 s)."""
    return {norad: compare(sat, T) for norad, sat in SATS.items()}


def circular_state(a_km: float, incl_deg: float = 0.0):
    """Position and velocity on a circular orbit, node on the x axis."""
    v = np.sqrt(MU_EARTH_KM3_S2 / a_km)
    i = np.radians(incl_deg)
    return np.array([a_km, 0.0, 0.0]), np.array([0.0, v * np.cos(i), v * np.sin(i)])


# ---------- Force model ----------
def test_point_mass_is_inverse_square():
    r = np.array([7000.0, -1000.0, 2500.0])
    rn = np.linalg.norm(r)
    np.testing.assert_allclose(
        acceleration(r, j2=False), -MU_EARTH_KM3_S2 * r / rn**3, rtol=1e-14
    )


def test_j2_pulls_harder_at_the_equator_than_at_the_pole():
    """Textbook values: -mu/r² ∓ J2 terms of 3/2 and 3 (mu R²/r⁴)."""
    r = 7000.0
    extra = J2 * MU_EARTH_KM3_S2 * R_EARTH_KM**2 / r**4
    equator = acceleration(np.array([r, 0.0, 0.0]))
    pole = acceleration(np.array([0.0, 0.0, r]))
    assert equator[0] == pytest.approx(-MU_EARTH_KM3_S2 / r**2 - 1.5 * extra)
    assert pole[2] == pytest.approx(-MU_EARTH_KM3_S2 / r**2 + 3 * extra)
    np.testing.assert_allclose(equator[1:], 0, atol=1e-18)
    np.testing.assert_allclose(pole[:2], 0, atol=1e-18)


def test_acceleration_is_the_gradient_of_the_potential_in_energy():
    """acceleration() and energy() use the same J2 field: a = -grad(U)."""
    r = np.array([5200.0, 3100.0, 4400.0])
    zero_v = np.zeros(3)
    step = 1e-3  # km
    grad = np.array(
        [
            (energy(r + step * e, zero_v) - energy(r - step * e, zero_v)) / (2 * step)
            for e in np.eye(3)
        ]
    )
    np.testing.assert_allclose(acceleration(r), -grad, rtol=1e-7)


def test_vectorised_acceleration_matches_one_by_one():
    r = np.array([[7000.0, 0, 100], [0, 26000.0, 9000], [-42164.0, 10, -5]])
    np.testing.assert_allclose(
        acceleration(r), np.array([acceleration(x) for x in r]), rtol=1e-15
    )


# ---------- Integrator ----------
def test_two_body_circle_matches_the_exact_solution():
    """A circular LEO orbit for a day: RK4 at 5 s stays within a metre."""
    a = R_EARTH_KM + 500
    r0, v0 = circular_state(a)
    r, _ = propagate(r0, v0, T, j2=False)
    n = np.sqrt(MU_EARTH_KM3_S2 / a**3)
    exact = a * np.column_stack([np.cos(n * T), np.sin(n * T), np.zeros_like(T)])
    assert np.linalg.norm(r - exact, axis=1).max() < 1e-3  # km


def test_energy_and_polar_angular_momentum_are_conserved_with_j2():
    """J2 is axially symmetric and time-independent: energy and h_z are
    constants of the motion, so their drift is pure integration error."""
    r0, v0 = circular_state(R_EARTH_KM + 500, incl_deg=97.5)
    r, v = propagate(r0, v0, T, j2=True)
    # RK4 isn't symplectic, so energy drifts slowly (~1e-11 here); 1e-10 of
    # the energy is a semi-major axis error of about a millimetre.
    e = energy(r, v)
    assert np.abs(e - e[0]).max() / abs(e[0]) < 1e-10
    hz = np.cross(r, v)[:, 2]
    assert np.abs(hz - hz[0]).max() / abs(hz[0]) < 1e-10


def test_halving_the_step_changes_little():
    t = T[: 6 * 720 + 1]  # six hours
    r0, v0 = circular_state(R_EARTH_KM + 500, incl_deg=97.5)
    one, _ = propagate(r0, v0, t, substeps=1)
    two, _ = propagate(r0, v0, t, substeps=2)
    assert np.linalg.norm(one - two, axis=1).max() < 1e-4  # 10 cm


def test_non_uniform_grid_is_refused():
    r0, v0 = circular_state(7000.0)
    with pytest.raises(ValueError, match="uniform"):
        propagate(r0, v0, np.array([0.0, 5.0, 12.0]))


# ---------- Elements and J2 theory ----------
def test_osculating_elements_of_a_known_orbit():
    r0, v0 = circular_state(7000.0, incl_deg=30)
    el = osculating_elements(r0, v0)
    assert el["a"] == pytest.approx(7000.0)
    assert el["e"] == pytest.approx(0, abs=1e-12)
    assert el["i"] == pytest.approx(30)
    assert el["raan"] == pytest.approx(0, abs=1e-9)  # node on the +x axis


def test_secular_rates_textbook_cases():
    a = R_EARTH_KM + 500
    raan_polar, _ = j2_secular_rates(a, 0, 90)
    assert raan_polar == pytest.approx(0, abs=1e-12)  # a polar plane doesn't turn
    raan_pro, _ = j2_secular_rates(a, 0, 45)
    raan_retro, _ = j2_secular_rates(a, 0, 135)
    assert raan_pro < 0 < raan_retro  # westward if prograde, eastward if retrograde
    _, argp_critical = j2_secular_rates(a, 0.01, np.degrees(np.arccos(np.sqrt(0.2))))
    assert argp_critical == pytest.approx(0, abs=1e-9)  # the 63.4° "frozen" perigee


# ---------- What the propagation shows ----------
@pytest.mark.parametrize(
    "norad", [PULSAR, GALILEO, QZS2], ids=["pulsar", "galileo", "qzs2"]
)
def test_numerical_node_drift_matches_j2_theory(results, norad):
    numerical, theory, _ = node_drift(results[norad], T)
    assert numerical == pytest.approx(theory, rel=0.02)


def test_pulsar_is_sun_synchronous(results):
    numerical, _, incl = node_drift(results[PULSAR], T)
    assert 96 < incl < 99  # retrograde, as sun-synchronous LEO orbits are
    assert numerical == pytest.approx(SUN_SYNC_DEG_DAY, rel=0.02)


def test_j2_explains_most_of_sgp4_for_low_earth_orbit(results):
    res = results[PULSAR]
    two_body, with_j2 = res["two_body"]["total"], res["j2"]["total"]
    assert two_body[-1] > 500  # km: a point-mass Earth is useless in LEO
    assert with_j2.max() < 10  # what's left is drag and higher harmonics
    assert with_j2[-1] < two_body[-1] / 100


@pytest.mark.parametrize("norad", list(SATS), ids=lambda n: SATS[n]["OBJECT_NAME"])
def test_j2_is_closer_to_sgp4_than_a_point_mass_by_the_end(results, norad):
    res = results[norad]
    assert res["j2"]["total"][-1] < res["two_body"]["total"][-1]


def test_both_models_start_on_sgp4(results):
    for res in results.values():
        assert res["two_body"]["total"][0] == 0
        assert res["j2"]["total"][0] == 0


def test_summary_lists_every_satellite_and_the_sun_rate(results):
    summary = build_summary(results)
    for sat in SATS.values():
        assert sat["OBJECT_NAME"] in summary
    assert f"{SUN_SYNC_DEG_DAY:+.4f}" in summary
    assert "SES-5's node is ill-defined" in summary
    assert SES5 in results
