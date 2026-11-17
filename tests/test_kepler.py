from pathlib import Path

import numpy as np
import pytest

from orbit_explorer.elements import (
    MU_EARTH_KM3_S2,
    load_elements,
    mean_motion_rad_s,
    semi_major_axis_km,
)
from orbit_explorer.kepler import propagate_eci, solve_kepler, time_grid_s

WORKSHOP_DATA = Path(__file__).parent / "data" / "workshop_2026-11-10.json"


def find_sat(name: str, path: Path | None = None) -> dict:
    """Satelit po delu imena (ne zavisi od velikih/malih slova)."""
    for sat in load_elements(path) if path else load_elements():
        if name.lower() in sat["OBJECT_NAME"].lower():
            return sat
    raise KeyError(name)


# ---------- 1. Vremenska mreža ----------
def test_time_grid():
    t = time_grid_s()
    assert len(t) == 17281
    assert t[0] == 0
    assert t[-1] == 86400
    assert np.all(np.diff(t) == 5)


# ---------- 2. Keplerova jednačina ----------
def test_circular_orbit_E_equals_M():
    M = np.linspace(0, 2 * np.pi, 100)
    E = solve_kepler(M, e=0.0)
    np.testing.assert_allclose(E, M, atol=1e-12)


@pytest.mark.parametrize("e", [0.0, 0.001, 0.1, 0.5, 0.9])
def test_kepler_residual(e):
    M = np.linspace(0, 2 * np.pi, 1000)
    E = solve_kepler(M, e)
    residual = E - e * np.sin(E) - M  # hint: koliko se E - e·sin(E) razlikuje od M
    assert np.max(np.abs(residual)) < 1e-10


# ---------- 3. Periodi prema referenci ----------
# Reference values from the GNSS Academy workshop, for the GP data downloaded on
# 2026-11-10. They are checked against that saved dataset, not the live data,
# because fresh elements change (QZS-2's period moves after station-keeping).
REFERENCE_PERIODS = [
    ("PULSAR-0", 1.578),
    ("GALILEO", 14.078),  # ako ima više Galileo satelita, upiši tačno ime (29)
    ("SES-5", 23.935),
    ("QZS-2", 23.940),
]


@pytest.mark.parametrize("name, expected_hours", REFERENCE_PERIODS)
def test_period_matches_reference(name, expected_hours):
    sat = find_sat(name, WORKSHOP_DATA)
    n = mean_motion_rad_s(sat["MEAN_MOTION"])  # rad/s
    period_hours = 2 * np.pi / n / 3600  # hint: T = 2π / n, pa sekunde → sati
    assert period_hours == pytest.approx(expected_hours, abs=1e-3)


# ---------- 4. Fizičke provere propagacije ----------
@pytest.fixture(params=[s["OBJECT_NAME"] for s in load_elements()])
def sat(request):
    return find_sat(request.param)


def test_radius_between_perigee_and_apogee(sat):
    pos, _ = propagate_eci(sat, time_grid_s())
    r = np.linalg.norm(pos, axis=1)
    a = semi_major_axis_km(mean_motion_rad_s(sat["MEAN_MOTION"]))
    e = sat["ECCENTRICITY"]
    r_perigee = a * (1 - e)  # hint: najmanja udaljenost na elipsi
    r_apogee = a * (1 + e)  # hint: najveća udaljenost
    assert r.min() >= r_perigee - 1e-6
    assert r.max() <= r_apogee + 1e-6


def test_orbital_energy_constant(sat):
    """Specifična energija v²/2 − μ/r mora biti ista u svakoj tački i jednaka −μ/(2a)."""
    pos, v = propagate_eci(sat, time_grid_s())
    r = np.linalg.norm(pos, axis=1)
    a = semi_major_axis_km(mean_motion_rad_s(sat["MEAN_MOTION"]))
    energy = (
        v**2 / 2 - MU_EARTH_KM3_S2 / r
    )  # hint: kinetička minus potencijalna, po jedinici mase
    expected = -MU_EARTH_KM3_S2 / (2 * a)  # hint: −μ / (2a)
    np.testing.assert_allclose(energy, expected, rtol=1e-9)


def test_faster_near_perigee(sat):
    """Drugi Keplerov zakon: najveća brzina je tamo gde je r najmanje."""
    pos, v = propagate_eci(sat, time_grid_s())
    r = np.linalg.norm(pos, axis=1)
    if sat["ECCENTRICITY"] < 1e-4:
        pytest.skip("skoro kružna orbita, brzina je praktično konstantna")
    assert np.argmax(v) == np.argmin(r)
