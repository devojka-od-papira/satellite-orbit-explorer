import numpy as np
import pytest

from orbit_explorer.elements import load_elements
from orbit_explorer.frames import (
    J2000,
    WGS84_A_KM,
    WGS84_F,
    ecef_to_geodetic,
    eci_to_ecef,
    geodetic_to_ecef,
    gmst_rad,
)
from orbit_explorer.kepler import START, propagate_eci, time_grid_s

B_KM = WGS84_A_KM * (1 - WGS84_F)  # polarni radijus
SIDEREAL_DAY_S = 86164.0905


def find_sat(name):
    return next(s for s in load_elements() if name.lower() in s["OBJECT_NAME"].lower())


# ---------- GMST ----------
def test_gmst_at_j2000():
    t = np.array([(J2000 - START).total_seconds()])  # J2000 kao "sekunde posle START"
    expected = np.radians(280.46061837)
    assert gmst_rad(t)[0] == pytest.approx(expected, abs=1e-9)


def test_gmst_repeats_after_sidereal_day():
    a, b = gmst_rad(np.array([0.0, SIDEREAL_DAY_S]))
    assert np.cos(a - b) == pytest.approx(1.0, abs=1e-9)


# ---------- ECI → ECEF ----------
def test_eci_to_ecef_zero_angle_is_identity():
    pos = np.array([[7000.0, 1000.0, -500.0]])
    np.testing.assert_allclose(eci_to_ecef(pos, np.array([0.0])), pos)


def test_rotation_keeps_length_and_z():
    pos = np.random.default_rng(0).normal(size=(100, 3)) * 7000
    theta = np.linspace(0, 2 * np.pi, 100)
    out = eci_to_ecef(pos, theta)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), np.linalg.norm(pos, axis=1))
    np.testing.assert_allclose(out[:, 2], pos[:, 2])


# ---------- ECEF → geodetske koordinate ----------
def test_equator_prime_meridian():
    lat, lon, h = ecef_to_geodetic(np.array([[WGS84_A_KM, 0.0, 0.0]]))
    assert (lat[0], lon[0], h[0]) == pytest.approx((0, 0, 0), abs=1e-9)


@pytest.mark.parametrize(
    "z, expected_lat, expected_h",
    [
        (B_KM, 90.0, 0.0),  # severni pol na elipsoidu
        (-B_KM - 500, -90.0, 500.0),  # 500 km iznad južnog pola
    ],
)
def test_poles(z, expected_lat, expected_h):
    lat, _, h = ecef_to_geodetic(np.array([[0.0, 0.0, z]]))
    assert lat[0] == pytest.approx(expected_lat, abs=1e-9)
    assert h[0] == pytest.approx(expected_h, abs=1e-6)


def test_round_trip():
    lat = np.array([44.8125, -89.9, 0.0, 60.0])  # Beograd, blizu pola, ekvator, sever
    lon = np.array([20.4612, -170.0, 0.0, 10.0])
    h = np.array([0.117, 35786.0, 0.0, 400.0])  # tlo, GEO, nula, ISS
    lat2, lon2, h2 = ecef_to_geodetic(geodetic_to_ecef(lat, lon, h))
    np.testing.assert_allclose(lat2, lat, atol=1e-9)
    np.testing.assert_allclose(lon2, lon, atol=1e-9)
    np.testing.assert_allclose(h2, h, atol=1e-6)  # 1e-6 km = 1 mm


# ---------- Ceo lanac na pravim satelitima ----------
def ground_track(sat):
    t = time_grid_s()
    pos_eci, _ = propagate_eci(sat, t)
    return ecef_to_geodetic(eci_to_ecef(pos_eci, gmst_rad(t)))


def test_geo_stays_put():
    """SES-5 je geostacionaran: za 24 h geografska dužina se skoro ne pomera."""
    _, lon, _ = ground_track(find_sat("SES-5"))
    lon_range = np.ptp(np.degrees(np.unwrap(np.radians(lon))))
    print(f"SES-5 raspon dužine: {lon_range:.3f}°")
    assert lon_range < 1.0


@pytest.mark.parametrize("sat", load_elements(), ids=lambda s: s["OBJECT_NAME"])
def test_max_latitude_equals_inclination(sat):
    """Najveća geografska širina ground tracka ≈ inklinacija orbite."""
    lat, _, _ = ground_track(sat)
    i = sat["INCLINATION"]
    expected = min(i, 180 - i)  # retrogradna orbita (i > 90°) → 180° − i
    assert np.max(np.abs(lat)) == pytest.approx(expected, abs=0.3)
