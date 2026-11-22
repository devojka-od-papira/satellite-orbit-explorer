"""Sun and Moon positions against known events, and their pull on a satellite."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from orbit_explorer.ephemeris import (
    AU_KM,
    J2000_TT,
    MU_MOON_KM3_S2,
    MU_SUN_KM3_S2,
    TT_MINUS_UTC_S,
    centuries_tt,
    moon_position,
    obliquity_rad,
    sun_position,
    third_body_acceleration,
)


def at(year, month, day, hour=0, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def ecliptic(p, when):
    """Ecliptic longitude and latitude [deg] of an equatorial position."""
    eps = obliquity_rad(centuries_tt(0, when))
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    ye = np.cos(eps) * y + np.sin(eps) * z
    ze = -np.sin(eps) * y + np.cos(eps) * z
    lon = np.degrees(np.arctan2(ye, x)) % 360
    lat = np.degrees(np.arcsin(ze / np.linalg.norm(p, axis=1)))
    return lon[0], lat[0]


def angle_gap(a, b):
    return abs((a - b + 180) % 360 - 180)


def test_time_argument():
    assert centuries_tt(0, J2000_TT - timedelta(seconds=TT_MINUS_UTC_S)) == 0
    assert centuries_tt(86400 * 36525, J2000_TT) == pytest.approx(1, abs=1e-6)


# ---------- The Sun ----------
@pytest.mark.parametrize(
    "when, longitude",
    [
        (at(2026, 3, 20, 14, 46), 0),  # March equinox 2026
        (at(2026, 6, 21, 8, 24), 90),  # June solstice
        (at(2026, 9, 23, 0, 5), 180),  # September equinox
        (at(2026, 12, 21, 20, 50), 270),  # December solstice
    ],
    ids=["march-equinox", "june-solstice", "september-equinox", "december-solstice"],
)
def test_sun_at_equinoxes_and_solstices(when, longitude):
    """Event times are for the apparent Sun (aberration −20", nutation up to
    17"); the geometric position agrees to about 0.01°."""
    lon, lat = ecliptic(sun_position(0, when), when)
    assert angle_gap(lon, longitude) < 0.03
    assert lat == pytest.approx(0, abs=1e-9)


def test_sun_distance_at_perihelion_and_aphelion():
    near = np.linalg.norm(sun_position(0, at(2026, 1, 3, 17))) / AU_KM
    far = np.linalg.norm(sun_position(0, at(2026, 7, 6, 18))) / AU_KM
    assert near == pytest.approx(0.98331, abs=3e-4)
    assert far == pytest.approx(1.01664, abs=3e-4)


def test_sun_declination_at_the_june_solstice():
    p = sun_position(0, at(2026, 6, 21, 8, 24))[0]
    assert np.degrees(np.arcsin(p[2] / np.linalg.norm(p))) == pytest.approx(
        23.436, abs=0.01
    )


# ---------- The Moon ----------
@pytest.mark.parametrize(
    "when, elongation",
    [(at(2026, 1, 3, 10, 3), 180), (at(2026, 1, 18, 19, 52), 0)],
    ids=["full-moon", "new-moon"],
)
def test_moon_phases(when, elongation):
    """Full and new moon: the Moon's ecliptic longitude is the Sun's + 180° / + 0°."""
    moon_lon, moon_lat = ecliptic(moon_position(0, when), when)
    sun_lon, _ = ecliptic(sun_position(0, when), when)
    assert angle_gap(moon_lon - sun_lon, elongation) < 0.3
    assert abs(moon_lat) < 5.3  # the Moon's orbit is inclined 5.1° to the ecliptic


def test_moon_distance_over_a_month():
    t = np.arange(0, 30 * 86400, 3600.0)
    r = np.linalg.norm(moon_position(t, at(2026, 11, 1)), axis=1)
    assert 356_000 < r.min() < 370_000  # perigee
    assert 400_000 < r.max() < 407_000  # apogee
    assert r.mean() == pytest.approx(385_000, abs=3_000)


def test_vectorised_matches_one_by_one():
    t = np.array([0.0, 40000.0, 86400.0])
    start = at(2026, 11, 17)
    one = np.vstack([moon_position(x, start) for x in t])
    np.testing.assert_allclose(moon_position(t, start), one, rtol=1e-12)


# ---------- Third-body acceleration ----------
def test_no_tidal_pull_at_the_centre_of_the_earth():
    s = np.array([384_400.0, 0.0, 0.0])
    np.testing.assert_allclose(
        third_body_acceleration(np.zeros(3), s, MU_MOON_KM3_S2), 0, atol=1e-20
    )


def test_tidal_pull_stretches_along_the_line_and_squeezes_across():
    """Far away (the Sun) the tide is the textbook 2:1 stretch-to-squeeze."""
    s = np.array([AU_KM, 0.0, 0.0])
    near = third_body_acceleration(np.array([29_600.0, 0, 0]), s, MU_SUN_KM3_S2)
    far = third_body_acceleration(np.array([-29_600.0, 0, 0]), s, MU_SUN_KM3_S2)
    side = third_body_acceleration(np.array([0, 29_600.0, 0]), s, MU_SUN_KM3_S2)
    assert near[0] > 0 and far[0] < 0  # both pulled away from the Earth
    assert side[1] < 0  # squeezed towards the line
    assert near[0] == pytest.approx(-2 * side[1], rel=1e-3)


@pytest.mark.parametrize(
    "mu, d",
    [(MU_MOON_KM3_S2, 384_400.0), (MU_SUN_KM3_S2, AU_KM)],
    ids=["moon", "sun"],
)
def test_size_of_the_pull_at_galileo_altitude(mu, d):
    """On the line to the body: exactly mu (1/(d - r)² - 1/d²). That is about
    2 mu r / d³ (Sun 2.3e-9 km/s²), but for the Moon, at only 13 Earth-Galileo
    distances, 12% more (5.8e-9): the Moon pulls 2.5 times as hard."""
    r = 29_600.0
    a = third_body_acceleration(np.array([r, 0, 0]), np.array([d, 0.0, 0.0]), mu)
    assert a[0] == pytest.approx(mu * (1 / (d - r) ** 2 - 1 / d**2), rel=1e-12)
    assert np.linalg.norm(a) == pytest.approx(
        5.77e-9 if mu == MU_MOON_KM3_S2 else 2.35e-9, rel=0.02
    )
