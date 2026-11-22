"""The Sun and the Moon: low-precision positions and their pull on a satellite.

- **Sun:** Meeus, *Astronomical Algorithms* (2nd ed.), chapter 25, the
  low-accuracy method: about 0.01° in longitude. (Montenbruck & Gill's
  shorter series keeps the perigee of the Earth's orbit fixed and drifts
  about 0.09° off by 2026.)
- **Moon:** Montenbruck & Gill, *Satellite Orbits* (2000), section 3.3.2: a
  few arcminutes and about 500 km.

For the third-body acceleration on a GNSS satellite that is plenty: the pull
itself is about 5e-9 km/s², and a 0.1° direction error changes it by well
under 1%.

Positions are geometric, geocentric, equatorial, referred to the mean equinox
and equator of date, the frame TEME approximates (TEME uses the true equator;
the ~10" nutation difference is negligible here). The time argument is TT,
which is UTC + 69.184 s since 2017 (37 leap seconds + 32.184 s).
"""

from datetime import UTC, datetime

import numpy as np

MU_SUN_KM3_S2 = 1.32712440018e11
MU_MOON_KM3_S2 = 4902.800066
TT_MINUS_UTC_S = 69.184  # 37 leap seconds + 32.184 s (valid since 2017-01-01)
J2000_TT = datetime(2000, 1, 1, 12, tzinfo=UTC)  # JD 2451545.0 TT
AU_KM = 149597870.7

ARCSEC = np.pi / (180 * 3600)


def centuries_tt(t_s: np.ndarray, start_utc: datetime) -> np.ndarray:
    """Julian centuries of TT since J2000 at times t_s after a UTC instant."""
    seconds = (start_utc - J2000_TT).total_seconds() + TT_MINUS_UTC_S
    return (seconds + np.asarray(t_s, dtype=float)) / (86400 * 36525)


def obliquity_rad(T: np.ndarray) -> np.ndarray:
    """Mean obliquity of the ecliptic of date."""
    return np.radians(23.43929111 - 0.0130042 * T)


def _ecliptic_to_equatorial(
    lon: np.ndarray, lat: np.ndarray, r: np.ndarray, eps: np.ndarray
) -> np.ndarray:
    x = r * np.cos(lon) * np.cos(lat)
    y = r * np.sin(lon) * np.cos(lat)
    z = r * np.sin(lat)
    return np.column_stack(
        [x, np.cos(eps) * y - np.sin(eps) * z, np.sin(eps) * y + np.cos(eps) * z]
    )


def sun_position(t_s: np.ndarray, start_utc: datetime) -> np.ndarray:
    """Geocentric position of the Sun [km], equatorial of date, shape (n, 3)."""
    T = np.atleast_1d(centuries_tt(t_s, start_utc))
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T**2  # mean longitude, of date
    M = np.radians(357.52911 + 35999.05029 * T - 0.0001537 * T**2)
    e = 0.016708634 - 0.000042037 * T - 0.0000001267 * T**2
    C = (
        (1.914602 - 0.004817 * T - 0.000014 * T**2) * np.sin(M)
        + (0.019993 - 0.000101 * T) * np.sin(2 * M)
        + 0.000289 * np.sin(3 * M)
    )  # equation of centre [deg]
    lon = np.radians(L0 + C)
    true_anomaly = M + np.radians(C)
    r_au = 1.000001018 * (1 - e**2) / (1 + e * np.cos(true_anomaly))
    return _ecliptic_to_equatorial(
        lon, np.zeros_like(lon), r_au * AU_KM, obliquity_rad(T)
    )


def moon_position(t_s: np.ndarray, start_utc: datetime) -> np.ndarray:
    """Geocentric position of the Moon [km], equatorial of date, shape (n, 3)."""
    T = np.atleast_1d(centuries_tt(t_s, start_utc))
    # mean longitude (equinox of date) and the fundamental arguments
    L0 = np.radians(218.31617 + 481267.88088 * T)
    l = np.radians(134.96292 + 477198.86753 * T)
    lp = np.radians(357.52543 + 35999.04944 * T)  # Sun's mean anomaly
    F = np.radians(93.27283 + 483202.01873 * T)  # argument of latitude
    D = np.radians(297.85027 + 445267.11135 * T)  # elongation from the Sun
    lon = L0 + ARCSEC * (
        22640 * np.sin(l)
        + 769 * np.sin(2 * l)
        - 4586 * np.sin(l - 2 * D)
        + 2370 * np.sin(2 * D)
        - 668 * np.sin(lp)
        - 412 * np.sin(2 * F)
        - 212 * np.sin(2 * l - 2 * D)
        - 206 * np.sin(l + lp - 2 * D)
        + 192 * np.sin(l + 2 * D)
        - 165 * np.sin(lp - 2 * D)
        + 148 * np.sin(l - lp)
        - 125 * np.sin(D)
        - 110 * np.sin(l + lp)
        - 55 * np.sin(2 * F - 2 * D)
    )
    lat = ARCSEC * (
        18520 * np.sin(F + lon - L0 + ARCSEC * (412 * np.sin(2 * F) + 541 * np.sin(lp)))
        - 526 * np.sin(F - 2 * D)
        + 44 * np.sin(l + F - 2 * D)
        - 31 * np.sin(-l + F - 2 * D)
        - 25 * np.sin(-2 * l + F)
        - 23 * np.sin(lp + F - 2 * D)
        + 21 * np.sin(-l + F)
        + 11 * np.sin(-lp + F - 2 * D)
    )
    r = (
        385000
        - 20905 * np.cos(l)
        - 3699 * np.cos(2 * D - l)
        - 2956 * np.cos(2 * D)
        - 570 * np.cos(2 * l)
        + 246 * np.cos(2 * l - 2 * D)
        - 205 * np.cos(lp - 2 * D)
        - 171 * np.cos(l + 2 * D)
        - 152 * np.cos(l + lp - 2 * D)
    )
    return _ecliptic_to_equatorial(lon, lat, r, obliquity_rad(T))


def third_body_acceleration(r: np.ndarray, s: np.ndarray, mu: float) -> np.ndarray:
    """Acceleration [km/s²] of a satellite at r from a body at s (both
    geocentric, km): the body's pull on the satellite minus its pull on the
    Earth, which the geocentric frame already falls with."""
    d = s - r
    dn = np.linalg.norm(d, axis=-1, keepdims=True)
    sn = np.linalg.norm(s, axis=-1, keepdims=True)
    return mu * (d / dn**3 - s / sn**3)
