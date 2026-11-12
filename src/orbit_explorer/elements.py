"""Keplerian elements derived from CelesTrak GP (OMM) records."""

import json
import math
from pathlib import Path

MU_EARTH_KM3_S2 = 398600.4418  # Earth's gravitational parameter
R_EARTH_KM = 6378.137  # equatorial radius, used for spherical-Earth altitudes
SECONDS_PER_DAY = 86400.0

DATA_PATH = Path("data/inputs/orbital_elements.json")


def load_elements(path: Path = DATA_PATH) -> list[dict]:
    """Read GP JSON records, one dict per satellite."""
    with path.open() as f:
        return json.load(f)


def mean_motion_rad_s(mean_motion_rev_day: float) -> float:
    """Convert mean motion from revolutions per day to radians per second."""
    return mean_motion_rev_day * 2 * math.pi / SECONDS_PER_DAY


def semi_major_axis_km(n_rad_s: float) -> float:
    """Semi-major axis from mean motion: a = (mu / n^2)^(1/3)."""
    return (MU_EARTH_KM3_S2 / n_rad_s**2) ** (1 / 3)


def period_hours(n_rad_s: float) -> float:
    """Orbital period in hours."""
    return 2 * math.pi / n_rad_s / 3600.0


def perigee_apogee_altitude_km(a_km: float, e: float) -> tuple[float, float]:
    """Perigee and apogee altitude above a spherical Earth."""
    perigee = a_km * (1 - e) - R_EARTH_KM
    apogee = a_km * (1 + e) - R_EARTH_KM
    return perigee, apogee


if __name__ == "__main__":
    for sat in load_elements():
        n = mean_motion_rad_s(sat["MEAN_MOTION"])
        a = semi_major_axis_km(n)
        hp, ha = perigee_apogee_altitude_km(a, sat["ECCENTRICITY"])
        print(
            f"{sat['OBJECT_NAME']:24s} a={a:9.1f} km  T={period_hours(n):7.3f} h  "
            f"perigee={hp:8.1f} km  apogee={ha:8.1f} km"
        )
