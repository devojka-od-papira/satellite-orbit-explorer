"""Frame transformations: ECI to ECEF, and ECEF to geodetic (WGS84)."""

from datetime import UTC, datetime

import numpy as np

from orbit_explorer.elements import load_elements
from orbit_explorer.kepler import START, propagate_eci, time_grid_s

WGS84_A_KM = 6378.137
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)  # first eccentricity squared

J2000 = datetime(2000, 1, 1, 12, tzinfo=UTC)


def gmst_rad(t_s: np.ndarray) -> np.ndarray:
    """Greenwich mean sidereal angle at times t_s after START (simple linear model)."""
    days = ((START - J2000).total_seconds() + t_s) / 86400.0
    deg = 280.46061837 + 360.98564736629 * days
    return np.radians(deg % 360.0)


def eci_to_ecef(pos_eci: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Rotate ECI positions about the z axis by the Earth rotation angle."""
    x, y, z = pos_eci[:, 0], pos_eci[:, 1], pos_eci[:, 2]
    xe = x * np.cos(theta) + y * np.sin(theta)
    ye = -x * np.sin(theta) + y * np.cos(theta)
    return np.column_stack([xe, ye, z])


def ecef_to_geodetic(pos_ecef: np.ndarray, iterations: int = 10):
    """Geodetic latitude [deg], longitude [deg] and height [km] on WGS84."""
    x, y, z = pos_ecef[:, 0], pos_ecef[:, 1], pos_ecef[:, 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - WGS84_E2))  # initial guess
    for _ in range(iterations):
        N = WGS84_A_KM / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - WGS84_E2 * N / (N + h)))
    return np.degrees(lat), np.degrees(lon), h


if __name__ == "__main__":
    t = time_grid_s()
    theta = gmst_rad(t)
    for sat in load_elements():
        pos_eci, _ = propagate_eci(sat, t)
        lat, lon, h = ecef_to_geodetic(eci_to_ecef(pos_eci, theta))
        print(
            f"{sat['OBJECT_NAME']:24s} lat={lat.min():7.2f}..{lat.max():6.2f}  "
            f"lon={lon.min():8.2f}..{lon.max():7.2f}  h={h.min():9.1f}..{h.max():9.1f} km"
        )
