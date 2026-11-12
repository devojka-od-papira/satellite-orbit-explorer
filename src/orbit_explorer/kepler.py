"""Keplerian propagation: from orbital elements to ECI positions over time."""

from datetime import UTC, datetime

import numpy as np

from orbit_explorer.elements import (
    MU_EARTH_KM3_S2,
    load_elements,
    mean_motion_rad_s,
    semi_major_axis_km,
)

START = datetime(2026, 11, 10, tzinfo=UTC)
SPAN_S = 86400
STEP_S = 5


def time_grid_s() -> np.ndarray:
    """Seconds since START, one day at a fixed step (17281 points)."""
    return np.arange(0, SPAN_S + STEP_S, STEP_S, dtype=float)


def solve_kepler(M: np.ndarray, e: float, tol: float = 1e-12) -> np.ndarray:
    """Solve Kepler's equation M = E - e*sin(E) for E with Newton-Raphson."""
    E = M.copy()  # good starting guess for small eccentricity
    for _ in range(50):
        f = E - e * np.sin(E) - M  # the equation written as f(E) = 0
        f_prime = 1 - e * np.cos(E)  # derivative of f with respect to E
        step = f / f_prime
        E = E - step
        if np.max(np.abs(step)) < tol:
            break
    return E


def propagate_eci(sat: dict, t_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ECI positions [km] and speeds [km/s] at times t_s after START."""
    n = mean_motion_rad_s(sat["MEAN_MOTION"])
    a = semi_major_axis_km(n)
    e = sat["ECCENTRICITY"]
    i = np.radians(sat["INCLINATION"])
    raan = np.radians(sat["RA_OF_ASC_NODE"])
    argp = np.radians(sat["ARG_OF_PERICENTER"])
    M0 = np.radians(sat["MEAN_ANOMALY"])

    epoch = datetime.fromisoformat(sat["EPOCH"]).replace(tzinfo=UTC)
    dt = t_s - (epoch - START).total_seconds()  # negative before the epoch

    M = M0 + n * dt
    M = np.remainder(M, 2 * np.pi)  # wrap to [0, 2*pi)
    E = solve_kepler(M, e)  # step 2
    r = a * (1 - e * np.cos(E))  # step 3: radius at each time
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    u = argp + nu  # argument of latitude

    x = r * (np.cos(raan) * np.cos(u) - np.sin(raan) * np.sin(u) * np.cos(i))
    y = r * (np.sin(raan) * np.cos(u) + np.cos(raan) * np.sin(u) * np.cos(i))
    z = r * (np.sin(u) * np.sin(i))

    v = np.sqrt(MU_EARTH_KM3_S2 * (2 / r - 1 / a))  # vis-viva equation
    return np.column_stack([x, y, z]), v


if __name__ == "__main__":
    t = time_grid_s()
    for sat in load_elements():
        pos, v = propagate_eci(sat, t)
        r = np.linalg.norm(pos, axis=1)
        print(
            f"{sat['OBJECT_NAME']:24s} points={len(t)}  "
            f"r={r.min():8.1f}..{r.max():8.1f} km  v={v.min():.3f}..{v.max():.3f} km/s"
        )
