"""Solar radiation pressure: sunlight pushing on a satellite.

Two models, both scaled with the inverse square of the distance to the Sun
and switched off in the Earth's shadow:

- **cannonball**: one parameter, the acceleration at 1 AU straight away from
  the Sun (a sphere: C_R A/m times the pressure);
- **DYB**: three constant accelerations in the Sun-oriented frame GNSS
  analysis centres use (the constant part of CODE's ECOM, Beutler et al.
  1994): D away from the Sun, Y along the solar panel axis, B completing the
  frame. A Y-bias, a small push along the panel axis, is a known feature of
  GNSS satellites.

Parameters are in units of 1e-9 km/s² (µm/s²) at 1 AU, the size of the effect
on a GNSS satellite (about 0.1).

The shadow is the conical model of Montenbruck & Gill (2000), section 3.4.2:
the fraction of the Sun's disc visible past a spherical Earth, 1 in sunlight,
0 in the umbra and in between in the penumbra.
"""

import numpy as np

from orbit_explorer.ephemeris import AU_KM

R_SUN_KM = 696000.0
R_EARTH_SHADOW_KM = 6378.137  # equatorial radius; no atmosphere
UNIT_KM_S2 = 1e-9  # the parameters' unit
MODELS = {"none": 0, "cannonball": 1, "dyb": 3}  # model: number of parameters


def shadow_fraction(r: np.ndarray, sun: np.ndarray) -> np.ndarray:
    """Fraction of the Sun's disc visible from geocentric positions r [km]
    with the Sun at ``sun`` [km], shape (..., 3) each."""
    d = sun - r
    dn = np.linalg.norm(d, axis=-1)
    rn = np.linalg.norm(r, axis=-1)
    a = np.arcsin(np.clip(R_SUN_KM / dn, -1, 1))  # apparent radius of the Sun
    b = np.arcsin(np.clip(R_EARTH_SHADOW_KM / rn, -1, 1))  # ... of the Earth
    c = np.arccos(np.clip(np.sum(-r * d, axis=-1) / (rn * dn), -1, 1))  # apart
    with np.errstate(invalid="ignore", divide="ignore"):
        x = (c * c + a * a - b * b) / (2 * c)
        y = np.sqrt(np.maximum(a * a - x * x, 0))
        overlap = (
            a * a * np.arccos(np.clip(x / a, -1, 1))
            + b * b * np.arccos(np.clip((c - x) / b, -1, 1))
            - c * y
        )
    partial = (np.abs(a - b) < c) & (c < a + b)
    nu = np.where(partial, 1 - overlap / (np.pi * a * a), 1.0)
    return np.where(c <= b - a, 0.0, nu)  # the Earth covers the whole Sun


def sun_frame(r: np.ndarray, sun: np.ndarray) -> tuple[np.ndarray, ...]:
    """Unit vectors D (satellite to Sun), Y (solar panel axis, perpendicular
    to the Sun and the Earth directions) and B = D x Y."""
    d = sun - r
    e_d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    e_z = -r / np.linalg.norm(r, axis=-1, keepdims=True)  # towards the Earth
    e_y = np.cross(e_z, e_d)
    e_y /= np.linalg.norm(e_y, axis=-1, keepdims=True)
    return e_d, e_y, np.cross(e_d, e_y)


def acceleration(
    r: np.ndarray, sun: np.ndarray, params: np.ndarray, model: str
) -> np.ndarray:
    """SRP acceleration [km/s²] at positions r (N, 3) for parameters
    ``params`` (N, MODELS[model]), one row per orbit."""
    if model == "none":
        return np.zeros_like(r)
    if params.shape[-1] != MODELS[model]:
        raise ValueError(f"{model} takes {MODELS[model]} parameters")
    dist = np.linalg.norm(sun - r, axis=-1, keepdims=True)
    scale = UNIT_KM_S2 * (AU_KM / dist) ** 2 * shadow_fraction(r, sun)[..., None]
    push = -params[:, :1] * (sun - r) / dist  # D: away from the Sun
    if model == "dyb":  # Y is undefined with the Sun exactly behind the Earth
        _, e_y, e_b = sun_frame(r, sun)
        push = push + params[:, 1:2] * e_y + params[:, 2:3] * e_b
    return scale * push
