"""Numerical orbit propagation with Earth's oblateness (J2).

The Kepler model treats the Earth as a point mass. The real Earth bulges at
the equator, and the largest term of that bulge, J2, is what turns orbits
around: it rotates the orbital plane (the ascending node) and the ellipse
within it (the perigee). Here the equations of motion are integrated directly,

    a = -mu r / |r|^3 + a_J2(r),

with a fixed-step fourth-order Runge-Kutta integrator. Both models start from
the same state, SGP4's position and velocity at 00:00, so the comparison shows
only what each force model adds or leaves out:

- two-body (point mass) minus SGP4: the cost of ignoring the bulge;
- J2 minus SGP4: what is left, i.e. drag, higher harmonics and, for the high
  orbits, the Moon and the Sun (SGP4/SDP4 models all of those).

PULSAR-0 is sun-synchronous: J2 turns its orbital plane eastward by about
0.986° a day, once around in a year, so it crosses each latitude at the same
local solar time all year.

Constants are WGS84 (mu, equatorial radius, J2), the modern standard. SGP4 and
the GP elements use the older WGS72 values; the summary also shows the J2 run
with those, which alone moves the result by up to about a kilometre a day.

    python -m orbit_explorer.j2
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from orbit_explorer import ephemeris, style
from orbit_explorer.elements import MU_EARTH_KM3_S2, R_EARTH_KM, load_elements
from orbit_explorer.kepler import SPAN_S, START, time_grid_s
from orbit_explorer.sgp4_compare import propagate_sgp4, ric_components

OUTPUT_DIR = Path("outputs")


@dataclass(frozen=True)
class Earth:
    """Gravity constants: mu [km³/s²], equatorial radius [km], J2."""

    name: str
    mu: float
    radius_km: float
    j2: float


WGS84 = Earth("WGS84", MU_EARTH_KM3_S2, R_EARTH_KM, 1.08262668e-3)
WGS72 = Earth("WGS72", 398600.8, 6378.135, 1.082616e-3)  # what SGP4 uses
J2 = WGS84.j2
TROPICAL_YEAR_DAYS = 365.2422
SUN_SYNC_DEG_DAY = 360 / TROPICAL_YEAR_DAYS  # the Sun's mean motion, ~0.9856°/day
REPORT_HOURS = (1, 6, 12, 24)


# ---------- Force model and integrator ----------
def acceleration(r: np.ndarray, j2: bool = True, earth: Earth = WGS84) -> np.ndarray:
    """Gravitational acceleration [km/s²] at position(s) r [km], shape (..., 3).

    Point mass plus, if ``j2``, the J2 term of an axially symmetric Earth
    (z along the rotation axis). Because J2 is symmetric about that axis, the
    Earth's rotation doesn't matter and the field is the same in any inertial
    frame sharing the axis.
    """
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    r2 = x * x + y * y + z * z
    rn = np.sqrt(r2)
    point = -earth.mu / (r2 * rn)
    ax, ay, az = point * x, point * y, point * z
    if j2:
        k = 1.5 * earth.j2 * earth.mu * earth.radius_km**2 / (r2 * r2 * rn)
        zr2 = 5 * z * z / r2
        ax = ax + k * x * (zr2 - 1)
        ay = ay + k * y * (zr2 - 1)
        az = az + k * z * (zr2 - 3)
    return np.stack([ax, ay, az], axis=-1)


def propagate(
    r0: np.ndarray,
    v0: np.ndarray,
    t_s: np.ndarray,
    j2: bool = True,
    substeps: int = 1,
    earth: Earth = WGS84,
    third_body_start: datetime | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Positions [km] and velocities [km/s] on the uniform grid t_s, integrated
    with classic RK4.

    ``(r0, v0)`` is the state at ``t_s[0]``. Several orbits at once: give r0
    and v0 of shape (N, 3) and get (len(t_s), N, 3) back, all in one loop.

    ``third_body_start``: add the pull of the Moon and the Sun (ephemeris.py).
    Then the absolute times matter: t_s are seconds after that UTC instant.
    Without it only the step between grid points matters.

    ``substeps`` RK4 steps are taken between consecutive grid points; at the
    5 s grid one step is already far below a metre per day (see the tests).
    """
    r = np.array(r0, dtype=float)
    v = np.array(v0, dtype=float)
    r_out = np.empty((len(t_s), *r.shape))
    v_out = np.empty((len(t_s), *v.shape))
    if len(t_s) == 0:
        return r_out, v_out
    dt = np.diff(t_s)
    if len(dt) and not np.allclose(dt, dt[0]):
        raise ValueError("propagate needs a uniform time grid")
    h = (dt[0] if len(dt) else 0.0) / substeps

    n_steps = (len(t_s) - 1) * substeps
    if third_body_start is not None:
        # RK4 needs the bodies at every step and half step: compute them once
        half = t_s[0] + 0.5 * h * np.arange(2 * n_steps + 1)
        moon = ephemeris.moon_position(half, third_body_start)
        sun = ephemeris.sun_position(half, third_body_start)

    def acc(x, half_index):
        a = acceleration(x, j2, earth)
        if third_body_start is not None:
            m, s = moon[half_index], sun[half_index]
            a = a + ephemeris.third_body_acceleration(x, m, ephemeris.MU_MOON_KM3_S2)
            a = a + ephemeris.third_body_acceleration(x, s, ephemeris.MU_SUN_KM3_S2)
        return a

    r_out[0], v_out[0] = r, v
    step = 0
    for k in range(1, len(t_s)):
        for _ in range(substeps):
            i = 2 * step  # half-step index of this step's start
            a1 = acc(r, i)
            r2_, v2_ = r + 0.5 * h * v, v + 0.5 * h * a1
            a2 = acc(r2_, i + 1)
            r3_, v3_ = r + 0.5 * h * v2_, v + 0.5 * h * a2
            a3 = acc(r3_, i + 1)
            r4_, v4_ = r + h * v3_, v + h * a3
            a4 = acc(r4_, i + 2)
            step += 1
            r = r + h / 6 * (v + 2 * v2_ + 2 * v3_ + v4_)
            v = v + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        r_out[k], v_out[k] = r, v
    return r_out, v_out


def energy(r: np.ndarray, v: np.ndarray, j2: bool = True) -> np.ndarray:
    """Specific orbital energy [km²/s²], kinetic plus potential (with J2).
    Conserved by the exact motion, so its drift measures integrator error."""
    rn = np.linalg.norm(r, axis=-1)
    potential = -MU_EARTH_KM3_S2 / rn
    if j2:
        sin_lat2 = (r[..., 2] / rn) ** 2
        potential = potential + (
            MU_EARTH_KM3_S2 * J2 * R_EARTH_KM**2 / (2 * rn**3) * (3 * sin_lat2 - 1)
        )
    return 0.5 * np.sum(v * v, axis=-1) + potential


# ---------- Orbital elements from a state ----------
def osculating_elements(r: np.ndarray, v: np.ndarray) -> dict[str, np.ndarray]:
    """Semi-major axis [km], eccentricity, inclination and RAAN [deg] of the
    two-body orbit through each state (the "osculating" elements)."""
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=-1)
    rn = np.linalg.norm(r, axis=-1)
    v2 = np.sum(v * v, axis=-1)
    e_vec = np.cross(v, h) / MU_EARTH_KM3_S2 - r / rn[..., None]
    return {
        "a": 1 / (2 / rn - v2 / MU_EARTH_KM3_S2),
        "e": np.linalg.norm(e_vec, axis=-1),
        "i": np.degrees(np.arccos(h[..., 2] / hn)),
        "raan": np.degrees(np.arctan2(h[..., 0], -h[..., 1])) % 360,
    }


def j2_secular_rates(a_km: float, e: float, i_deg: float) -> tuple[float, float]:
    """Mean drift of the ascending node and of the perigee [deg/day] due to J2
    (first-order secular theory)."""
    n = np.sqrt(MU_EARTH_KM3_S2 / a_km**3)
    p = a_km * (1 - e * e)
    k = 1.5 * n * J2 * (R_EARTH_KM / p) ** 2
    cos_i = np.cos(np.radians(i_deg))
    raan_dot = -k * cos_i
    argp_dot = 0.5 * k * (5 * cos_i**2 - 1)
    return float(np.degrees(raan_dot) * 86400), float(np.degrees(argp_dot) * 86400)


def unwrap_deg(angle_deg: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(angle_deg)))


# ---------- Comparison with SGP4 ----------
# (key, label, with J2, constants): the runs compared with SGP4
MODELS = (
    ("two_body", "two-body", False, WGS84),
    ("j2", "J2", True, WGS84),
    ("j2_wgs72", "J2 WGS72", True, WGS72),
)


def compare_all(
    sats: list[dict], t_s: np.ndarray, start: datetime = START
) -> dict[int, dict]:
    """For each satellite, the MODELS runs from SGP4's state at t_s[0], each
    minus SGP4: total distance and radial/in-track/cross-track components
    [km]. All satellites go through the integrator together, one run per model.
    """
    sgp4 = [propagate_sgp4(sat, t_s, start) for sat in sats]
    r0 = np.array([r[0] for r, _ in sgp4])
    v0 = np.array([v[0] for _, v in sgp4])
    out = {sat["NORAD_CAT_ID"]: {"sgp4": states} for sat, states in zip(sats, sgp4)}
    for key, _, with_j2, earth in MODELS:
        r_all, v_all = propagate(r0, v0, t_s, j2=with_j2, earth=earth)
        for k, (sat, (r_sgp4, v_sgp4)) in enumerate(zip(sats, sgp4, strict=True)):
            diff = r_all[:, k] - r_sgp4
            out[sat["NORAD_CAT_ID"]][key] = {
                "r": r_all[:, k],
                "v": v_all[:, k],
                "total": np.linalg.norm(diff, axis=1),
                "ric": ric_components(diff, r_sgp4, v_sgp4),
            }
    return out


def compare(sat: dict, t_s: np.ndarray) -> dict:
    """compare_all for one satellite."""
    return compare_all([sat], t_s)[sat["NORAD_CAT_ID"]]


def build_summary(results: dict[int, dict] | None = None) -> str:
    t = time_grid_s()
    sats = load_elements()
    if results is None:
        results = compare_all(sats, t)
    idx = [int(h * 3600 / (t[1] - t[0])) for h in REPORT_HOURS]
    hours = "".join(f"{f'{h:02d}h':>9s}" for h in REPORT_HOURS)
    window = f"{START:%Y-%m-%d %H:%M} UTC + {SPAN_S // 3600} h"
    lines = [
        "TWO-BODY vs J2, BOTH AGAINST SGP4",
        f"Window: {window}, RK4 step {t[1] - t[0]:.0f} s",
        (
            f"Constants: {WGS84.name} (mu {WGS84.mu} km³/s², "
            f"R {WGS84.radius_km} km, J2 {WGS84.j2:.8e})"
        ),
        "All runs start from SGP4's position and velocity at 00:00.",
        "",
        "Position difference from SGP4 [km]",
        f"{'Satellite':24s} {'model':>9s}{hours}{'max':>9s}",
    ]
    for sat in sats:
        res = results[sat["NORAD_CAT_ID"]]
        for key, label, _, _ in MODELS:
            total = res[key]["total"]
            at = "".join(f"{total[i]:9.2f}" for i in idx)
            lines.append(
                f"{sat['OBJECT_NAME'] if key == MODELS[0][0] else '':24s} "
                f"{label:>9s}{at}{total.max():9.2f}"
            )
    lines += [
        "",
        "J2 WGS72: the same J2 run with SGP4's own (older) constants. The gap to",
        "the WGS84 row is what the choice of constants alone does.",
        "",
        "Largest component of the J2 difference over the day [km]",
        f"{'Satellite':24s} {'radial':>9s} {'in-track':>9s} {'cross':>9s}",
    ]
    for sat in sats:
        radial, in_track, cross = np.abs(results[sat["NORAD_CAT_ID"]]["j2"]["ric"]).max(
            axis=0
        )
        lines.append(
            f"{sat['OBJECT_NAME']:24s} {radial:9.2f} {in_track:9.2f} {cross:9.2f}"
        )
    lines += [
        "",
        "Orbital plane drift (ascending node), J2 propagation [deg/day]",
        f"{'Satellite':24s} {'incl':>7s} {'numerical':>10s} {'theory':>8s}",
    ]
    for sat in sats:
        res = results[sat["NORAD_CAT_ID"]]
        drift, theory, incl = node_drift(res, t)
        lines.append(
            f"{sat['OBJECT_NAME']:24s} {incl:6.2f}° {drift:+10.4f} {theory:+8.4f}"
        )
    lines += [
        "",
        f"Sun-synchronous rate: {SUN_SYNC_DEG_DAY:+.4f} deg/day (360° per year).",
        "Numerical: least-squares slope of the osculating node over the day;",
        "theory: first-order J2 secular rate from the day's mean a, e and i.",
        "SES-5's node is ill-defined at 0.05° inclination.",
        "",
    ]
    return "\n".join(lines)


def node_drift(res: dict, t_s: np.ndarray) -> tuple[float, float, float]:
    """(numerical, theoretical) node drift [deg/day] and mean inclination."""
    el = osculating_elements(res["j2"]["r"], res["j2"]["v"])
    raan = unwrap_deg(el["raan"])
    days = t_s / 86400
    numerical = np.polyfit(days, raan, 1)[0]
    theory, _ = j2_secular_rates(el["a"].mean(), el["e"].mean(), el["i"].mean())
    return float(numerical), theory, float(el["i"].mean())


# ---------- Figures ----------
TWO_BODY_COLOR = style.SECONDARY
LABEL_HOUR = 13.0  # where the direct labels sit
LABEL_HALF_WIDTH_H = 1.6  # about the label's half width, in hours
SUN_SYNC_TOLERANCE = 0.03  # within 3% of the Sun's rate counts as sun-synchronous


def is_sun_synchronous(node_drift_deg_day: float) -> bool:
    return abs(node_drift_deg_day / SUN_SYNC_DEG_DAY - 1) <= SUN_SYNC_TOLERANCE


@style.styled
def plot_differences(path: Path, results: dict[int, dict]) -> None:
    """Distance from SGP4 for two-body and J2, one panel per satellite."""
    t = time_grid_s()
    h = t / 3600
    sats = load_elements()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.subplots_adjust(hspace=0.35, wspace=0.22)
    for ax, sat in zip(axes.flat, sats, strict=True):
        res = results[sat["NORAD_CAT_ID"]]
        color = style.sat_color(sat)
        two_body, with_j2 = res["two_body"]["total"], res["j2"]["total"]
        # skip 00:00, where both start exactly on SGP4 (log scale)
        ax.plot(h[1:], two_body[1:], color=TWO_BODY_COLOR, lw=1.8, zorder=2)
        ax.plot(h[1:], with_j2[1:], color=color, lw=2.2, zorder=3)
        # direct labels clear of their curve: the point-mass one above its
        # highest point near mid-day, the J2 one below its lowest
        near = np.abs(h - LABEL_HOUR) <= LABEL_HALF_WIDTH_H
        for values, text, above in (
            (two_body, "point-mass Earth", True),
            (with_j2, "with J2", False),
        ):
            y = values[near].max() if above else values[near].min()
            ax.annotate(
                text,
                xy=(LABEL_HOUR, y),
                xytext=(0, 5 if above else -5),
                textcoords="offset points",
                va="bottom" if above else "top",
                ha="center",
                color=style.SECONDARY,
                fontsize=10,
                zorder=4,
            )
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1e4)
        ticks = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1e4]
        names = ["1 m", "10 m", "100 m", "1 km", "10 km", "100 km", "1000 km", ""]
        ax.set_yticks(ticks, names)
        ax.minorticks_off()
        ax.yaxis.tick_right()
        ax.grid(True, which="major", axis="y")
        ax.set_title(style.short_name(sat))
        style.time_axis(ax, SPAN_S / 3600, step_h=6)
    style.header(
        axes[0, 0],
        "What J2 adds",
        "Distance from SGP4 for a point-mass Earth (grey) and with J2, both "
        f"starting on SGP4 at 00:00 · {style.date_label(START)} · UTC",
        caption="Force models",
        pad=36,  # above the panel title
    )
    fig.savefig(path)
    plt.close(fig)


@style.styled
def plot_node_drift(path: Path, sat: dict, res: dict) -> None:
    """The orbital plane turning over the day: J2 propagation against the Sun's
    mean motion; a point-mass Earth leaves it fixed."""
    t = time_grid_s()
    h = t / 3600
    color = style.sat_color(sat)
    j2_el = osculating_elements(res["j2"]["r"], res["j2"]["v"])
    j2_raan = unwrap_deg(j2_el["raan"])
    numerical, theory, _ = node_drift(res, t)

    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.plot(h, np.zeros_like(h), color=TWO_BODY_COLOR, lw=1.8, zorder=2)
    ax.plot(h, j2_raan - j2_raan[0], color=color, lw=2.2, zorder=3)
    ax.plot(
        h,
        SUN_SYNC_DEG_DAY * t / 86400,
        color=style.LABEL,
        lw=1.2,
        linestyle=(0, (4, 3)),
        zorder=4,
    )
    style.time_axis(ax, SPAN_S / 3600)
    ax.yaxis.tick_right()
    ax.yaxis.set_major_formatter("{x:+.2f}°")
    name = style.short_name(sat)
    if is_sun_synchronous(numerical):
        title, caption = f"{name}'s orbit turns with the Sun", "Sun-synchronous orbit"
    else:
        title, caption = f"{name}'s orbital plane turns", "Nodal drift"
    style.header(
        ax,
        title,
        f"Ascending node since 00:00 · J2 drift {numerical:+.3f}°/day "
        f"(theory {theory:+.3f}°/day) · {style.date_label(START)}",
        caption=caption,
    )
    style.dot_legend(
        ax,
        [
            "With J2",
            "Point-mass Earth: the plane stays put",
            f"The Sun's mean motion, {SUN_SYNC_DEG_DAY:.4f}°/day",
        ],
        [color, TWO_BODY_COLOR, style.LABEL],
    )
    fig.savefig(path)
    plt.close(fig)


def main(output_dir: Path = OUTPUT_DIR) -> None:
    t = time_grid_s()
    sats = load_elements()
    results = compare_all(sats, t)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(results)
    print(summary)
    (output_dir / "j2_comparison.txt").write_text(summary)
    plot_differences(output_dir / "j2_differences.png", results)
    leo = max(sats, key=lambda s: s["MEAN_MOTION"])  # the fastest: PULSAR-0
    plot_node_drift(output_dir / "j2_node_drift.png", leo, results[leo["NORAD_CAT_ID"]])
    print(
        f"Written {output_dir}/j2_comparison.txt, j2_differences.png "
        "and j2_node_drift.png"
    )


if __name__ == "__main__":
    main()
