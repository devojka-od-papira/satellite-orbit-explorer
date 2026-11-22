"""Validation against precise orbits: how far each model is from the truth.

The models so far were only compared with each other. Here they are compared
with GFZ's precise orbits (SP3, centimetre level) for the two satellites that
have them, GALILEO 29 and QZS-2, on 2026-11-17: the day of the stored SP3 file
and of the GP elements kept next to it in data/inputs/precise/.

Four models, all for that day:

- **Kepler** from the GP elements (two-body, mean elements read as osculating);
- **SGP4** from the same GP elements;
- **J2 from SGP4's state**: the numerical J2 propagator started on SGP4 at 00:00;
- **J2 from the precise state**: the same propagator started on the precise
  orbit at 00:00, so its error is the force model alone (no Moon, Sun or solar
  radiation pressure), not the GP elements.

Comparing needs the same frame and time scale on both sides:

- SP3 epochs are GPS time, converted to UTC (18 s, see sp3.py);
- SP3 positions are Earth-fixed (IGc20, i.e. ITRF2020). TEME positions are
  rotated by Greenwich mean sidereal time at UT1 (UT1 = UTC + dUT1) and then
  by polar motion, with the IERS values for the day (finals2000A). Without
  them the frame alone would add about 0.2 km.

Distances are computed in TEME, where the radial/in-track/cross-track split
uses the inertial velocity.

    python -m orbit_explorer.validation
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sgp4.api import jday

from orbit_explorer import j2, style
from orbit_explorer.elements import load_elements
from orbit_explorer.kepler import STEP_S, propagate_eci
from orbit_explorer.sgp4_compare import propagate_sgp4, ric_components
from orbit_explorer.sp3 import PRECISE_DIR, PRECISE_ORBITS, SP3_IDS, read_sp3

OUTPUT_DIR = Path("outputs")
PRECISE_GP = PRECISE_DIR / "gp_2026-11-17.json"  # the GP elements of that day
PRECISE_START = datetime(2026, 11, 17, tzinfo=UTC)
EARTH_ROTATION_RAD_S = 7.292115e-5  # IERS nominal mean angular velocity


@dataclass(frozen=True)
class EarthOrientation:
    """IERS Earth orientation parameters for one day."""

    xp_arcsec: float  # polar motion
    yp_arcsec: float
    ut1_minus_utc_s: float
    source: str


# IERS Rapid Service, finals2000A (datacenter.iers.org), line for MJD 61361:
# "261117 61361.00 I  0.118181 0.000042  0.327930 0.000042  I-0.0718374 ..."
EOP_2026_11_17 = EarthOrientation(
    0.118181, 0.327930, -0.0718374, "IERS finals2000A, MJD 61361 (flag I)"
)
NO_EOP = EarthOrientation(0.0, 0.0, 0.0, "none (UT1 = UTC, no polar motion)")

MODELS = (
    ("kepler", "Kepler (GP elements)"),
    ("sgp4", "SGP4 (GP elements)"),
    ("j2_sgp4", "J2 from SGP4's state"),
    ("j2_precise", "J2 from the precise state"),
)
REPORT_HOURS = (1, 6, 12, 23)  # the SP3 day ends at 23:59:42 UTC


# ---------- Frames ----------
def gmst_ut1(t_s: np.ndarray, start: datetime, eop: EarthOrientation) -> np.ndarray:
    """Greenwich mean sidereal time [rad] (IAU 1982, the angle TEME is defined
    with) at UT1 for times t_s after the UTC instant ``start``."""
    jd, fr = jday(start.year, start.month, start.day, start.hour, start.minute, 0)
    # days from J2000 kept as a small number, for precision
    days = (jd - 2451545.0) + fr + (np.asarray(t_s) + eop.ut1_minus_utc_s) / 86400
    t = days / 36525  # Julian centuries of UT1
    seconds = (
        67310.54841
        + (876600.0 * 3600 + 8640184.812866) * t
        + 0.093104 * t**2
        - 6.2e-6 * t**3
    )
    return np.radians(seconds / 240) % (2 * np.pi)  # 240 s of time = 1°


def polar_motion(eop: EarthOrientation) -> np.ndarray:
    """W with r_PEF = W @ r_ITRF (Vallado's convention)."""
    xp, yp = np.radians(np.array([eop.xp_arcsec, eop.yp_arcsec]) / 3600)
    cx, sx, cy, sy = np.cos(xp), np.sin(xp), np.cos(yp), np.sin(yp)
    return np.array(
        [
            [cx, 0.0, -sx],
            [sx * sy, cy, cx * sy],
            [sx * cy, -sy, cx * cy],
        ]
    )


def _rot3(theta: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v (n, 3) about z by theta (n,): x' = R3(theta) x."""
    c, s = np.cos(theta), np.sin(theta)
    return np.column_stack(
        [c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]]
    )


def teme_to_itrf(
    r_teme: np.ndarray,
    t_s: np.ndarray,
    start: datetime = PRECISE_START,
    eop: EarthOrientation = EOP_2026_11_17,
) -> np.ndarray:
    """Earth-fixed (ITRF) positions [km] from TEME positions."""
    r_pef = _rot3(gmst_ut1(t_s, start, eop), r_teme)
    return r_pef @ polar_motion(eop)  # r_ITRF = W.T @ r_PEF, row by row


def itrf_to_teme(
    r_itrf: np.ndarray,
    v_itrf: np.ndarray,
    t_s: np.ndarray,
    start: datetime = PRECISE_START,
    eop: EarthOrientation = EOP_2026_11_17,
) -> tuple[np.ndarray, np.ndarray]:
    """TEME position [km] and inertial velocity [km/s] from an Earth-fixed
    state; the velocity gains the Earth's rotation, omega x r."""
    w = polar_motion(eop)
    r_pef = r_itrf @ w.T
    v_pef = v_itrf @ w.T
    v_inertial_pef = v_pef + np.cross([0.0, 0.0, EARTH_ROTATION_RAD_S], r_pef)
    theta = gmst_ut1(t_s, start, eop)
    return _rot3(-theta, r_pef), _rot3(-theta, v_inertial_pef)


# ---------- Truth from SP3 ----------
def precise_state(
    sp3, sat_id: str, t_s: np.ndarray, start: datetime = PRECISE_START
) -> tuple[np.ndarray, np.ndarray]:
    """Earth-fixed position [km] and velocity [km/s] from the SP3 file; the
    velocity is the central difference of the interpolant over ±1 s."""
    from orbit_explorer.sp3 import positions_at

    r = positions_at(sp3, sat_id, t_s, start)
    v = (
        positions_at(sp3, sat_id, t_s + 1, start)
        - positions_at(sp3, sat_id, t_s - 1, start)
    ) / 2
    return r, v


def time_grid() -> np.ndarray:
    """The 5 s grid over the part of the day the SP3 file covers (it is in GPS
    time, so it ends 18 s before 24:00 UTC); the last point has a neighbour
    1 s later for the velocity."""
    sp3 = read_sp3(PRECISE_ORBITS)
    end = sp3.seconds_since(PRECISE_START)[-1] - 1
    return np.arange(0.0, end + 1e-9, STEP_S)


# ---------- Comparison ----------
def validate(
    eop: EarthOrientation = EOP_2026_11_17, t_s: np.ndarray | None = None
) -> dict[int, dict]:
    """Each model minus the precise orbit, in TEME, for each satellite with an
    SP3 orbit: total distance and radial/in-track/cross-track [km]."""
    t = time_grid() if t_s is None else t_s
    sp3 = read_sp3(PRECISE_ORBITS)
    sats = {s["NORAD_CAT_ID"]: s for s in load_elements(PRECISE_GP)}
    out = {}
    for norad, sat_id in SP3_IDS.items():
        sat = sats[norad]
        r_true, v_true = itrf_to_teme(*precise_state(sp3, sat_id, t), t, eop=eop)
        r_sgp4, v_sgp4 = propagate_sgp4(sat, t, PRECISE_START)
        r_kepler, _ = propagate_eci(sat, t, PRECISE_START)
        r_j2_sgp4, _ = j2.propagate(r_sgp4[0], v_sgp4[0], t)
        r_j2_precise, _ = j2.propagate(r_true[0], v_true[0], t)
        positions = {
            "kepler": r_kepler,
            "sgp4": r_sgp4,
            "j2_sgp4": r_j2_sgp4,
            "j2_precise": r_j2_precise,
        }
        res = {"sat": sat, "sat_id": sat_id, "t": t}
        for key, r in positions.items():
            diff = r - r_true
            res[key] = {
                "total": np.linalg.norm(diff, axis=1),
                "ric": ric_components(diff, r_true, v_true),
            }
        out[norad] = res
    return out


def frame_effect_km(t_s: np.ndarray | None = None) -> dict[int, float]:
    """Largest change of SGP4's distance from the truth when UT1 and polar
    motion are left out: what the Earth orientation data is worth."""
    t = time_grid() if t_s is None else t_s
    sp3 = read_sp3(PRECISE_ORBITS)
    sats = {s["NORAD_CAT_ID"]: s for s in load_elements(PRECISE_GP)}
    out = {}
    for norad, sat_id in SP3_IDS.items():
        r_sgp4, _ = propagate_sgp4(sats[norad], t, PRECISE_START)
        r_itrf, v_itrf = precise_state(sp3, sat_id, t)
        with_eop, _ = itrf_to_teme(r_itrf, v_itrf, t, eop=EOP_2026_11_17)
        without, _ = itrf_to_teme(r_itrf, v_itrf, t, eop=NO_EOP)
        gap = np.linalg.norm(r_sgp4 - with_eop, axis=1) - np.linalg.norm(
            r_sgp4 - without, axis=1
        )
        out[norad] = float(np.abs(gap).max())
    return out


def build_summary(results: dict[int, dict], frame_km: dict[int, float]) -> str:
    t = next(iter(results.values()))["t"]
    idx = [int(np.argmin(np.abs(t - h * 3600))) for h in REPORT_HOURS]
    hours = "".join(f"{f'{h:02d}h':>9s}" for h in REPORT_HOURS)
    lines = [
        "MODELS vs PRECISE ORBITS (SP3)",
        f"Day: {PRECISE_START:%Y-%m-%d} UTC, truth: {PRECISE_ORBITS.name}",
        (
            f'Earth orientation: {EOP_2026_11_17.source}: xp {EOP_2026_11_17.xp_arcsec}", '
            f'yp {EOP_2026_11_17.yp_arcsec}", UT1-UTC {EOP_2026_11_17.ut1_minus_utc_s} s'
        ),
        "",
        "Distance from the precise orbit [km]",
        f"{'Satellite':24s} {'model':26s}{hours}{'RMS':>9s}{'max':>9s}",
    ]
    for res in results.values():
        name = res["sat"]["OBJECT_NAME"]
        for k, (key, label) in enumerate(MODELS):
            total = res[key]["total"]
            at = "".join(f"{total[i]:9.2f}" for i in idx)
            rms = np.sqrt(np.mean(total**2))
            lines.append(
                f"{name if k == 0 else '':24s} {label:26s}{at}{rms:9.2f}{total.max():9.2f}"
            )
    lines += [
        "",
        "Largest component of each model's error over the day [km]",
        f"{'Satellite':24s} {'model':26s} {'radial':>8s} {'in-track':>9s} {'cross':>8s}",
    ]
    for res in results.values():
        name = res["sat"]["OBJECT_NAME"]
        for k, (key, label) in enumerate(MODELS):
            radial, in_track, cross = np.abs(res[key]["ric"]).max(axis=0)
            lines.append(
                f"{name if k == 0 else '':24s} {label:26s} {radial:8.2f} "
                f"{in_track:9.2f} {cross:8.2f}"
            )
    lines += [
        "",
        "Leaving out UT1-UTC and polar motion would change SGP4's distance by up to:",
    ]
    lines += [
        f"  {results[n]['sat']['OBJECT_NAME']:24s} {km:.3f} km"
        for n, km in frame_km.items()
    ]
    lines += [
        "",
        "J2 from the precise state starts on the truth, so its error is the force",
        "model alone: the Moon, the Sun and solar radiation pressure are missing.",
        "",
    ]
    return "\n".join(lines)


# ---------- Figure ----------
MODEL_STYLE = {
    "kepler": {"color": style.SECONDARY, "lw": 1.6, "linestyle": (0, (4, 3))},
    "sgp4": {"color": style.LABEL, "lw": 2.0},
    "j2_sgp4": {"lw": 1.6, "alpha": 0.55},  # satellite colour, lighter
    "j2_precise": {"lw": 2.4},  # satellite colour
}


@style.styled
def plot_validation(path: Path, results: dict[int, dict]) -> None:
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    fig.subplots_adjust(wspace=0.1)
    colors = [style.sat_color(res["sat"]) for res in results.values()]
    for k, (ax, res) in enumerate(zip(axes, results.values(), strict=True)):
        h = res["t"] / 3600
        for key, _ in MODELS:
            kw = {"color": colors[k], **MODEL_STYLE[key]}
            ax.plot(h[1:], res[key]["total"][1:], **kw)  # 00:00 is 0 for one
        ax.set_yscale("log")
        ax.minorticks_off()
        ax.grid(True, which="major", axis="y")
        ax.set_title(f"{style.short_name(res['sat'])} ({res['sat_id']})")
        style.time_axis(ax, 24, step_h=6)
    axes[0].set_xticks([0, 6, 12, 18], ["00:00", "06:00", "12:00", "18:00"])
    axes[0].set_ylim(1e-3, 300)
    axes[0].set_yticks(
        [1e-3, 1e-2, 1e-1, 1, 10, 100],
        ["1 m", "10 m", "100 m", "1 km", "10 km", "100 km"],
    )
    axes[0].tick_params(axis="y", left=False, labelleft=False)
    axes[1].tick_params(
        axis="y", left=False, labelleft=False, right=True, labelright=True
    )
    style.header(
        axes[0],
        "Models against the precise orbit",
        f"Distance from GFZ's SP3 orbit · {style.date_label(PRECISE_START)} · UTC",
        caption="Validation",
        pad=36,
    )

    def line(key, color):
        kw = {"color": color, **MODEL_STYLE[key]}
        return Line2D([], [], **kw)

    handles = [
        (line(key, colors[0]), line(key, colors[1]))
        if "color" not in MODEL_STYLE[key]
        else line(key, None)
        for key, _ in MODELS
    ]
    style.dot_legend(
        axes[0],
        [],
        [],
        extra=list(zip(handles, [label for _, label in MODELS], strict=True)),
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
        handlelength=3.2,
        handletextpad=0.6,
    )
    fig.savefig(path)
    plt.close(fig)


def main(output_dir: Path = OUTPUT_DIR) -> None:
    results = validate()
    summary = build_summary(results, frame_effect_km())
    print(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation.txt").write_text(summary)
    plot_validation(output_dir / "validation.png", results)
    print(f"Written {output_dir}/validation.txt and validation.png")


if __name__ == "__main__":
    main()
