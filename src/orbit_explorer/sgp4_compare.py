"""Compare our two-body Kepler propagation with SGP4/SDP4 over the same day.

GP elements are SGP4 mean elements in the TEME frame. SGP4 adds what the
two-body model leaves out: Earth's oblateness (J2-J4), atmospheric drag (B*)
and, for orbits above ~225 min period, lunar and solar gravity (SDP4). Our
Kepler positions are computed in the frame the elements define, so both are
TEME positions and can be subtracted directly.

The difference is split into radial, in-track and cross-track components
relative to the SGP4 orbit, which shows where the two models drift apart.
"""

from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sgp4 import omm
from sgp4.api import SGP4_ERRORS, Satrec, jday

from orbit_explorer import style
from orbit_explorer.elements import load_elements
from orbit_explorer.kepler import SPAN_S, START, propagate_eci, time_grid_s

OUTPUT_DIR = Path("outputs")
REPORT_HOURS = (0, 6, 12, 24)


def satrec(sat: dict) -> Satrec:
    """SGP4 satellite record straight from a GP (OMM) JSON record."""
    rec = Satrec()
    omm.initialize(rec, sat)
    return rec


def propagate_sgp4(
    sat: dict, t_s: np.ndarray, start: datetime = START
) -> tuple[np.ndarray, np.ndarray]:
    """TEME positions [km] and velocities [km/s] at times t_s after ``start``
    (midnight UTC; by default the window start of the current GP data)."""
    jd0, fr0 = jday(start.year, start.month, start.day, 0, 0, 0)
    err, r, v = satrec(sat).sgp4_array(np.full_like(t_s, jd0), fr0 + t_s / 86400)
    if np.any(err):
        code = int(err[np.nonzero(err)[0][0]])
        raise RuntimeError(f"SGP4 failed for {sat['OBJECT_NAME']}: {SGP4_ERRORS[code]}")
    return r, v


def ric_components(
    diff: np.ndarray, r_ref: np.ndarray, v_ref: np.ndarray
) -> np.ndarray:
    """Split position differences into radial, in-track and cross-track [km].

    Radial points away from the Earth, cross-track along the orbit normal
    (r x v), and in-track completes the right-handed frame (close to the
    direction of motion).
    """
    radial = r_ref / np.linalg.norm(r_ref, axis=1, keepdims=True)
    normal = np.cross(r_ref, v_ref)
    cross = normal / np.linalg.norm(normal, axis=1, keepdims=True)
    in_track = np.cross(cross, radial)
    return np.column_stack(
        [
            np.sum(diff * radial, axis=1),
            np.sum(diff * in_track, axis=1),
            np.sum(diff * cross, axis=1),
        ]
    )


def compare(sat: dict, t_s: np.ndarray, start: datetime = START) -> dict:
    """Kepler minus SGP4 at times t_s after ``start``: total distance and RIC
    components [km]."""
    r_sgp4, v_sgp4 = propagate_sgp4(sat, t_s, start)
    r_kepler, _ = propagate_eci(sat, t_s, start)
    diff = r_kepler - r_sgp4
    return {
        "total": np.linalg.norm(diff, axis=1),
        "ric": ric_components(diff, r_sgp4, v_sgp4),
    }


def epoch_age_days(sat: dict, start: datetime = START) -> float:
    """Days between the GP epoch and ``start`` (positive if the epoch is
    earlier); by default the window START."""
    epoch = datetime.fromisoformat(sat["EPOCH"]).replace(tzinfo=UTC)
    return (start - epoch).total_seconds() / 86400


def build_summary() -> str:
    t = time_grid_s()
    idx = [int(h * 3600 / (t[1] - t[0])) for h in REPORT_HOURS]
    hours = "".join(f"{f'{h:02d}h':>9s}" for h in REPORT_HOURS)
    window = f"{START:%Y-%m-%d %H:%M} UTC + {SPAN_S // 3600} h"
    lines = [
        "KEPLER vs SGP4",
        f"Window: {window}, step {t[1] - t[0]:.0f} s",
        "",
        "Position difference |r_kepler - r_sgp4| [km]",
        f"{'Satellite':24s} {'Epoch age[d]':>12s}{hours}{'max':>9s}",
    ]
    peaks = []
    for sat in load_elements():
        c = compare(sat, t)
        at = "".join(f"{c['total'][i]:9.1f}" for i in idx)
        lines.append(
            f"{sat['OBJECT_NAME']:24s} {epoch_age_days(sat):12.2f}{at}"
            f"{c['total'].max():9.1f}"
        )
        peaks.append((sat["OBJECT_NAME"], np.abs(c["ric"]).max(axis=0)))

    lines += [
        "",
        "Largest component of the difference [km]",
        f"{'Satellite':24s} {'radial':>9s} {'in-track':>9s} {'cross':>9s}",
    ]
    lines += [f"{name:24s} {r:9.1f} {i:9.1f} {c:9.1f}" for name, (r, i, c) in peaks]
    lines += [
        "",
        "Epoch age > 0: the GP epoch is before the window, so both models",
        "propagate that many days before reaching 00h.",
        "",
    ]
    return "\n".join(lines)


COMPONENTS = [
    ("Radial", style.RADIAL),
    ("In-track", style.IN_TRACK),
    ("Cross-track", style.CROSS_TRACK),
]


@style.styled
def plot_differences(path: Path) -> None:
    t = time_grid_s()
    h = t / 3600
    sats = load_elements()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.subplots_adjust(hspace=0.35, wspace=0.18)
    for ax, sat in zip(axes.flat, sats):
        c = compare(sat, t)
        ax.axhline(0, color=style.SEPARATOR, linewidth=1, zorder=1)
        for k, (_, color) in enumerate(COMPONENTS):
            ax.plot(h, c["ric"][:, k], color=color, linewidth=1.5, zorder=2)
        ax.plot(h, c["total"], color=style.LABEL, linewidth=2.2, zorder=3)
        ax.set_title(style.short_name(sat))
        hours_before = epoch_age_days(sat) * 24
        when = "before" if hours_before >= 0 else "after"
        ax.annotate(
            f"GP epoch {abs(hours_before):.1f} h {when} 00:00",
            xy=(1, 1),
            xycoords="axes fraction",
            xytext=(0, 10),
            textcoords="offset points",
            ha="right",
            color=style.SECONDARY,
            fontsize=10,
        )
        ax.yaxis.tick_right()
        ax.yaxis.set_major_formatter("{x:.0f} km")
        style.time_axis(ax, SPAN_S / 3600, step_h=6)
    style.header(
        axes[0, 0],
        "Kepler vs SGP4",
        f"Two-body minus SGP4/SDP4 position, {style.date_label(START)}, UTC",
        caption="Model comparison",
        pad=36,  # above the panel title
    )
    style.dot_legend(
        axes[1, 0],
        ["Total"] + [name for name, _ in COMPONENTS],
        [style.LABEL] + [color for _, color in COMPONENTS],
    )
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary = build_summary()
    print(summary)
    (OUTPUT_DIR / "sgp4_comparison.txt").write_text(summary)
    plot_differences(OUTPUT_DIR / "sgp4_differences.png")
    print("Written outputs/sgp4_comparison.txt and sgp4_differences.png")
