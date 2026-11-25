"""Orbit determination: fit the force model to the precise orbit, then predict.

validation.py starts the propagator on the precise orbit and lets it run: its
error is the force model's. Here the model also learns from the data, as an
operations centre does. A batch least-squares fit estimates the satellite's
state at 00:00 and its solar radiation pressure from the precise positions
of the first 12 hours, and the fitted orbit is then *predicted* over the next
12 hours, which the fit never saw. The prediction error is the honest score.

The force model, in TEME:

- **gravity:** EGM2008 to degree and order 8 (gravity.py), evaluated in the
  Earth-fixed frame (GMST at UT1 and polar motion, as validation.py);
- **the Moon and the Sun** as point masses (ephemeris.py);
- **solar radiation pressure** (srp.py): none, cannonball (1 parameter) or
  DYB (3).

The fit is Gauss-Newton on the position residuals, every 5 minutes. The
partial derivatives come from central differences: all the perturbed
trajectories go through the integrator together, as one batch. The formal
uncertainties assume independent residuals of equal size; they are not, so
read them as a lower bound.

    python -m orbit_explorer.od
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from orbit_explorer import ephemeris, gravity, j2, srp, style
from orbit_explorer.sgp4_compare import ric_components
from orbit_explorer.sp3 import PRECISE_ORBITS, SP3_IDS, read_sp3
from orbit_explorer.validation import (
    GALILEO_AREA_TO_MASS_M2_KG,
    PRECISE_EOP,
    PRECISE_GP,
    PRECISE_START,
    EarthOrientation,
    _rot3,
    gmst_ut1,
    itrf_to_teme,
    polar_motion,
    precise_state,
)

OUTPUT_DIR = Path("outputs")
OBS_STEP_S = 300.0  # the SP3 file's own step
SUBSTEPS = 5  # RK4 steps of 60 s between observations
FIT_HOURS = 12
MAX_ITERATIONS = 10
# central difference steps: position [km], velocity [km/s], SRP [1e-9 km/s²]
DIFF_STEPS = (1e-3, 1e-6, 1e-2)
CONVERGED_KM = 1e-6  # a millimetre: the position correction that ends the fit
SRP_MODELS = ("none", "cannonball", "dyb")
SRP_LABELS = {"none": "no SRP", "cannonball": "cannonball SRP", "dyb": "DYB SRP"}


# ---------- Propagation with the full force model ----------
def propagate(
    x0: np.ndarray,
    t_s: np.ndarray,
    srp_params: np.ndarray | None = None,
    srp_model: str = "none",
    field: gravity.GravityField | None = None,
    start: datetime = PRECISE_START,
    eop: EarthOrientation = PRECISE_EOP,
    substeps: int = SUBSTEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Positions and velocities (len(t_s), N, 3) of N orbits from their TEME
    states x0 (N, 6) at t_s[0], seconds after ``start`` (UTC), each with its
    own SRP parameters (N, srp.MODELS[srp_model])."""
    x0 = np.atleast_2d(x0)
    field = field or gravity.load_field()
    params = np.zeros((len(x0), 0)) if srp_params is None else np.atleast_2d(srp_params)
    half = j2.half_step_times(t_s, substeps)
    moon = ephemeris.moon_position(half, start)
    sun = ephemeris.sun_position(half, start)
    theta = gmst_ut1(half, start, eop)
    w = polar_motion(eop)

    def acc(r, i):
        angle = np.full(len(r), theta[i])
        r_itrf = _rot3(angle, r) @ w  # as validation.teme_to_itrf
        a = _rot3(-angle, gravity.acceleration(r_itrf, field) @ w.T)
        a = a + ephemeris.third_body_acceleration(r, moon[i], ephemeris.MU_MOON_KM3_S2)
        a = a + ephemeris.third_body_acceleration(r, sun[i], ephemeris.MU_SUN_KM3_S2)
        return a + srp.acceleration(r, sun[i], params, srp_model)

    return j2.rk4(x0[:, :3], x0[:, 3:], t_s, acc, substeps)


# ---------- Batch least squares ----------
@dataclass(frozen=True)
class Fit:
    """A converged fit: the state at t_s[0] (6,), the SRP parameters, their
    formal standard deviations, the post-fit RMS [km] and the iterations."""

    srp_model: str
    state: np.ndarray
    srp_params: np.ndarray
    sigma: np.ndarray
    rms_km: float
    iterations: int


def fit(
    t_s: np.ndarray,
    r_obs: np.ndarray,
    x_guess: np.ndarray,
    srp_model: str = "none",
    field: gravity.GravityField | None = None,
    start: datetime = PRECISE_START,
    substeps: int = SUBSTEPS,
) -> Fit:
    """Fit the state at t_s[0] and the SRP parameters to observed TEME
    positions r_obs (len(t_s), 3), starting from x_guess (6,)."""
    field = field or gravity.load_field()
    k = srp.MODELS[srp_model]
    x = np.concatenate([x_guess, np.zeros(k)])
    steps = np.repeat(DIFF_STEPS, [3, 3, k])
    n = len(x)
    for iteration in range(1, MAX_ITERATIONS + 1):
        batch = np.tile(x, (2 * n + 1, 1))  # nominal, then +step and -step
        for j in range(n):
            batch[1 + 2 * j, j] += steps[j]
            batch[2 + 2 * j, j] -= steps[j]
        r, _ = propagate(
            batch[:, :6], t_s, batch[:, 6:], srp_model, field, start, substeps=substeps
        )
        residual = (r_obs - r[:, 0]).ravel()
        H = np.stack(
            [
                ((r[:, 1 + 2 * j] - r[:, 2 + 2 * j]) / (2 * steps[j])).ravel()
                for j in range(n)
            ],
            axis=1,
        )
        dx, *_ = np.linalg.lstsq(H, residual, rcond=None)
        x = x + dx
        if np.abs(dx[:3]).max() < CONVERGED_KM:
            break
    else:
        raise RuntimeError(f"the {srp_model} fit did not converge")
    variance = residual @ residual / (len(residual) - n)
    covariance = variance * np.linalg.inv(H.T @ H)
    return Fit(
        srp_model=srp_model,
        state=x[:6],
        srp_params=x[6:],
        sigma=np.sqrt(np.diag(covariance)),
        rms_km=float(np.sqrt(np.mean(residual**2) * 3)),  # 3D, per epoch
        iterations=iteration,
    )


# ---------- The analysis ----------
RUNS = (
    ("j2", "J2 + Moon + Sun from the precise state"),
    ("egm", "EGM2008 8×8 + Moon + Sun from the precise state"),
    *((f"fit_{m}", f"Fitted to 00-12 h, {SRP_LABELS[m]}") for m in SRP_MODELS),
)

README_ROWS = dict(RUNS)  # the README's table: these labels, prediction RMS


def time_grid() -> np.ndarray:
    """Every 5 minutes over the part of the day the SP3 file covers."""
    sp3 = read_sp3(PRECISE_ORBITS)
    end = sp3.seconds_since(PRECISE_START)[-1] - 1
    return np.arange(0.0, end, OBS_STEP_S)


def observed(sp3, sat_id: str, t_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The precise orbit in TEME: positions and inertial velocities."""
    return itrf_to_teme(*precise_state(sp3, sat_id, t_s), t_s)


def analyse(t_s: np.ndarray | None = None, fit_hours: float = FIT_HOURS) -> dict:
    """For each satellite with an SP3 orbit: the runs of RUNS against the
    precise orbit (distance and radial/in-track/cross-track [km]) and the
    fits."""
    t = time_grid() if t_s is None else t_s
    in_fit = t <= fit_hours * 3600
    sp3 = read_sp3(PRECISE_ORBITS)
    field = gravity.load_field()
    from orbit_explorer.elements import load_elements

    names = {s["NORAD_CAT_ID"]: s for s in load_elements(PRECISE_GP)}
    out = {}
    for norad, sat_id in SP3_IDS.items():
        r_true, v_true = observed(sp3, sat_id, t)
        x_true = np.concatenate([r_true[0], v_true[0]])
        positions = {
            "j2": j2.propagate(r_true[0], v_true[0], t, third_body_start=PRECISE_START)[
                0
            ],
            "egm": propagate(x_true, t, field=field)[0][:, 0],
        }
        fits = {}
        for model in SRP_MODELS:
            f = fit(t[in_fit], r_true[in_fit], x_true, model, field)
            fits[model] = f
            positions[f"fit_{model}"] = propagate(
                f.state, t, f.srp_params, model, field
            )[0][:, 0]
        res = {"sat": names[norad], "sat_id": sat_id, "t": t, "in_fit": in_fit}
        for key, r in positions.items():
            diff = r - r_true
            res[key] = {
                "total": np.linalg.norm(diff, axis=1),
                "ric": ric_components(diff, r_true, v_true),
            }
        res["fits"] = fits
        out[norad] = res
    return out


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def galileo_srp_estimate() -> float:
    """The SRP size validation.py estimated from public area and mass, in
    srp.UNIT_KM_S2 (cannonball, C_R 1.3)."""
    return ephemeris.srp_acceleration_km_s2(GALILEO_AREA_TO_MASS_M2_KG) / srp.UNIT_KM_S2


def build_summary(results: dict) -> str:
    first = next(iter(results.values()))
    t = first["t"]
    lines = [
        "ORBIT DETERMINATION: FIT 00-12 h, PREDICT 12-24 h",
        f"Day: {PRECISE_START:%Y-%m-%d} UTC, truth: {PRECISE_ORBITS.name}",
        (
            f"Observations: SP3 positions every {OBS_STEP_S / 60:.0f} min; fit "
            f"{first['in_fit'].sum()}, predicted {(~first['in_fit']).sum()}; "
            f"RK4 step {OBS_STEP_S / SUBSTEPS:.0f} s"
        ),
        "Force model: EGM2008 8x8, Moon and Sun as point masses, SRP as fitted",
        "",
        "Distance from the precise orbit [m]",
        f"{'Satellite':24s} {'run':48s}{'fit RMS':>9s}{'pred RMS':>10s}{'pred max':>10s}",
    ]
    for res in results.values():
        fit_part, pred = res["in_fit"], ~res["in_fit"]
        for k, (key, label) in enumerate(RUNS):
            total = res[key]["total"] * 1000
            lines.append(
                f"{res['sat']['OBJECT_NAME'] if k == 0 else '':24s} {label:48s}"
                f"{rms(total[fit_part]):9.1f}{rms(total[pred]):10.1f}"
                f"{total[pred].max():10.1f}"
            )
    lines += [
        "",
        "Fitted SRP [1e-9 km/s² at 1 AU, ± formal sigma] and post-fit RMS",
        f"{'Satellite':24s} {'model':16s} {'D':>14s} {'Y':>14s} {'B':>14s} {'RMS':>8s}",
    ]
    for res in results.values():
        for k, model in enumerate(SRP_MODELS):
            f = res["fits"][model]
            cells = [
                f"{p:+.3f} ± {s:.3f}"
                for p, s in zip(f.srp_params, f.sigma[6:], strict=True)
            ]
            cells += [""] * (3 - len(cells))
            lines.append(
                f"{res['sat']['OBJECT_NAME'] if k == 0 else '':24s} {SRP_LABELS[model]:16s} "
                + " ".join(f"{c:>14s}" for c in cells)
                + f" {f.rms_km * 1000:6.1f} m"
            )
    lines += [
        "",
        "D: away from the Sun, Y: along the solar panel axis, B: completes the frame.",
        (
            f"Expected for a Galileo satellite from public area and mass: about "
            f"{galileo_srp_estimate():.2f} (A/m 0.02 m²/kg, C_R 1.3)."
        ),
        (
            f"Predicted: {t[~first['in_fit']][0] / 3600:.2f}-{t[-1] / 3600:.2f} h UTC, "
            "never seen by the fit."
        ),
        "",
    ]
    return "\n".join(lines)


# ---------- Figure ----------
RUN_STYLE = {
    "j2": {"color": style.SECONDARY, "lw": 1.6, "linestyle": (0, (4, 3))},
    "egm": {"lw": 1.6, "alpha": 0.5},  # satellite colour, lighter
    "fit_cannonball": {"lw": 2.4},  # satellite colour
}
FIGURE_LABELS = {
    "j2": "J2 + Moon + Sun from the precise state",
    "egm": "EGM2008 8×8 + Moon + Sun from the precise state",
    "fit_cannonball": "Fitted to 00-12 h with SRP, then predicted",
}


@style.styled
def plot_od(path: Path, results: dict) -> None:
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    fig.subplots_adjust(wspace=0.1)
    colors = [style.sat_color(res["sat"]) for res in results.values()]
    for k, (ax, res) in enumerate(zip(axes, results.values(), strict=True)):
        h = res["t"] / 3600
        ax.axvspan(0, FIT_HOURS, color=style.FILL, zorder=0, lw=0)
        ax.text(
            FIT_HOURS / 2,
            0.97,
            "fit",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            color=style.SECONDARY,
            fontsize=10,
        )
        ax.text(
            (FIT_HOURS + 24) / 2,
            0.97,
            "prediction",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            color=style.SECONDARY,
            fontsize=10,
        )
        for key, kw in RUN_STYLE.items():
            ax.plot(h[1:], res[key]["total"][1:], **{"color": colors[k], **kw})
        ax.set_yscale("log")
        ax.minorticks_off()
        ax.grid(True, which="major", axis="y")
        ax.set_title(f"{style.short_name(res['sat'])} ({res['sat_id']})")
        style.time_axis(ax, 24, step_h=6)
    axes[0].set_xticks([0, 6, 12, 18], ["00:00", "06:00", "12:00", "18:00"])
    axes[0].set_ylim(1e-4, 3)
    axes[0].set_yticks(
        [1e-4, 1e-3, 1e-2, 1e-1, 1], ["10 cm", "1 m", "10 m", "100 m", "1 km"]
    )
    axes[0].tick_params(axis="y", left=False, labelleft=False)
    axes[1].tick_params(
        axis="y", left=False, labelleft=False, right=True, labelright=True
    )
    style.header(
        axes[0],
        "Fitted to the first half, predicted over the second",
        f"Distance from GFZ's SP3 orbit · {style.date_label(PRECISE_START)} · UTC",
        caption="Orbit determination",
        pad=36,
    )

    def line(key, color):
        return Line2D([], [], **{"color": color, **RUN_STYLE[key]})

    handles = [
        line(key, None)
        if "color" in RUN_STYLE[key]
        else (line(key, colors[0]), line(key, colors[1]))
        for key in RUN_STYLE
    ]
    style.dot_legend(
        axes[0],
        [],
        [],
        extra=list(zip(handles, FIGURE_LABELS.values(), strict=True)),
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
        ncol=2,
        handlelength=3.2,
        handletextpad=0.6,
    )
    fig.savefig(path)
    plt.close(fig)


def main(output_dir: Path = OUTPUT_DIR) -> None:
    results = analyse()
    summary = build_summary(results)
    print(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "od.txt").write_text(summary)
    plot_od(output_dir / "od.png", results)
    print(f"Written {output_dir}/od.txt and od.png")


if __name__ == "__main__":
    main()
