"""Satellite visibility from a ground station: look angles, passes and a sky plot.

Positions come from SGP4 (the more accurate model); the summary also shows how
much the two-body Kepler model shifts each rise and set. Both are rotated to
ECEF with the same GMST angle and then into the station's local East-North-Up
frame, which gives azimuth (clockwise from north), elevation and range.

    python -m orbit_explorer.visibility                      # Belgrade
    python -m orbit_explorer.visibility --name Tokyo --lat 35.68 --lon 139.77
"""

import argparse
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from orbit_explorer.elements import load_elements
from orbit_explorer.frames import eci_to_ecef, geodetic_to_ecef, gmst_rad
from orbit_explorer.kepler import SPAN_S, START, propagate_eci, time_grid_s
from orbit_explorer.sgp4_compare import propagate_sgp4

OUTPUT_DIR = Path("outputs")
ELEVATION_MASK_DEG = 10.0  # common GNSS cut-off angle
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


@dataclass(frozen=True)
class Station:
    name: str
    lat_deg: float
    lon_deg: float
    height_km: float = 0.0


BELGRADE = Station("Belgrade", 44.8178, 20.4569, 0.117)


@dataclass
class Pass:
    rise_s: float  # seconds after START; window start if already up
    max_s: float
    set_s: float  # window end if still up
    max_el_deg: float
    max_az_deg: float
    rises_in_window: bool
    sets_in_window: bool


def enu_rotation(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Rows are the East, North and Up unit vectors in ECEF at the station."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.array(
        [
            [-np.sin(lon), np.cos(lon), 0.0],
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        ]
    )


def look_angles(
    pos_ecef: np.ndarray, station: Station
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Azimuth [deg, 0..360 from north], elevation [deg] and range [km]."""
    site = geodetic_to_ecef(station.lat_deg, station.lon_deg, station.height_km)[0]
    enu = (pos_ecef - site) @ enu_rotation(station.lat_deg, station.lon_deg).T
    east, north, up = enu[:, 0], enu[:, 1], enu[:, 2]
    rng = np.linalg.norm(enu, axis=1)
    az = np.degrees(np.arctan2(east, north)) % 360
    el = np.degrees(np.arcsin(up / rng))
    return az, el, rng


def satellite_look_angles(
    sat: dict, t_s: np.ndarray, station: Station, model: str = "sgp4"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if model == "sgp4":
        pos, _ = propagate_sgp4(sat, t_s)
    elif model == "kepler":
        pos, _ = propagate_eci(sat, t_s)
    else:
        raise ValueError(f"Unknown model {model!r}")
    return look_angles(eci_to_ecef(pos, gmst_rad(t_s)), station)


def _crossing(t: np.ndarray, el: np.ndarray, i: int, mask: float) -> float:
    """Time where elevation crosses the mask between samples i and i+1."""
    frac = (mask - el[i]) / (el[i + 1] - el[i])
    return float(t[i] + frac * (t[i + 1] - t[i]))


def find_passes(
    t: np.ndarray,
    el: np.ndarray,
    az: np.ndarray,
    mask: float = ELEVATION_MASK_DEG,
) -> list[Pass]:
    """Intervals with elevation above the mask, rise/set interpolated."""
    above = el > mask
    edges = np.diff(above.astype(int))
    starts = list(np.nonzero(edges == 1)[0])  # crossing between i and i+1
    ends = list(np.nonzero(edges == -1)[0])
    if above[0]:
        starts.insert(0, None)
    if above[-1]:
        ends.append(None)

    passes = []
    for s, e in zip(starts, ends):
        lo = 0 if s is None else s + 1
        hi = len(t) - 1 if e is None else e
        i_max = lo + int(np.argmax(el[lo : hi + 1]))
        passes.append(
            Pass(
                rise_s=float(t[0]) if s is None else _crossing(t, el, s, mask),
                max_s=float(t[i_max]),
                set_s=float(t[-1]) if e is None else _crossing(t, el, e, mask),
                max_el_deg=float(el[i_max]),
                max_az_deg=float(az[i_max]),
                rises_in_window=s is not None,
                sets_in_window=e is not None,
            )
        )
    return passes


def clock(seconds: float) -> str:
    if seconds >= SPAN_S:  # end of the window, not 00:00 of the same day
        return f"{SPAN_S // 3600:02d}:00:00"
    return f"{START + timedelta(seconds=seconds):%H:%M:%S}"


def build_summary(station: Station = BELGRADE) -> str:
    t = time_grid_s()
    where = f"{station.lat_deg:.4f}° N, {station.lon_deg:.4f}° E"
    window = f"{START:%Y-%m-%d %H:%M} UTC + {SPAN_S // 3600} h"
    mask = f"elevation mask {ELEVATION_MASK_DEG:.0f}°"
    lines = [
        f"VISIBILITY FROM {station.name.upper()}",
        f"Station: {where}, {station.height_km * 1000:.0f} m",
        f"Window: {window}, {mask}, positions from SGP4",
        "",
    ]
    for sat in load_elements():
        az, el, _ = satellite_look_angles(sat, t, station)
        passes = find_passes(t, el, az)
        k_az, k_el, _ = satellite_look_angles(sat, t, station, "kepler")
        kepler = find_passes(t, k_el, k_az)
        lines.append(
            f"{sat['OBJECT_NAME']}: {len(passes)} pass(es), "
            f"elevation {el.min():.1f}° to {el.max():.1f}°"
        )
        if not passes:
            lines += ["  never above the mask", ""]
            continue
        lines.append(
            f"  {'rise':>9s} {'max el':>9s} {'set':>9s} {'duration':>9s}"
            f" {'max el':>7s} {'az':>6s}   Kepler shift (rise/set)"
        )
        for p in passes:
            rise = clock(p.rise_s) if p.rises_in_window else "  (up)  "
            set_ = clock(p.set_s) if p.sets_in_window else "  (up)  "
            minutes = (p.set_s - p.rise_s) / 60
            lines.append(
                f"  {rise:>9s} {clock(p.max_s):>9s} {set_:>9s} {minutes:7.1f} m"
                f" {p.max_el_deg:6.1f}° {p.max_az_deg:5.0f}°   "
                + kepler_shift(p, kepler)
            )
        lines.append("")
    lines += [
        "(up): already above the mask at the start, or still above it at the end.",
        "Kepler shift: Kepler time minus SGP4 time for the same pass, in seconds.",
        "",
    ]
    return "\n".join(lines)


def kepler_shift(p: Pass, kepler: list[Pass]) -> str:
    """Rise/set offsets of the overlapping Kepler pass, or a note if none."""
    match = [k for k in kepler if k.rise_s <= p.set_s and k.set_s >= p.rise_s]
    if not match:
        return "no Kepler pass"
    k = min(match, key=lambda k: abs(k.max_s - p.max_s))
    parts = [
        f"{k.rise_s - p.rise_s:+.0f} s"
        if p.rises_in_window and k.rises_in_window
        else "–",
        f"{k.set_s - p.set_s:+.0f} s" if p.sets_in_window and k.sets_in_window else "–",
    ]
    return " / ".join(parts)


def plot_elevation(path: Path, station: Station = BELGRADE) -> None:
    t = time_grid_s()
    fig, ax = plt.subplots(figsize=(12, 5))
    for sat, color in zip(load_elements(), COLORS):
        _, el, _ = satellite_look_angles(sat, t, station)
        ax.plot(t / 3600, el, color=color, linewidth=1.2, label=sat["OBJECT_NAME"])
    ax.axhspan(-90, ELEVATION_MASK_DEG, color="0.85", zorder=0)
    ax.axhline(ELEVATION_MASK_DEG, color="0.5", linewidth=0.8, linestyle="--")
    ax.text(
        0.2,
        ELEVATION_MASK_DEG + 1.5,
        f"{ELEVATION_MASK_DEG:.0f}° mask",
        color="0.4",
        fontsize=9,
    )
    ax.set(
        xlim=(0, SPAN_S / 3600),
        ylim=(-90, 90),
        xlabel=f"Hours since {START:%Y-%m-%d %H:%M} UTC",
        ylabel="Elevation [deg]",
        title=f"Elevation seen from {station.name}",
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_skyplot(path: Path, station: Station = BELGRADE) -> None:
    """Polar sky view: north up, azimuth clockwise, zenith at the centre."""
    t = time_grid_s()
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    for sat, color in zip(load_elements(), COLORS):
        az, el, _ = satellite_look_angles(sat, t, station)
        r = np.where(el >= 0, 90 - el, np.nan)  # hide below the horizon
        theta = np.radians(az)
        jumps = np.nonzero(np.abs(np.diff(r)) > 5)[0] + 1  # don't join separate passes
        ax.plot(
            np.insert(theta, jumps, np.nan),
            np.insert(r, jumps, np.nan),
            color=color,
            linewidth=1.2,
            label=sat["OBJECT_NAME"],
        )
        visible = el >= 0
        if visible.any() and np.ptp(az[visible]) < 1 and np.ptp(el[visible]) < 1:
            # geostationary: the track is a single point, draw it as a dot
            ax.scatter(
                np.radians(az[visible].mean()),
                90 - el[visible].mean(),
                s=60,
                color=color,
                edgecolor="black",
                zorder=3,
            )
    ax.fill_between(
        np.linspace(0, 2 * np.pi, 200),
        90 - ELEVATION_MASK_DEG,
        90,
        color="0.85",
        zorder=0,
    )
    ax.set_rlim(0, 90)
    ax.set_rticks([0, 30, 60, 90 - ELEVATION_MASK_DEG])
    ax.set_yticklabels(["90°", "60°", "30°", f"{ELEVATION_MASK_DEG:.0f}°"], fontsize=8)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"])
    ax.set_title(f"Sky plot from {station.name}, {START:%Y-%m-%d} (24 h)", pad=16)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.95, 1.08))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_station(argv: list[str] | None = None) -> Station:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default=BELGRADE.name)
    parser.add_argument("--lat", type=float, default=BELGRADE.lat_deg, help="deg N")
    parser.add_argument("--lon", type=float, default=BELGRADE.lon_deg, help="deg E")
    parser.add_argument(
        "--height",
        type=float,
        default=BELGRADE.height_km * 1000,
        help="metres above the WGS84 ellipsoid",
    )
    args = parser.parse_args(argv)
    return Station(args.name, args.lat, args.lon, args.height / 1000)


if __name__ == "__main__":
    station = parse_station()
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary = build_summary(station)
    print(summary)
    (OUTPUT_DIR / "visibility.txt").write_text(summary)
    plot_elevation(OUTPUT_DIR / "visibility_elevation.png", station)
    plot_skyplot(OUTPUT_DIR / "visibility_skyplot.png", station)
    print(
        "Written outputs/visibility.txt, visibility_elevation.png "
        "and visibility_skyplot.png"
    )
