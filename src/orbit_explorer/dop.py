"""Dilution of precision (DOP): how the satellites' geometry in the sky scales
ranging errors into position errors, for GPS, Galileo and both together.

A receiver solves for its position and clock offset from pseudoranges. With
unit line-of-sight vectors u_i to the satellites in view (East, North, Up),
each pseudorange gives one row of the geometry matrix G = [-u_i, 1], and

    Q = (G^T G)^-1        (in units of the ranging error variance)
    PDOP = sqrt(Q_EE + Q_NN + Q_UU)   HDOP = sqrt(Q_EE + Q_NN)
    VDOP = sqrt(Q_UU)                 GDOP = sqrt(trace Q)

so a 1 m ranging error becomes about PDOP metres of 3D position error.
Satellites spread over the sky give a small DOP; few, or bunched, a large one.

GPS and Galileo keep separate system times, so the two together have one
clock unknown per system (a column [1, 0] for GPS rows, [0, 1] for Galileo
rows), the way multi-GNSS receivers solve for the inter-system offset.

Positions come from SGP4 on the whole constellations in data/inputs/gnss/
(orbit_explorer.gnss), over the snapshot's UTC day, with the same look angles
and 10° elevation mask as orbit_explorer.visibility.

    python -m orbit_explorer.dop                     # Belgrade + world map
    python -m orbit_explorer.dop --station tokyo     # or sydney, quito
    python -m orbit_explorer.dop --name "Novi Sad" --lat 45.25 --lon 19.84

Belgrade writes outputs/dop.txt and dop.png, the other stations add their
name (dop_tokyo.txt), and every run writes dop_world.png.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from orbit_explorer import style
from orbit_explorer.frames import eci_to_ecef, geodetic_to_ecef, gmst_rad
from orbit_explorer.gnss import GNSS_DIR, load_snapshot, usable
from orbit_explorer.plots import load_world_map
from orbit_explorer.sgp4_compare import propagate_sgp4
from orbit_explorer.visibility import (
    BELGRADE,
    ELEVATION_MASK_DEG,
    STATIONS_BY_SLUG,
    Station,
    enu_rotation,
    parse_station,
)

OUTPUT_DIR = Path("outputs")
SPAN_S = 86400
STEP_S = 60  # DOP changes over minutes: GNSS satellites cross the sky in hours
MAP_STEP_S = 900  # the world map: every 15 minutes of the day
MAP_GRID_DEG = 3.0
SIDEREAL_DAY_S = 86164.0905
# GPS orbits twice a sidereal day, so its sky repeats after about one; J2
# turns the orbit planes west, and the ground track repeats some 9 s sooner
# (23 h 55 m 55 s in this snapshot), which tests/test_dop.py measures.
BOTH = "GPS + Galileo"
# Line colours: system hues not used for the four satellites' own colours,
# except Galileo's orange (GALILEO 29 is a Galileo satellite); both together
# in the label ink, darkest, as the result the other two add up to.
COLORS = {"GPS": "#5856D6", "Galileo": "#FF9500", BOTH: style.LABEL}
PDOP_GOOD = 2.0  # a common threshold for good geometry


def time_grid_s(step_s: float = STEP_S) -> np.ndarray:
    """The day's epochs, 00:00 included and 24:00 not (it is the next day's
    00:00), so every moment of the day counts once."""
    return np.arange(0, SPAN_S, step_s, dtype=float)


def ecef_positions(sats: list[dict], t_s: np.ndarray, start: datetime) -> np.ndarray:
    """Earth-fixed positions [km] of each satellite: (satellite, time, xyz)."""
    theta = gmst_rad(t_s, start)
    return np.stack(
        [eci_to_ecef(propagate_sgp4(sat, t_s, start)[0], theta) for sat in sats]
    )


def lines_of_sight(
    pos_ecef: np.ndarray, lat_deg: float, lon_deg: float, height_km: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Unit line-of-sight vectors in East-North-Up (..., 3) and elevation [deg]
    from a site to positions (..., 3), as visibility.look_angles computes them."""
    site = geodetic_to_ecef(lat_deg, lon_deg, height_km)[0]
    enu = (pos_ecef - site) @ enu_rotation(lat_deg, lon_deg).T
    u = enu / np.linalg.norm(enu, axis=-1, keepdims=True)
    return u, np.degrees(np.arcsin(u[..., 2]))


def unit_vectors(az_deg, el_deg) -> np.ndarray:
    """East-North-Up unit vectors from azimuth (clockwise from north) and
    elevation, for geometries given as angles (e.g. in tests)."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.stack(
        [np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], axis=-1
    )


@dataclass
class DOP:
    """DOP values and the number of satellites used, one per epoch; NaN
    where too few satellites are in view to solve."""

    gdop: np.ndarray
    pdop: np.ndarray
    hdop: np.ndarray
    vdop: np.ndarray
    n: np.ndarray


def dop(u: np.ndarray, visible: np.ndarray, system: np.ndarray | None = None) -> DOP:
    """DOP from unit lines of sight u (satellite, epoch, 3) of the satellites
    ``visible`` (satellite, epoch) at each epoch.

    ``system`` (satellite,) numbers each satellite's system 0, 1, ...: one
    clock unknown per system. A system with no satellite in view at an epoch
    drops out (its clock can't be estimated, and isn't needed). An epoch needs
    as many satellites as unknowns: 4 for one system, 5 for two.
    """
    n_sat, n_t, _ = u.shape
    system = np.zeros(n_sat, dtype=int) if system is None else np.asarray(system)
    k = int(system.max()) + 1
    clocks = np.eye(k)[system]  # (satellite, k): which clock each row uses
    g = np.concatenate([-u, np.broadcast_to(clocks[:, None, :], (n_sat, n_t, k))], -1)
    w = visible.astype(float)
    rows = g.transpose(1, 0, 2)  # (epoch, satellite, unknown)
    normal = (rows * w.T[..., None]).transpose(0, 2, 1) @ rows  # G^T G per epoch
    in_view = np.einsum("st,sk->tk", w, clocks) > 0  # (epoch, system)
    # An unseen system's clock row and column are all zero: a 1 on the
    # diagonal makes the matrix invertible without touching anything else.
    for j in range(k):
        normal[:, 3 + j, 3 + j] += ~in_view[:, j]
    n = w.sum(axis=0)
    solvable = n >= 3 + in_view.sum(axis=1)
    # A geometry with too little spread (e.g. all satellites in one plane)
    # is as unsolvable as too few satellites.
    eig = np.linalg.eigvalsh(normal)  # ascending; G^T G is symmetric
    with np.errstate(divide="ignore", invalid="ignore"):
        solvable &= eig[:, -1] < 1e10 * eig[:, 0]
    q = np.full_like(normal, np.nan)
    q[solvable] = np.linalg.inv(normal[solvable])
    diag = np.diagonal(q, axis1=1, axis2=2)
    trace = diag.sum(axis=1) - (~in_view).sum(axis=1)  # without the 1s added
    return DOP(
        gdop=np.sqrt(trace),
        pdop=np.sqrt(diag[:, :3].sum(axis=1)),
        hdop=np.sqrt(diag[:, :2].sum(axis=1)),
        vdop=np.sqrt(diag[:, 2]),
        n=n,
    )


def combinations(
    positions: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each system alone and all together: (positions, system index)."""
    out = {
        name: (pos, np.zeros(len(pos), dtype=int)) for name, pos in positions.items()
    }
    out[BOTH] = (
        np.concatenate(list(positions.values())),
        np.concatenate(
            [np.full(len(pos), j) for j, pos in enumerate(positions.values())]
        ),
    )
    return out


def station_dop(
    positions: dict[str, np.ndarray],
    station: Station,
    mask_deg: float = ELEVATION_MASK_DEG,
) -> dict[str, DOP]:
    """DOP over time at a station for GPS, Galileo and both."""
    result = {}
    for name, (pos, system) in combinations(positions).items():
        u, el = lines_of_sight(pos, station.lat_deg, station.lon_deg, station.height_km)
        result[name] = dop(u, el >= mask_deg, system)
    return result


def world_pdop(
    positions: dict[str, np.ndarray],
    grid_deg: float = MAP_GRID_DEG,
    mask_deg: float = ELEVATION_MASK_DEG,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Latitudes, longitudes and, per combination, the 95th percentile of
    PDOP over the times in ``positions`` at each grid point (at sea level):
    the value PDOP stays under 95% of the time, as availability is usually
    specified. (The single worst epoch is one unlucky moment.)"""
    lats = np.arange(-90 + grid_deg / 2, 90, grid_deg)
    lons = np.arange(-180 + grid_deg / 2, 180, grid_deg)
    p95 = {}
    for name, (pos, system) in combinations(positions).items():
        n_t = pos.shape[1]
        grid = np.empty((len(lats), len(lons)))
        for i, lat in enumerate(lats):
            # one row of sites at once: (satellite, site, epoch, 3), and the
            # sites' epochs side by side as epochs of one dop() call
            site = geodetic_to_ecef(np.full(len(lons), lat), lons, np.zeros(len(lons)))
            rot = np.stack([enu_rotation(lat, lon) for lon in lons])
            enu = (pos[:, None] - site[None, :, None]) @ rot.transpose(0, 2, 1)
            u = enu / np.linalg.norm(enu, axis=-1, keepdims=True)
            el = np.degrees(np.arcsin(u[..., 2]))
            d = dop(
                u.reshape(len(pos), -1, 3),
                (el >= mask_deg).reshape(len(pos), -1),
                system,
            )
            grid[i] = np.nanpercentile(d.pdop.reshape(len(lons), n_t), 95, axis=1)
        p95[name] = grid
    return lats, lons, p95


def load_positions(
    folder: Path = GNSS_DIR, step_s: float = STEP_S, all_listed: bool = False
) -> tuple[dict[str, np.ndarray], np.ndarray, datetime]:
    """Positions of the usable satellites per system (``all_listed``: of
    every satellite CelesTrak lists, the excluded ones too)."""
    systems, start = load_snapshot(folder) if all_listed else usable(folder)[:2]
    t = time_grid_s(step_s)
    return {name: ecef_positions(s, t, start) for name, s in systems.items()}, t, start


# ---------- Text ----------
def build_summary(
    station: Station,
    result: dict[str, DOP],
    start: datetime,
    counts: dict[str, int],
    excluded: dict[str, list[dict]] | None = None,
    all_listed: dict[str, DOP] | None = None,
) -> str:
    """The table for the usable satellites; ``excluded`` (manifest entries)
    and ``all_listed`` (DOP with them back in) say what leaving them out did."""
    excluded = excluded or {}

    def count(name: str, n: int) -> str:
        out = len(excluded.get(name, []))
        return f"{name} {n} satellites" + (
            f" ({out} set unusable left out)" if out else ""
        )

    lines = [
        f"DILUTION OF PRECISION AT {station.name.upper()}",
        f"Station: {station.coordinates}, {station.height_km * 1000:.0f} m",
        (
            f"Window: {start:%Y-%m-%d %H:%M} UTC + 24 h, every {STEP_S} s, "
            f"elevation mask {ELEVATION_MASK_DEG:.0f}°, positions from SGP4"
        ),
        "Constellations: " + ", ".join(count(n, k) for n, k in counts.items()),
        "",
        (
            f"{'':15s} {'in view':>13s} {'PDOP':>21s} {'HDOP':>6s} {'VDOP':>6s}"
            f" {'PDOP ≤ 2':>9s}"
        ),
        (
            f"{'':15s} {'min  mean max':>13s} {'mean   95%    max':>21s}"
            f" {'mean':>6s} {'mean':>6s} {'of day':>9s}"
        ),
    ]
    for name, d in result.items():
        good = np.mean(d.pdop <= PDOP_GOOD) * 100
        lines.append(
            f"{name:15s} {d.n.min():3.0f} {d.n.mean():5.1f} {d.n.max():3.0f}"
            f"   {np.nanmean(d.pdop):5.2f} {np.nanpercentile(d.pdop, 95):5.2f}"
            f" {np.nanmax(d.pdop):6.2f}   {np.nanmean(d.hdop):5.2f}"
            f" {np.nanmean(d.vdop):5.2f} {good:8.0f}%"
        )
    if all_listed and excluded:
        lines += [
            "",
            "With the satellites set unusable back in, mean PDOP would be "
            + ", ".join(
                f"{name} {np.nanmean(d.pdop):.2f}"
                for name, d in all_listed.items()
                if np.nanmean(d.pdop) != np.nanmean(result[name].pdop)
            )
            + ".",
        ]
    lines += [
        "",
        "PDOP: 3D position error per metre of ranging error (HDOP horizontal,",
        "VDOP vertical). GPS + Galileo solves one clock offset per system.",
        "",
    ]
    return "\n".join(lines)


# ---------- Figures ----------
@style.styled
def plot_station(path: Path, station: Station, result: dict[str, DOP], t, start):
    """Satellites in view and PDOP over the day."""
    h = t / 3600
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(12, 7.6), sharex=True, gridspec_kw={"hspace": 0.35}
    )
    for name, d in result.items():
        top.step(h, d.n, where="post", color=COLORS[name], linewidth=1.8)
        bottom.plot(h, d.pdop, color=COLORS[name], linewidth=1.8)
    top.set_ylim(0, max(d.n.max() for d in result.values()) + 2)
    top.yaxis.tick_right()
    top.set_title("Satellites in view", loc="left", pad=8)
    peak = max(np.nanmax(d.pdop) for d in result.values())
    bottom.set_ylim(0, max(4.0, np.ceil(peak)))
    bottom.axhline(PDOP_GOOD, color=style.SEPARATOR, linewidth=1, linestyle=(0, (3, 3)))
    bottom.yaxis.tick_right()
    bottom.set_title("PDOP (lower is better)", loc="left", pad=8)
    style.time_axis(bottom, SPAN_S / 3600)
    style.header(
        top,
        f"GNSS geometry over {station.name}",
        f"{style.date_label(start)} · UTC · SGP4 · {ELEVATION_MASK_DEG:.0f}° mask"
        f" · dashed: PDOP {PDOP_GOOD:.0f}",
        caption="Dilution of precision",
        pad=34,
    )
    style.dot_legend(bottom, list(result), [COLORS[n] for n in result])
    fig.savefig(path)
    plt.close(fig)


# Sequential: one hue (indigo, GPS's), light for good geometry to dark.
PDOP_CMAP = LinearSegmentedColormap.from_list(
    "pdop", ["#F2F2F7", "#C9C8F2", "#8E8CE6", "#5856D6", "#2C2A80"]
)


@style.styled
def plot_world(path: Path, lats, lons, p95: dict[str, np.ndarray], start):
    """PDOP's 95th percentile over the day everywhere: GPS alone, then GPS +
    Galileo, on one colour scale."""
    shown = ["GPS", BOTH]
    vmax = 3.5
    fig, axes = plt.subplots(2, 1, figsize=(11, 11.4), gridspec_kw={"hspace": 0.32})
    world = load_world_map()
    for ax, name in zip(axes, shown, strict=True):
        image = ax.pcolormesh(
            lons, lats, p95[name], cmap=PDOP_CMAP, vmin=1, vmax=vmax, shading="nearest"
        )
        style.map_axes(ax, world, range(-180, 181, 60), range(-90, 91, 30))
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect("equal")
        grid = p95[name]
        ax.set_title(
            f"{name}: 95th percentile from {np.nanmin(grid):.1f} to "
            f"{np.nanmax(grid):.1f}, median {np.nanmedian(grid):.1f}",
            loc="left",
            pad=8,
        )
    bar = fig.colorbar(
        image, ax=axes, orientation="horizontal", fraction=0.03, pad=0.06, extend="max"
    )
    bar.outline.set_visible(False)
    bar.set_label("PDOP not exceeded 95% of the day", color=style.SECONDARY)
    style.header(
        axes[0],
        "GNSS geometry around the world",
        f"{style.date_label(start)} · every {MAP_STEP_S // 60} min · "
        f"{MAP_GRID_DEG:.0f}° grid · {ELEVATION_MASK_DEG:.0f}° mask · SGP4",
        caption="Dilution of precision",
        pad=34,
    )
    fig.savefig(path)
    plt.close(fig)


def output_paths(station: Station, output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    """Names as orbit_explorer.visibility gives them: Belgrade plain, the
    viewer's other stations by slug, your own marked custom."""
    if station == BELGRADE:
        stem = "dop"
    elif STATIONS_BY_SLUG.get(station.slug) == station:
        stem = f"dop_{station.slug}"
    else:
        stem = f"dop_custom-{station.slug}"
    return {
        "summary": output_dir / f"{stem}.txt",
        "station": output_dir / f"{stem}.png",
        "world": output_dir / "dop_world.png",
    }


def main(
    argv: list[str] | None = None,
    output_dir: Path = OUTPUT_DIR,
    folder: Path = GNSS_DIR,
) -> dict[str, Path]:
    station = parse_station(argv)
    paths = output_paths(station, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    positions, t, start = load_positions(folder)
    result = station_dop(positions, station)
    counts = {name: len(pos) for name, pos in positions.items()}
    _, _, excluded = usable(folder)
    everything = station_dop(load_positions(folder, all_listed=True)[0], station)
    summary = build_summary(station, result, start, counts, excluded, everything)
    print(summary)
    paths["summary"].write_text(summary)
    plot_station(paths["station"], station, result, t, start)
    every = int(MAP_STEP_S // STEP_S)
    sampled = {n: p[:, ::every] for n, p in positions.items()}
    lats, lons, p95 = world_pdop(sampled, MAP_GRID_DEG)
    plot_world(paths["world"], lats, lons, p95, start)
    print("Written " + ", ".join(str(p) for p in paths.values()))
    return paths


if __name__ == "__main__":
    main()
