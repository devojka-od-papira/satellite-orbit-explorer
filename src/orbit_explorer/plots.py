"""Figures: ground tracks on a world map and speed profiles over the day."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from orbit_explorer import style
from orbit_explorer.elements import load_elements
from orbit_explorer.frames import ecef_to_geodetic, eci_to_ecef, gmst_rad
from orbit_explorer.kepler import START, propagate_eci, time_grid_s

WORLD_MAP_PATH = Path("data/inputs/world_map.txt")
OUTPUT_DIR = Path("outputs")
GEO_IGSO = (38652, 42738)  # SES-5, QZS-2
# Keeps annotation text readable over coastlines.
LABEL_BOX = {
    "boxstyle": "round,pad=0.3",
    "fc": style.BACKGROUND,
    "ec": "none",
    "alpha": 0.85,
}


def load_world_map(path: Path = WORLD_MAP_PATH) -> list[np.ndarray]:
    """Coastline segments as arrays of (lat, lon); blank lines separate segments."""
    segments, current = [], []
    for line in path.read_text().splitlines():
        if line.strip():
            lat, lon = map(float, line.split())
            current.append((lat, lon))
        elif current:
            segments.append(np.array(current))
            current = []
    if current:
        segments.append(np.array(current))
    return segments


def break_at_dateline(
    lon: np.ndarray, lat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaN where the track jumps across +-180 deg, so no line crosses the map."""
    jumps = np.where(np.abs(np.diff(lon)) > 180)[0]
    return np.insert(lon, jumps + 1, np.nan), np.insert(lat, jumps + 1, np.nan)


def plot_ground_tracks(path: Path) -> None:
    style.use()
    t = time_grid_s()
    theta = gmst_rad(t)
    sats = load_elements()
    fig, ax = plt.subplots(figsize=(12, 6.6))
    style.map_axes(ax, load_world_map(), range(-180, 181, 60), range(-90, 91, 30))
    for sat in sats:
        color = style.sat_color(sat)
        pos_eci, _ = propagate_eci(sat, t)
        lat, lon, _ = ecef_to_geodetic(eci_to_ecef(pos_eci, theta))
        if np.ptp(lon) < 1:  # geostationary: the track is a single point
            ax.scatter(lon.mean(), lat.mean(), s=110, color=color, **style.DOT)
            continue
        lon_b, lat_b = break_at_dateline(lon, lat)
        ax.plot(lon_b, lat_b, color=color, linewidth=1.2, zorder=2)
        # start: filled dot, end: ring, so the direction of motion is visible
        ax.scatter(lon[0], lat[0], s=60, color=color, **style.DOT)
        ax.scatter(
            lon[-1],
            lat[-1],
            s=50,
            facecolor=style.BACKGROUND,
            edgecolor=color,
            linewidth=2,
            zorder=4,
        )
    ax.set(xlim=(-180, 180), ylim=(-90, 90), aspect="equal")
    style.header(
        ax,
        "Ground tracks",
        f"{style.date_label(START)} · 24 h from 00:00 UTC · two-body Kepler",
        caption="Orbits",
    )
    style.dot_legend(
        ax,
        [style.short_name(s) for s in sats],
        [style.sat_color(s) for s in sats],
        extra=[
            (style.dot(style.SECONDARY), "Start"),
            (style.dot(style.SECONDARY, hollow=True), "End"),
        ],
    )
    fig.savefig(path)
    plt.close(fig)


def plot_speeds(path: Path) -> None:
    style.use()
    t = time_grid_s()
    sats = load_elements()
    fig, ax = plt.subplots(figsize=(10, 5))
    for sat in sats:
        _, v = propagate_eci(sat, t)
        ax.plot(t / 3600, v, color=style.sat_color(sat))
    style.time_axis(ax)
    ax.set_ylim(0, 8)
    ax.yaxis.tick_right()
    ax.yaxis.set_major_formatter("{x:.0f} km/s")
    style.header(
        ax,
        "Orbital speed",
        f"{style.date_label(START)} · UTC · two-body Kepler",
        caption="Orbits",
    )
    style.dot_legend(
        ax, [style.short_name(s) for s in sats], [style.sat_color(s) for s in sats]
    )
    fig.savefig(path)
    plt.close(fig)


def plot_geo_vs_igso(path: Path) -> None:
    """LinkedIn landscape figure (1200x627 px): a geostationary dot versus an
    inclined figure-eight."""
    style.use()
    t = time_grid_s()
    theta = gmst_rad(t)
    fig = plt.figure(figsize=(12, 6.27))
    ax = fig.add_axes((0.06, 0.12, 0.88, 0.66))
    style.map_axes(ax, load_world_map(), range(-30, 181, 30), range(-45, 61, 15))

    tracks = {}
    for sat in load_elements():
        if sat["NORAD_CAT_ID"] not in GEO_IGSO:
            continue
        pos_eci, _ = propagate_eci(sat, t)
        lat, lon, _ = ecef_to_geodetic(eci_to_ecef(pos_eci, theta))
        tracks[sat["NORAD_CAT_ID"]] = (lat, lon, style.sat_color(sat))

    lat, lon, color = tracks[42738]
    ax.plot(lon, lat, color=color, linewidth=3, zorder=2)
    top = np.argmax(lat)
    ax.annotate(
        "QZS-2: same ~24 h period,\ntilted 39° → figure-eight",
        xy=(lon[top], lat[top]),
        xytext=(52, 47),
        fontsize=14,
        color=style.LABEL,
        bbox=LABEL_BOX,
        arrowprops={"arrowstyle": "-", "color": style.SECONDARY, "lw": 1},
    )

    lat, lon, color = tracks[38652]
    ax.scatter(lon.mean(), lat.mean(), s=220, color=color, **style.DOT)
    ax.annotate(
        "SES-5: geostationary,\nstays at 5° E all day",
        xy=(lon.mean(), lat.mean()),
        xytext=(-27, -38),
        fontsize=14,
        color=style.LABEL,
        bbox=LABEL_BOX,
        arrowprops={
            "arrowstyle": "-",
            "color": style.SECONDARY,
            "lw": 1,
            "shrinkB": 9,
        },
    )

    ax.set(xlim=(-40, 190), ylim=(-50, 65), aspect="equal")
    style.header(
        ax,
        "Same orbital period, different inclination",
        f"24 h propagated from CelesTrak data, {style.date_label(START)}",
        caption="Geostationary vs inclined geosynchronous",
    )
    fig.text(
        0.5,
        0.025,
        "github.com/devojka-od-papira/satellite-orbit-explorer",
        ha="center",
        fontsize=9,
        color=style.SECONDARY,
    )
    fig.savefig(path, dpi=100, bbox_inches=fig.bbox_inches)  # exact 1200x627
    plt.close(fig)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_ground_tracks(OUTPUT_DIR / "ground_tracks.png")
    plot_speeds(OUTPUT_DIR / "speeds.png")
    plot_geo_vs_igso(OUTPUT_DIR / "geo_vs_igso.png")
    print("Written outputs/ground_tracks.png, speeds.png and geo_vs_igso.png")
