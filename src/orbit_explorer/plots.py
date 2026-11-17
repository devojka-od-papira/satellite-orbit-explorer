"""Figures: ground tracks on a world map and speed profiles over the day."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from orbit_explorer.elements import load_elements
from orbit_explorer.frames import ecef_to_geodetic, eci_to_ecef, gmst_rad
from orbit_explorer.kepler import START, propagate_eci, time_grid_s

WORLD_MAP_PATH = Path("data/inputs/world_map.txt")
OUTPUT_DIR = Path("outputs")
GEO_IGSO = {38652: ("SES-5", "#2ca02c"), 42738: ("QZS-2", "#d62728")}


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
    t = time_grid_s()
    theta = gmst_rad(t)
    fig, ax = plt.subplots(figsize=(12, 6))
    for seg in load_world_map():
        ax.plot(seg[:, 1], seg[:, 0], color="0.6", linewidth=0.5)
    for sat in load_elements():
        pos_eci, _ = propagate_eci(sat, t)
        lat, lon, _ = ecef_to_geodetic(eci_to_ecef(pos_eci, theta))
        lon_b, lat_b = break_at_dateline(lon, lat)
        line = ax.plot(lon_b, lat_b, linewidth=1, label=sat["OBJECT_NAME"])[0]
        color = line.get_color()
        if np.ptp(lon) < 1:  # geostationary: the track is a single point
            ax.scatter(
                lon.mean(), lat.mean(), s=80, color=color, edgecolor="black", zorder=3
            )
            continue
        # start: filled dot, end: hollow ring, so the direction of motion is visible
        ax.scatter(lon[0], lat[0], s=50, color=color, edgecolor="black", zorder=3)
        ax.scatter(
            lon[-1], lat[-1], s=50, facecolor="white", edgecolor=color, lw=2, zorder=3
        )
    ax.scatter([], [], s=50, color="0.5", edgecolor="black", label="Start (00:00)")
    ax.scatter(
        [], [], s=50, facecolor="white", edgecolor="0.5", lw=2, label="End (24:00)"
    )
    ax.set(
        xlim=(-180, 180),
        ylim=(-90, 90),
        xlabel="Longitude [deg]",
        ylabel="Latitude [deg]",
        title=f"Ground tracks, {START:%Y-%m-%d} (24 h)",
    )
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_speeds(path: Path) -> None:
    t = time_grid_s()
    fig, ax = plt.subplots(figsize=(10, 5))
    for sat in load_elements():
        _, v = propagate_eci(sat, t)
        ax.plot(t / 3600, v, label=sat["OBJECT_NAME"])
    ax.set(
        xlabel=f"Hours since {START:%Y-%m-%d %H:%M} UTC",
        ylabel="Speed [km/s]",
        title="Orbital speed over the day",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_geo_vs_igso(path: Path) -> None:
    """LinkedIn landscape figure (1200x627 px): a geostationary dot versus an
    inclined figure-eight."""
    t = time_grid_s()
    theta = gmst_rad(t)
    fig, ax = plt.subplots(figsize=(12, 6.27))
    fig.subplots_adjust(top=0.9, bottom=0.18)
    for seg in load_world_map():
        ax.plot(seg[:, 1], seg[:, 0], color="0.65", linewidth=0.6)

    tracks = {}
    for sat in load_elements():
        if sat["NORAD_CAT_ID"] not in GEO_IGSO:
            continue
        name, color = GEO_IGSO[sat["NORAD_CAT_ID"]]
        pos_eci, _ = propagate_eci(sat, t)
        lat, lon, _ = ecef_to_geodetic(eci_to_ecef(pos_eci, theta))
        tracks[name] = (lat, lon)
        ax.plot(lon, lat, color=color, linewidth=2.5)

    lat, lon = tracks["SES-5"]
    ax.scatter(
        lon.mean(), lat.mean(), s=160, color="#2ca02c", edgecolor="black", zorder=3
    )
    ax.annotate(
        "SES-5: geostationary,\nstays at 5° E all day",
        xy=(lon.mean(), lat.mean()),
        xytext=(-37, -40),
        fontsize=14,
        arrowprops={"arrowstyle": "->", "lw": 1.5},
    )

    lat, lon = tracks["QZS-2"]
    top = np.argmax(lat)
    ax.annotate(
        "QZS-2: same ~24 h period,\ntilted 39° → figure-eight",
        xy=(lon[top], lat[top]),
        xytext=(45, 50),
        fontsize=14,
        arrowprops={"arrowstyle": "->", "lw": 1.5},
    )

    ax.set_xlim(-40, 190)
    ax.set_ylim(-50, 65)
    ax.set_aspect("equal")
    ax.set_title("Same orbital period, different inclination", fontsize=18, pad=12)
    ax.set_xlabel("Longitude [deg]", fontsize=12)
    ax.set_ylabel("Latitude [deg]", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.3)
    fig.text(
        0.5,
        0.04,
        f"24 h propagated from CelesTrak data, {START.day} {START:%b %Y} · "
        "github.com/devojka-od-papira/satellite-orbit-explorer",
        ha="center",
        fontsize=9,
        color="0.4",
    )
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_ground_tracks(OUTPUT_DIR / "ground_tracks.png")
    plot_speeds(OUTPUT_DIR / "speeds.png")
    plot_geo_vs_igso(OUTPUT_DIR / "geo_vs_igso.png")
    print("Written outputs/ground_tracks.png, speeds.png and geo_vs_igso.png")
