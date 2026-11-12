"""Figures: ground tracks on a world map and speed profiles over the day."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from orbit_explorer.elements import load_elements
from orbit_explorer.frames import ecef_to_geodetic, eci_to_ecef, gmst_rad
from orbit_explorer.kepler import propagate_eci, time_grid_s

WORLD_MAP_PATH = Path("data/inputs/world_map.txt")
OUTPUT_DIR = Path("outputs")


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
        if np.ptp(lon) < 1:  # geostationary: the track is a single point
            ax.scatter(
                lon.mean(),
                lat.mean(),
                s=80,
                color=line.get_color(),
                edgecolor="black",
                zorder=3,
            )
    ax.set(
        xlim=(-180, 180),
        ylim=(-90, 90),
        xlabel="Longitude [deg]",
        ylabel="Latitude [deg]",
        title="Ground tracks, 2026-11-10 (24 h)",
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
        xlabel="Hours since 2026-11-10 00:00 UTC",
        ylabel="Speed [km/s]",
        title="Orbital speed over the day",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_ground_tracks(OUTPUT_DIR / "ground_tracks.png")
    plot_speeds(OUTPUT_DIR / "speeds.png")
    print("Written outputs/ground_tracks.png and outputs/speeds.png")
