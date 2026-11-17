"""CZML export for CesiumJS: animated satellites, ground tracks and inertial orbits.

Each satellite gets five packets:
- sat-<id>:       the moving satellite (Kepler). Positions are Earth-fixed (ECEF)
                  from our own frames.py, so the web viewer shows exactly the
                  tracks in the plots.
- track-<id>:     the 24 h ground track, drawn on the globe surface.
- orbit-<id>:     the orbit in the inertial frame (ECI), drawn as a path one
                  period long, so the orbit's shape in space is visible.
- sgp4-<id>:      the same satellite propagated with SGP4, as a hollow ring.
                  TEME positions go through the same GMST rotation as Kepler,
                  so the distance on screen is the one in sgp4_compare.
- sgp4orbit-<id>: the SGP4 orbit in the inertial frame, dashed.
"""

import json
from datetime import timedelta
from pathlib import Path

import numpy as np

from orbit_explorer.elements import (
    load_elements,
    mean_motion_rad_s,
    perigee_apogee_altitude_km,
    period_hours,
    semi_major_axis_km,
)
from orbit_explorer.frames import ecef_to_geodetic, eci_to_ecef, gmst_rad
from orbit_explorer.kepler import SPAN_S, START, propagate_eci, time_grid_s
from orbit_explorer.sgp4_compare import propagate_sgp4

OUTPUT_PATH = Path("outputs/orbits.czml")
CZML_STEP = 12  # keep every 12th epoch (60 s), plenty for Lagrange interpolation
COLORS_RGB = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40)]


def iso(dt) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


INTERVAL = f"{iso(START)}/{iso(START + timedelta(seconds=SPAN_S))}"


def cartesian_samples(t_s: np.ndarray, pos_km: np.ndarray) -> list[int]:
    """Flat CZML sample list [t, x, y, z, t, x, y, z, ...] in seconds and metres."""
    samples = np.column_stack([t_s, pos_km * 1000])
    return np.rint(samples).astype(int).ravel().tolist()


def ground_track_degrees(lat: np.ndarray, lon: np.ndarray) -> list[float]:
    """Flat CZML list [lon, lat, height, ...] on the surface (height 0)."""
    coords = np.column_stack([lon, lat, np.zeros_like(lat)])
    return np.round(coords, 4).ravel().tolist()


def rgba(rgb: tuple[int, int, int], alpha: int = 255) -> list[int]:
    return [*rgb, alpha]


def description(sat: dict, speed_km_s: np.ndarray) -> str:
    """HTML table for Cesium's info box, with the same numbers as the report."""
    n = mean_motion_rad_s(sat["MEAN_MOTION"])
    a = semi_major_axis_km(n)
    perigee, apogee = perigee_apogee_altitude_km(a, sat["ECCENTRICITY"])
    rows = [
        ("NORAD ID", sat["NORAD_CAT_ID"]),
        ("GP epoch (UTC)", sat["EPOCH"][:19].replace("T", " ")),
        ("Period", f"{period_hours(n):.3f} h"),
        ("Semi-major axis", f"{a:.1f} km"),
        ("Eccentricity", f"{sat['ECCENTRICITY']:.6f}"),
        ("Inclination", f"{sat['INCLINATION']:.2f}°"),
        ("Perigee altitude", f"{perigee:.1f} km"),
        ("Apogee altitude", f"{apogee:.1f} km"),
        ("Speed", f"{speed_km_s.min():.3f} – {speed_km_s.max():.3f} km/s"),
    ]
    body = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f"<table>{body}</table>"


def document_packet() -> dict:
    return {
        "id": "document",
        "name": f"Satellite Orbit Explorer, {START:%Y-%m-%d}",
        "version": "1.0",
        "clock": {
            "interval": INTERVAL,
            "currentTime": iso(START),
            "multiplier": 600,  # 10 minutes per second: one day in 2.4 minutes
            "range": "LOOP_STOP",
            "step": "SYSTEM_CLOCK_MULTIPLIER",
        },
    }


def satellite_packets(sat: dict, rgb: tuple[int, int, int]) -> list[dict]:
    norad = sat["NORAD_CAT_ID"]
    name = sat["OBJECT_NAME"]
    period_s = period_hours(mean_motion_rad_s(sat["MEAN_MOTION"])) * 3600

    t = time_grid_s()[::CZML_STEP]
    theta = gmst_rad(t)
    pos_eci, speed = propagate_eci(sat, t)
    pos_ecef = eci_to_ecef(pos_eci, theta)
    lat, lon, _ = ecef_to_geodetic(pos_ecef)
    pos_sgp4, _ = propagate_sgp4(sat, t)

    def position(frame: str, pos_km: np.ndarray) -> dict:
        return {
            "epoch": iso(START),
            "referenceFrame": frame,
            "interpolationAlgorithm": "LAGRANGE",
            "interpolationDegree": 5,
            "cartesian": cartesian_samples(t, pos_km),
        }

    solid = {"solidColor": {"color": {"rgba": rgba(rgb)}}}
    return [
        {
            "id": f"sat-{norad}",
            "name": name,
            "availability": INTERVAL,
            "description": description(sat, speed),
            "position": position("FIXED", pos_ecef),
            "point": {
                "pixelSize": 10,
                "color": {"rgba": rgba(rgb)},
                "outlineColor": {"rgba": [0, 0, 0, 255]},
                "outlineWidth": 1,
            },
            "label": {
                "text": name.split(" (")[0],
                "font": "13px sans-serif",
                "fillColor": {"rgba": rgba(rgb)},
                "outlineColor": {"rgba": [0, 0, 0, 255]},
                "outlineWidth": 2,
                "style": "FILL_AND_OUTLINE",
                "horizontalOrigin": "LEFT",
                "pixelOffset": {"cartesian2": [10, 0]},
            },
        },
        {
            "id": f"track-{norad}",
            "name": f"{name} ground track",
            "polyline": {
                "positions": {"cartographicDegrees": ground_track_degrees(lat, lon)},
                "clampToGround": True,
                "width": 1.5,
                "material": {"solidColor": {"color": {"rgba": rgba(rgb, 170)}}},
            },
        },
        {
            "id": f"orbit-{norad}",
            "name": f"{name} orbit (inertial)",
            "availability": INTERVAL,
            "position": position("INERTIAL", pos_eci),
            "path": {
                "leadTime": period_s / 2,
                "trailTime": period_s / 2,
                "width": 1.5,
                "resolution": 120,
                "material": solid,
            },
        },
        {
            "id": f"sgp4-{norad}",
            "name": f"{name} (SGP4)",
            "availability": INTERVAL,
            "description": "<p>Position from SGP4/SDP4, propagated from the same "
            "GP elements. The filled dot is the two-body Kepler position.</p>",
            "position": position("FIXED", eci_to_ecef(pos_sgp4, theta)),
            "point": {
                "pixelSize": 9,
                "color": {"rgba": [0, 0, 0, 0]},
                "outlineColor": {"rgba": rgba(rgb)},
                "outlineWidth": 2,
            },
        },
        {
            "id": f"sgp4orbit-{norad}",
            "name": f"{name} orbit (SGP4, inertial)",
            "availability": INTERVAL,
            "position": position("INERTIAL", pos_sgp4),
            "path": {
                "leadTime": period_s / 2,
                "trailTime": period_s / 2,
                "width": 1.5,
                "resolution": 120,
                "material": {
                    "polylineDash": {"color": {"rgba": rgba(rgb)}, "dashLength": 12}
                },
            },
        },
    ]


def build_czml() -> list[dict]:
    packets = [document_packet()]
    for sat, rgb in zip(load_elements(), COLORS_RGB):
        packets.extend(satellite_packets(sat, rgb))
    return packets


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(build_czml(), separators=(",", ":")))
    print(f"Written to {OUTPUT_PATH}")
