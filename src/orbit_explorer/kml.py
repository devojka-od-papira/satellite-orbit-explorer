"""KML export for Google Earth: ground tracks and 3D orbits."""

from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from orbit_explorer import style
from orbit_explorer.elements import load_elements
from orbit_explorer.frames import ecef_to_geodetic, eci_to_ecef, gmst_rad
from orbit_explorer.kepler import START, propagate_eci, time_grid_s

OUTPUT_PATH = Path("outputs/orbits.kml")
KML_STEP = 12  # keep every 12th epoch (60 s) so Google Earth stays fast


def coordinates(lat: np.ndarray, lon: np.ndarray, h_km: np.ndarray) -> str:
    """KML coordinate tuples 'lon,lat,alt_m' separated by spaces."""
    return " ".join(
        f"{lo:.6f},{la:.6f},{h * 1000:.1f}" for la, lo, h in zip(lat, lon, h_km)
    )


def placemark(name: str, color: str, coords: str, altitude_mode: str) -> str:
    return (
        f"    <Placemark>\n"
        f"      <name>{escape(name)}</name>\n"
        f"      <Style><LineStyle><color>{color}</color><width>2</width></LineStyle></Style>\n"
        f"      <LineString>\n"
        f"        <altitudeMode>{altitude_mode}</altitudeMode>\n"
        f"        <coordinates>{coords}</coordinates>\n"
        f"      </LineString>\n"
        f"    </Placemark>"
    )


def folder(title: str, items: list[str]) -> str:
    return (
        f"  <Folder>\n    <name>{title}</name>\n" + "\n".join(items) + "\n  </Folder>"
    )


def build_kml() -> str:
    t = time_grid_s()[::KML_STEP]
    theta = gmst_rad(t)
    theta0 = np.full_like(t, theta[0])  # Earth frozen at the start epoch
    tracks, orbits_ecef, orbits_eci = [], [], []
    for sat in load_elements():
        color = style.kml_color(style.sat_color(sat, dark=True))  # dark globe
        name = sat["OBJECT_NAME"]
        pos_eci, _ = propagate_eci(sat, t)

        lat, lon, h = ecef_to_geodetic(eci_to_ecef(pos_eci, theta))
        tracks.append(
            placemark(name, color, coordinates(lat, lon, 0 * h), "clampToGround")
        )
        orbits_ecef.append(placemark(name, color, coordinates(lat, lon, h), "absolute"))

        lat_i, lon_i, h_i = ecef_to_geodetic(eci_to_ecef(pos_eci, theta0))
        orbits_eci.append(
            placemark(name, color, coordinates(lat_i, lon_i, h_i), "absolute")
        )

    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<kml xmlns="http://www.opengis.net/kml/2.2">',
            "<Document>",
            f"  <name>Satellite Orbit Explorer, {START:%Y-%m-%d}</name>",
            folder("Ground tracks", tracks),
            folder("Orbits (Earth-fixed, 3D)", orbits_ecef),
            folder("Orbits (inertial, Earth frozen at start)", orbits_eci),
            "</Document>",
            "</kml>",
            "",
        ]
    )


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(build_kml())
    print(f"Written to {OUTPUT_PATH}")
