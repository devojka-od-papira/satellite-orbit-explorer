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

Each ground station (visibility.STATIONS; the viewer shows one at a time)
adds, with <st> its slug such as "belgrade":
- station-<st>:      a marker with the day's passes as a JSON property, which
                     the viewer lists in its panel.
- ring-<st>-<id>:    for LEO satellites, the circle on the ground inside which
                     the satellite is above the elevation mask from the station.
- link-<st>-<id>:    a line from the station to the SGP4 satellite, shown only
                     during passes (the same passes as visibility.py).
"""

import json
from datetime import timedelta
from pathlib import Path

import numpy as np

from orbit_explorer import style
from orbit_explorer.elements import (
    R_EARTH_KM,
    load_elements,
    mean_motion_rad_s,
    perigee_apogee_altitude_km,
    period_hours,
    semi_major_axis_km,
)
from orbit_explorer.frames import ecef_to_geodetic, eci_to_ecef, gmst_rad
from orbit_explorer.kepler import SPAN_S, START, propagate_eci, time_grid_s
from orbit_explorer.sgp4_compare import propagate_sgp4
from orbit_explorer.visibility import (
    ELEVATION_MASK_DEG,
    STATIONS,
    Station,
    find_passes,
    look_angles,
)

OUTPUT_PATH = Path("outputs/orbits.czml")
CZML_STEP = 12  # keep every 12th epoch (60 s), plenty for Lagrange interpolation
LEO_MAX_ALTITUDE_KM = 2000.0
WHITE = [255, 255, 255, 255]
BLACK = [0, 0, 0, 255]


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


def satellite_packets(
    sat: dict, rgb: tuple[int, int, int], pos_sgp4: np.ndarray
) -> list[dict]:
    """``pos_sgp4``: SGP4 TEME positions on the CZML grid (every CZML_STEP-th
    point), from the one SGP4 run in build_czml."""
    norad = sat["NORAD_CAT_ID"]
    name = sat["OBJECT_NAME"]
    period_s = period_hours(mean_motion_rad_s(sat["MEAN_MOTION"])) * 3600

    t = time_grid_s()[::CZML_STEP]
    theta = gmst_rad(t)
    pos_eci, speed = propagate_eci(sat, t)
    pos_ecef = eci_to_ecef(pos_eci, theta)
    lat, lon, _ = ecef_to_geodetic(pos_ecef)

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
                "outlineColor": {"rgba": WHITE},
                "outlineWidth": 1,
            },
            "label": {
                "text": style.short_name(sat),  # the viewer's panel uses it too
                "font": "500 13px -apple-system, BlinkMacSystemFont, Inter, sans-serif",
                "fillColor": {"rgba": rgba(rgb)},
                "outlineColor": {"rgba": BLACK},  # dark: bright labels
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


def visibility_radius_deg(orbit_radius_km: float, mask_deg: float) -> float:
    """Earth central angle from the station to the sub-satellite point at which
    a satellite at this distance from Earth's centre sits exactly on the
    elevation mask (spherical Earth of radius R_EARTH_KM, as for altitudes)."""
    el = np.radians(mask_deg)
    ratio = R_EARTH_KM * np.cos(el) / orbit_radius_km
    return float(np.degrees(np.arccos(ratio) - el))


def circle_degrees(lat_deg: float, lon_deg: float, radius_deg: float) -> list:
    """Flat CZML [lon, lat, 0, ...] for a small circle on the sphere."""
    lat0, lon0, d = np.radians([lat_deg, lon_deg, radius_deg])
    bearing = np.radians(np.arange(0, 361, 2.0))
    lat = np.arcsin(
        np.sin(lat0) * np.cos(d) + np.cos(lat0) * np.sin(d) * np.cos(bearing)
    )
    lon = lon0 + np.arctan2(
        np.sin(bearing) * np.sin(d) * np.cos(lat0),
        np.cos(d) - np.sin(lat0) * np.sin(lat),
    )
    lon = (np.degrees(lon) + 180) % 360 - 180
    return ground_track_degrees(np.degrees(lat), lon)


def at(seconds: float) -> str:
    """ISO time of a moment in the window, to the nearest second. Pass lists and
    link availability both use it, so the two switch at the same second."""
    return iso(START + timedelta(seconds=round(seconds)))


def pass_interval(rise_s: float, set_s: float) -> str:
    return f"{at(rise_s)}/{at(set_s)}"


def hemisphere(value: float, positive: str, negative: str) -> str:
    return f"{abs(value):.4f}° {positive if value >= 0 else negative}"


def station_packets(
    station: Station, sats: list[dict], sgp4_ecef: dict[int, np.ndarray], order: int
) -> list[dict]:
    """``sgp4_ecef``: each satellite's SGP4 ECEF positions on the full 5 s grid,
    propagated once and shared by all stations."""
    t = time_grid_s()
    for norad, pos in sgp4_ecef.items():
        if pos.shape != (len(t), 3):
            raise ValueError(
                f"SGP4 positions for {norad} have shape {pos.shape}; expected "
                f"{(len(t), 3)}, one per point of the full time grid"
            )
    st = station.slug
    passes, packets = [], []
    for sat in sats:
        norad = sat["NORAD_CAT_ID"]
        rgb = style.hex_to_rgb(style.sat_color(sat, dark=True))
        az, el, _ = look_angles(sgp4_ecef[norad], station)
        sat_passes = find_passes(t, el, az)
        for p in sat_passes:
            passes.append(
                {
                    "norad": norad,
                    "name": style.short_name(sat),
                    "rise": at(p.rise_s),
                    "set": at(p.set_s),
                    "maxEl": round(p.max_el_deg, 1),
                    "risesInWindow": p.rises_in_window,
                    "setsInWindow": p.sets_in_window,
                }
            )
        if sat_passes:
            packets.append(
                {
                    "id": f"link-{st}-{norad}",
                    "name": f"{style.short_name(sat)} seen from {station.name}",
                    "availability": [
                        pass_interval(p.rise_s, p.set_s) for p in sat_passes
                    ],
                    "polyline": {
                        "positions": {
                            "references": [
                                f"station-{st}#position",
                                f"sgp4-{norad}#position",
                            ]
                        },
                        "arcType": "NONE",
                        "width": 1.5,
                        "material": {"solidColor": {"color": {"rgba": rgba(rgb, 200)}}},
                    },
                }
            )

        a = semi_major_axis_km(mean_motion_rad_s(sat["MEAN_MOTION"]))
        _, apogee = perigee_apogee_altitude_km(a, sat["ECCENTRICITY"])
        if apogee < LEO_MAX_ALTITUDE_KM:
            radius = visibility_radius_deg(a, ELEVATION_MASK_DEG)
            packets.append(
                {
                    "id": f"ring-{st}-{norad}",
                    "name": f"Where {style.short_name(sat)} is visible from "
                    f"{station.name}",
                    "description": f"<p>While {style.short_name(sat)} is inside "
                    f"this circle (radius {np.radians(radius) * R_EARTH_KM:.0f}"
                    f" km), it is more than {ELEVATION_MASK_DEG:.0f}° above "
                    f"{station.name}'s horizon.</p>",
                    "polyline": {
                        "positions": {
                            "cartographicDegrees": circle_degrees(
                                station.lat_deg, station.lon_deg, radius
                            )
                        },
                        "clampToGround": True,
                        "width": 1.5,
                        # white, so it is not mistaken for the satellite's track
                        "material": {
                            "polylineDash": {
                                "color": {"rgba": [255, 255, 255, 200]},
                                "dashLength": 10,
                            }
                        },
                    },
                }
            )

    passes.sort(key=lambda p: p["rise"])
    station_packet = {
        "id": f"station-{st}",
        "name": f"{station.name} ground station",
        "description": f"<p>{hemisphere(station.lat_deg, 'N', 'S')}, "
        f"{hemisphere(station.lon_deg, 'E', 'W')}, "
        f"{station.height_km * 1000:.0f} m. Satellites count as visible above "
        f"{ELEVATION_MASK_DEG:.0f}° elevation.</p>",
        "position": {
            "cartographicDegrees": [
                station.lon_deg,
                station.lat_deg,
                station.height_km * 1000,
            ]
        },
        "point": {
            "pixelSize": 9,
            "color": {"rgba": WHITE},
            "outlineColor": {"rgba": BLACK},
            "outlineWidth": 2,
        },
        "label": {
            "text": station.name,
            "font": "600 13px -apple-system, BlinkMacSystemFont, Inter, sans-serif",
            "fillColor": {"rgba": WHITE},
            "outlineColor": {"rgba": BLACK},
            "outlineWidth": 3,
            "style": "FILL_AND_OUTLINE",
            "horizontalOrigin": "LEFT",
            "pixelOffset": {"cartesian2": [10, 0]},
        },
        "properties": {
            "order": order,  # the viewer lists stations in this order
            "maskDeg": ELEVATION_MASK_DEG,
            "passes": json.dumps(passes, separators=(",", ":")),
        },
    }
    return [station_packet, *packets]


def build_czml(stations: tuple[Station, ...] = STATIONS) -> list[dict]:
    sats = load_elements()
    t = time_grid_s()
    theta = gmst_rad(t)
    packets = [document_packet()]
    sgp4_ecef = {}
    for sat in sats:
        # One SGP4 run per satellite on the full grid serves both the CZML
        # samples (every CZML_STEP-th point) and every station's look angles.
        r_teme, _ = propagate_sgp4(sat, t)
        sgp4_ecef[sat["NORAD_CAT_ID"]] = eci_to_ecef(r_teme, theta)
        rgb = style.hex_to_rgb(style.sat_color(sat, dark=True))  # dark globe
        packets.extend(satellite_packets(sat, rgb, r_teme[::CZML_STEP]))
    for order, station in enumerate(stations):
        packets.extend(station_packets(station, sats, sgp4_ecef, order))
    return packets


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(build_czml(), separators=(",", ":")))
    print(f"Written to {OUTPUT_PATH}")
