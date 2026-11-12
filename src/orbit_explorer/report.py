"""Text report with the orbit summary and the elements at GP epoch."""

from pathlib import Path

from orbit_explorer.elements import (
    load_elements,
    mean_motion_rad_s,
    perigee_apogee_altitude_km,
    period_hours,
    semi_major_axis_km,
)
from orbit_explorer.kepler import SPAN_S, START, STEP_S, propagate_eci, time_grid_s

OUTPUT_PATH = Path("outputs/orbit_report.txt")


def build_report() -> str:
    """Return the full report as text."""
    t = time_grid_s()
    lines = [
        "ORBIT REPORT",
        f"Window: {START:%Y-%m-%d %H:%M} UTC + {SPAN_S // 3600} h, step {STEP_S} s",
        "",
        "Orbit summary",
        (
            f"{'Satellite':24s} {'NORAD':>6s} {'Period[h]':>10s} "
            f"{'Perigee[km]':>12s} {'Apogee[km]':>11s} {'Speed[km/s]':>14s}"
        ),
    ]
    for sat in load_elements():
        n = mean_motion_rad_s(sat["MEAN_MOTION"])
        a = semi_major_axis_km(n)
        hp, ha = perigee_apogee_altitude_km(a, sat["ECCENTRICITY"])
        _, v = propagate_eci(sat, t)
        lines.append(
            f"{sat['OBJECT_NAME']:24s} {sat['NORAD_CAT_ID']:>6d} {period_hours(n):>10.3f} "
            f"{hp:>12.1f} {ha:>11.1f} {f'{v.min():.3f} - {v.max():.3f}':>14s}",
        )

    lines += [
        "",
        "Orbital elements at GP epoch (angles in degrees)",
        f"{'Satellite':24s} {'Epoch (UTC)':>19s} {'a[km]':>10s} {'e':>10s} {'i':>10s} {'RAAN':>10s} {'argp':>10s} {'M':>10s}",
    ]
    for sat in load_elements():
        a = semi_major_axis_km(mean_motion_rad_s(sat["MEAN_MOTION"]))
        epoch = sat["EPOCH"][:19].replace("T", " ")
        lines.append(
            f"{sat['OBJECT_NAME']:24s} {epoch:>19s} {a:>10.1f} "
            f"{sat['ECCENTRICITY']:>10.6f} {sat['INCLINATION']:>10.2f} "
            f"{sat['RA_OF_ASC_NODE']:>10.2f} {sat['ARG_OF_PERICENTER']:>10.2f} "
            f"{sat['MEAN_ANOMALY']:>10.2f}"
        )
    lines += ["", "Altitudes are above a spherical Earth (R = 6378.137 km)."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    report = build_report()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report)
    print(report)
    print(f"Written to {OUTPUT_PATH}")
