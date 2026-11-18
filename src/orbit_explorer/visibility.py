"""Satellite visibility from a ground station: look angles, passes and a sky plot.

Positions come from SGP4 (the more accurate model); the summary also shows how
much the two-body Kepler model shifts each rise and set. Both are rotated to
ECEF with the same GMST angle and then into the station's local East-North-Up
frame, which gives azimuth (clockwise from north), elevation and range.

    python -m orbit_explorer.visibility                    # Belgrade
    python -m orbit_explorer.visibility --station tokyo    # or sydney, quito
    python -m orbit_explorer.visibility --name "Novi Sad" --lat 45.25 --lon 19.84

Belgrade writes outputs/visibility.txt, visibility_elevation.png and
visibility_skyplot.png; the viewer's other stations add their name, e.g.
visibility_tokyo.txt, and your own station is marked custom, e.g.
visibility_custom-novi-sad.txt.
"""

import argparse
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from orbit_explorer import style
from orbit_explorer.elements import load_elements
from orbit_explorer.frames import eci_to_ecef, geodetic_to_ecef, gmst_rad
from orbit_explorer.kepler import SPAN_S, START, propagate_eci, time_grid_s
from orbit_explorer.sgp4_compare import propagate_sgp4

OUTPUT_DIR = Path("outputs")
ELEVATION_MASK_DEG = 10.0  # common GNSS cut-off angle


@dataclass(frozen=True)
class Station:
    name: str
    lat_deg: float
    lon_deg: float
    height_km: float = 0.0

    @property
    def slug(self) -> str:
        """Lower-case id for CZML packets, URLs and file names, e.g. 'novi-sad'.
        Only letters, digits and single hyphens between them, so it is safe in
        a path; empty if the name has no letter or digit."""
        kept = re.sub(r"[^\w\s-]", "", self.name.lower()).replace("_", " ")
        return re.sub(r"[\s-]+", "-", kept).strip("-")

    @property
    def coordinates(self) -> str:
        """E.g. '33.8688° S, 151.2093° E'."""

        def hemi(value: float, positive: str, negative: str) -> str:
            return f"{abs(value):.4f}° {positive if value >= 0 else negative}"

        return f"{hemi(self.lat_deg, 'N', 'S')}, {hemi(self.lon_deg, 'E', 'W')}"


BELGRADE = Station("Belgrade", 44.8178, 20.4569, 0.117)
# Stations offered in the 3D viewer: Europe, East Asia (where QZS-2 is high),
# the southern hemisphere and the equator.
STATIONS = (
    BELGRADE,
    Station("Tokyo", 35.6762, 139.6503, 0.040),
    Station("Sydney", -33.8688, 151.2093, 0.058),
    Station("Quito", -0.1807, -78.4678, 2.850),
)
STATIONS_BY_SLUG = {s.slug: s for s in STATIONS}
assert len(STATIONS_BY_SLUG) == len(STATIONS), "station slugs must differ"


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
    """Intervals with elevation at or above the mask, rise/set interpolated.

    ``>=`` here, in with_mask_crossings and in the 3D viewer, so the table, the
    figure and the viewer agree on what counts as visible."""
    above = el >= mask
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
    where = station.coordinates
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


def with_mask_crossings(
    x: np.ndarray, el: np.ndarray, mask: float
) -> tuple[np.ndarray, np.ndarray]:
    """Insert the interpolated points where ``el`` crosses ``mask``, so a curve
    drawn only where el >= mask starts and ends exactly on the mask line."""
    above = el >= mask
    i = np.nonzero(above[1:] != above[:-1])[0]  # crossing between i and i + 1
    frac = (mask - el[i]) / (el[i + 1] - el[i])
    x_cross = x[i] + frac * (x[i + 1] - x[i])
    return np.insert(x, i + 1, x_cross), np.insert(el, i + 1, np.full(len(i), mask))


def mask_label_hour(
    h: np.ndarray,
    elevations: list[np.ndarray],
    width_h: float,
    preferred_h: float = 6.2,
    margin_h: float = 0.3,
) -> float:
    """Where the "10° mask" label starts on the mask line.

    The label's box runs from just below the line to about 12° above it and is
    ``width_h`` hours wide (measured from the drawn text), plus ``margin_h`` on
    each side. Takes the clear spot nearest ``preferred_h``; if no spot is
    clear, the one that the fewest curve samples pass through.
    """
    band = np.zeros_like(h, dtype=bool)
    for el in elevations:
        band |= (el > ELEVATION_MASK_DEG - 2) & (el < ELEVATION_MASK_DEG + 12)
    starts = np.round(np.arange(0.2, h[-1] - width_h, 0.1), 1)
    blocked = [
        np.count_nonzero(band[(h >= x - margin_h) & (h <= x + width_h + margin_h)])
        for x in starts
    ]
    best = min(zip(blocked, np.abs(starts - preferred_h), starts, strict=True))
    return float(best[2])


def text_width(ax, text) -> float:
    """Width of a drawn text in the axes' x data units (hours here)."""
    box = text.get_window_extent(ax.figure.canvas.get_renderer())
    (x0, _), (x1, _) = ax.transData.inverted().transform([(box.x0, 0), (box.x1, 0)])
    return x1 - x0


@style.styled
def plot_elevation(path: Path, station: Station = BELGRADE) -> None:
    """Elevation over the day; full colour only where the station sees it."""
    t = time_grid_s()
    h = t / 3600
    sats = load_elements()
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.axhspan(-90, ELEVATION_MASK_DEG, color=style.FILL, linewidth=0, zorder=0)
    ax.axhline(
        ELEVATION_MASK_DEG,
        color=style.SEPARATOR,
        linewidth=1,
        linestyle=(0, (3, 3)),
        zorder=1,
    )
    elevations = []
    for sat in sats:
        color = style.sat_color(sat)
        _, el, _ = satellite_look_angles(sat, t, station)
        elevations.append(el)
        ax.plot(h, el, color=color, linewidth=1.2, alpha=0.3, zorder=2)
        # bright exactly where find_passes counts the satellite as visible
        h_x, el_x = with_mask_crossings(h, el, ELEVATION_MASK_DEG)
        visible = np.where(el_x >= ELEVATION_MASK_DEG, el_x, np.nan)
        ax.plot(h_x, visible, color=color, linewidth=2.5, zorder=3)
    style.time_axis(ax, SPAN_S / 3600)
    ax.set_ylim(-90, 90)
    label = ax.annotate(
        f"{ELEVATION_MASK_DEG:.0f}° mask",
        xy=(0, ELEVATION_MASK_DEG),
        xytext=(0, 4),
        textcoords="offset points",
        color=style.SECONDARY,
        fontsize=10,
    )
    # placed once the axes limits are set, so its width in hours is right
    label.xy = (mask_label_hour(h, elevations, text_width(ax, label)), label.xy[1])
    ax.set_yticks([-90, -45, 0, 45, 90], ["−90°", "−45°", "0°", "45°", "90°"])
    ax.yaxis.tick_right()
    style.header(
        ax,
        f"Elevation seen from {station.name}",
        f"{style.date_label(START)} · UTC · SGP4 · faded below the "
        f"{ELEVATION_MASK_DEG:.0f}° mask",
        caption="Visibility",
    )
    style.dot_legend(
        ax, [style.short_name(s) for s in sats], [style.sat_color(s) for s in sats]
    )
    fig.savefig(path)
    plt.close(fig)


@style.styled
def plot_skyplot(path: Path, station: Station = BELGRADE) -> None:
    """Polar sky view: north up, azimuth clockwise, zenith at the centre."""
    t = time_grid_s()
    sats = load_elements()
    fig, ax = plt.subplots(figsize=(7, 7.6), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.fill_between(
        np.linspace(0, 2 * np.pi, 200),
        90 - ELEVATION_MASK_DEG,
        90,
        color=style.FILL,
        linewidth=0,
        zorder=0,
    )
    never_up = []  # legend indices of satellites with no track in this sky
    for i, sat in enumerate(sats):
        color = style.sat_color(sat)
        az, el, _ = satellite_look_angles(sat, t, station)
        if not (el >= 0).any():
            never_up.append(i)
        r = np.where(el >= 0, 90 - el, np.nan)  # hide below the horizon
        theta = np.radians(az)
        jumps = np.nonzero(np.abs(np.diff(r)) > 5)[0] + 1  # don't join separate passes
        ax.plot(
            np.insert(theta, jumps, np.nan),
            np.insert(r, jumps, np.nan),
            color=color,
            linewidth=2,
            zorder=2,
        )
        visible = el >= 0
        if visible.any() and np.ptp(az[visible]) < 1 and np.ptp(el[visible]) < 1:
            # geostationary: the track is a single point, draw it as a dot
            ax.scatter(
                np.radians(az[visible].mean()),
                90 - el[visible].mean(),
                s=110,
                color=color,
                **style.DOT,
            )
    ax.grid(True, color=style.GRID, linewidth=0.8)
    ax.spines["polar"].set_color(style.SEPARATOR)
    ax.set_rlim(0, 90)
    ax.set_rticks([0, 30, 60, 90 - ELEVATION_MASK_DEG])
    ax.set_yticklabels(["90°", "60°", "30°", f"{ELEVATION_MASK_DEG:.0f}°"])
    ax.set_rlabel_position(22.5)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"], fontweight="semibold", color=style.LABEL)
    style.header(
        ax,
        f"Sky from {station.name}",
        f"{style.date_label(START)} · 24 h · SGP4",
        caption="Visibility",
        pad=28,  # clear of the N label
    )
    legend_args = (
        ax,
        [
            style.short_name(s) + (" · below horizon" if i in never_up else "")
            for i, s in enumerate(sats)
        ],
        [style.sat_color(s) for s in sats],
    )
    legend_options = {
        "dimmed": never_up,
        "bbox_to_anchor": (0, -0.07),  # below the S label
        "bbox_transform": ax.transAxes,
    }
    legend = style.dot_legend(*legend_args, **legend_options)
    renderer = fig.canvas.get_renderer()
    if legend.get_window_extent(renderer).width > ax.get_window_extent(renderer).width:
        legend.remove()  # the "below horizon" notes made one row too wide
        style.dot_legend(*legend_args, ncol=2, **legend_options)
    fig.savefig(path)
    plt.close(fig)


# The options that describe your own station: (name, type, help).
CUSTOM_OPTIONS = (
    ("name", str, f"default {BELGRADE.name}"),
    ("lat", float, "deg N (negative for S)"),
    ("lon", float, "deg E (negative for W)"),
    ("height", float, "metres above the WGS84 ellipsoid"),
)
# Lowest and highest ground: the Dead Sea shore (about -430 m) to above Everest.
HEIGHT_RANGE_M = (-500.0, 9000.0)


def parse_station(argv: list[str] | None = None) -> Station:
    """One of the viewer's stations (--station), or your own (--lat/--lon).
    Belgrade when neither is given."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--station",
        type=str.lower,
        choices=list(STATIONS_BY_SLUG),
        help=f"one of the 3D viewer's stations (default {BELGRADE.slug})",
    )
    custom = parser.add_argument_group(
        "your own station", "instead of --station; Belgrade's values fill any gaps"
    )
    for option, kind, text in CUSTOM_OPTIONS:
        custom.add_argument(f"--{option}", type=kind, help=text)
    args = parser.parse_args(argv)
    given = [o for o, _, _ in CUSTOM_OPTIONS if getattr(args, o) is not None]
    if args.station:
        if given:
            parser.error(
                f"--station can't be combined with --{', --'.join(given)}; "
                "pick a station or describe your own"
            )
        return STATIONS_BY_SLUG[args.station]
    if not given:
        return BELGRADE
    station = Station(
        BELGRADE.name if args.name is None else args.name,
        BELGRADE.lat_deg if args.lat is None else args.lat,
        BELGRADE.lon_deg if args.lon is None else args.lon,
        BELGRADE.height_km if args.height is None else args.height / 1000,
    )
    if not -90 <= station.lat_deg <= 90:
        parser.error(f"--lat must be between -90 and 90, not {station.lat_deg:g}")
    if not -180 <= station.lon_deg <= 180:
        parser.error(f"--lon must be between -180 and 180, not {station.lon_deg:g}")
    low, high = HEIGHT_RANGE_M
    height_m = station.height_km * 1000
    if not low <= height_m <= high:  # also refuses nan and inf
        parser.error(
            f"--height must be between {low:g} and {high:g} m, not {height_m:g}"
        )
    if not station.slug:
        parser.error("--name needs at least one letter or digit")
    return station


def output_paths(station: Station, output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    """Belgrade keeps the plain names (the README images and the daily PR use
    them); the viewer's other stations add their slug, and your own station is
    marked custom, so it never overwrites a viewer station of the same name."""
    if station == BELGRADE:
        stem = "visibility"
    elif STATIONS_BY_SLUG.get(station.slug) == station:
        stem = f"visibility_{station.slug}"
    else:
        stem = f"visibility_custom-{station.slug}"
    return {
        "summary": output_dir / f"{stem}.txt",
        "elevation": output_dir / f"{stem}_elevation.png",
        "skyplot": output_dir / f"{stem}_skyplot.png",
    }


def main(
    argv: list[str] | None = None, output_dir: Path = OUTPUT_DIR
) -> dict[str, Path]:
    station = parse_station(argv)
    paths = output_paths(station, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(station)
    print(summary)
    paths["summary"].write_text(summary)
    plot_elevation(paths["elevation"], station)
    plot_skyplot(paths["skyplot"], station)
    print("Written " + ", ".join(str(p) for p in paths.values()))
    return paths


if __name__ == "__main__":
    main()
