"""One visual style for every figure and export, after Apple's chart design.

Colours are Apple's system colours. Each satellite keeps its colour everywhere
(PNG figures, KML, CZML, the 3D viewer), keyed by NORAD ID rather than by list
order. The four were chosen so every pair stays distinguishable with colour
blindness, measured as the worst pair's CVD ΔE in OKLab ×100 (target ≥ 8):
18.4 for the light variants on white, 14.5 for the dark variants.

The typeface is Inter (SIL Open Font License, bundled in ``fonts/``), an open
font close to Apple's SF Pro, which may only be used for Apple-platform UI.
"""

import functools
from pathlib import Path

FONT_DIR = Path(__file__).parent / "fonts"
FONT = "Inter"
DISPLAY_FONT = "Inter Display"

# Apple system colours. Light mode for figures on white; dark mode for the
# dark globe in Google Earth and the 3D viewer.
SATELLITES = {
    # NORAD ID: (short name, light, dark)
    64541: ("PULSAR-0", "#007AFF", "#0A84FF"),  # blue
    59598: ("GALILEO 29", "#FF9500", "#FF9F0A"),  # orange
    38652: ("SES-5", "#30B0C7", "#40C8E0"),  # teal
    42738: ("QZS-2", "#D30F45", "#FF375F"),  # pink (accessible variant on white)
}
FALLBACK = ("", "#8E8E93", "#98989D")  # system gray, for any other satellite

# Kepler − SGP4 components: hues not used for any satellite.
RADIAL = "#5856D6"  # indigo
IN_TRACK = "#D70015"  # red
CROSS_TRACK = "#00C7BE"  # mint

# Ink and surfaces (Apple label / secondary label / separators / grouped fill).
LABEL = "#1D1D1F"
SECONDARY = "#6E6E73"
SEPARATOR = "#C7C7CC"
GRID = "#E5E5EA"
FILL = "#F2F2F7"
LAND = "#AEAEB2"
BACKGROUND = "#FFFFFF"


def sat_color(sat: dict, dark: bool = False) -> str:
    _, light, dark_hex = SATELLITES.get(sat["NORAD_CAT_ID"], FALLBACK)
    return dark_hex if dark else light


def short_name(sat: dict) -> str:
    return SATELLITES.get(sat["NORAD_CAT_ID"], FALLBACK)[0] or sat["OBJECT_NAME"]


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def kml_color(color: str, alpha: int = 255) -> str:
    """KML wants aabbggrr."""
    r, g, b = hex_to_rgb(color)
    return f"{alpha:02x}{b:02x}{g:02x}{r:02x}"


# ---------- Matplotlib ----------
RC = {
    "font.family": FONT,
    "font.size": 11,
    "text.color": LABEL,
    "axes.edgecolor": SEPARATOR,
    "axes.labelcolor": SECONDARY,
    "axes.titlecolor": LABEL,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "axes.unicode_minus": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": SECONDARY,
    "ytick.color": SECONDARY,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.major.pad": 6,
    "ytick.major.pad": 6,
    "lines.linewidth": 2,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "legend.frameon": False,
    "figure.facecolor": BACKGROUND,
    "axes.facecolor": BACKGROUND,
    "savefig.facecolor": BACKGROUND,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
}

# A white ring around dots lifts them off lines and fills.
DOT = {"edgecolor": BACKGROUND, "linewidth": 1.5, "zorder": 4}


@functools.cache
def register_fonts() -> None:
    """Add the bundled Inter files to matplotlib's font list, once per process."""
    from matplotlib import font_manager

    known = {f.fname for f in font_manager.fontManager.ttflist}
    for path in sorted(FONT_DIR.glob("*.ttf")):
        if str(path) not in known:
            font_manager.fontManager.addfont(str(path))


def styled(plot):
    """Decorator: draw inside the style without changing global rcParams,
    so figures made afterwards (e.g. in a notebook) keep their own look."""

    @functools.wraps(plot)
    def wrapper(*args, **kwargs):
        import matplotlib.pyplot as plt

        register_fonts()
        with plt.rc_context(RC):
            return plot(*args, **kwargs)

    return wrapper


def use() -> None:
    """Apply the style globally, for interactive use."""
    import matplotlib.pyplot as plt

    register_fonts()
    plt.rcParams.update(RC)


def header(
    ax, title: str, subtitle: str = "", caption: str = "", pad: float = 12
) -> None:
    """Left-aligned header above the axes: small caps caption, title, subtitle.

    ``pad`` is the gap in points between the axes top and the lowest line.
    """
    kw = {"xy": (0, 1), "xycoords": "axes fraction", "textcoords": "offset points"}
    y = pad
    if subtitle:
        ax.annotate(subtitle, xytext=(0, y), color=SECONDARY, fontsize=11, **kw)
        y += 20
    ax.annotate(
        title,
        xytext=(0, y),
        fontsize=20,
        fontweight="semibold",
        family=DISPLAY_FONT,
        **kw,
    )
    if caption:
        ax.annotate(
            caption.upper(),
            xytext=(0, y + 32),
            color=SECONDARY,
            fontsize=10,
            fontweight="semibold",
            **kw,
        )


def dot(color: str, hollow: bool = False):
    """A legend handle: a filled dot, or a ring."""
    from matplotlib.lines import Line2D

    if hollow:
        return Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=BACKGROUND,
            markeredgecolor=color,
            markeredgewidth=2,
        )
    return Line2D([], [], marker="o", linestyle="", markersize=8, color=color)


def dot_legend(ax, labels, colors, extra=(), **kwargs):
    """One row of coloured dots and names below the axes, aligned left
    (where Swift Charts puts its legend).

    ``extra`` adds (handle, label) pairs after the dots, e.g. marker keys.
    """
    from matplotlib.transforms import ScaledTranslation

    handles = [dot(c) for c in colors] + [h for h, _ in extra]
    labels = list(labels) + [label for _, label in extra]
    below_ticks = ScaledTranslation(-6 / 72, -30 / 72, ax.figure.dpi_scale_trans)
    options = {
        "loc": "upper left",
        "bbox_to_anchor": (0, 0),
        "bbox_transform": ax.transAxes + below_ticks,
        "ncol": len(handles),
        "fontsize": 11,
        "handletextpad": 0.2,
        "columnspacing": 1.4,
        "borderaxespad": 0,
    }
    options.update(kwargs)
    return ax.legend(handles, labels, **options)


def time_axis(ax, span_h: float, step_h: int = 3) -> None:
    """Hours since the window start as 00:00, 03:00, … up to ``span_h``."""
    hours = range(0, int(span_h) + 1, step_h)
    ax.set_xlim(0, span_h)
    ax.set_xticks(list(hours), [f"{h:02d}:00" for h in hours])


def lon_label(deg: float) -> str:
    deg = (deg + 180) % 360 - 180 if abs(deg) > 180 else deg
    if deg == 0 or abs(deg) == 180:
        return f"{abs(deg):.0f}°"
    return f"{abs(deg):.0f}°{'E' if deg > 0 else 'W'}"


def lat_label(deg: float) -> str:
    return "0°" if deg == 0 else f"{abs(deg):.0f}°{'N' if deg > 0 else 'S'}"


def map_axes(ax, segments, xticks, yticks) -> None:
    """Coastlines in light gray, a graticule instead of a frame, °E/°N labels."""
    for seg in segments:
        ax.plot(seg[:, 1], seg[:, 0], color=LAND, linewidth=0.6, zorder=1)
    ax.grid(True, axis="both")
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks(xticks, [lon_label(x) for x in xticks])
    ax.set_yticks(yticks, [lat_label(y) for y in yticks])


def date_label(start) -> str:
    return f"{start.day} {start:%B %Y}"
