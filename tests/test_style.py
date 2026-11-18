import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager

from orbit_explorer import style
from orbit_explorer.elements import load_elements

SATS = load_elements()


def test_every_satellite_has_its_own_colour_in_both_modes():
    for dark in (False, True):
        colours = [style.sat_color(s, dark=dark) for s in SATS]
        assert len(set(colours)) == len(SATS)


def test_colour_follows_the_satellite_not_the_order():
    assert [style.sat_color(s) for s in reversed(SATS)] == [
        style.sat_color(s) for s in SATS
    ][::-1]


def test_unknown_satellite_falls_back_to_gray_and_its_own_name():
    sat = {"NORAD_CAT_ID": 1, "OBJECT_NAME": "SPUTNIK"}
    assert style.sat_color(sat) == style.FALLBACK[1]
    assert style.short_name(sat) == "SPUTNIK"


def test_component_colours_differ_from_satellite_colours():
    components = {style.RADIAL, style.IN_TRACK, style.CROSS_TRACK}
    assert len(components) == 3
    assert not components & {style.sat_color(s) for s in SATS}


def test_colour_conversions():
    assert style.hex_to_rgb("#007AFF") == (0, 122, 255)
    assert style.kml_color("#007AFF") == "ffff7a00"  # aabbggrr
    assert style.kml_color("#007AFF", alpha=128) == "80ff7a00"


def test_bundled_fonts_are_registered():
    style.register_fonts()
    names = {f.name for f in font_manager.fontManager.ttflist}
    assert {style.FONT, style.DISPLAY_FONT} <= names
    # no silent fallback to DejaVu Sans
    found = font_manager.findfont(style.FONT, fallback_to_default=False)
    assert str(style.FONT_DIR) in found


def test_use_applies_the_style_globally():
    try:
        style.use()
        assert plt.rcParams["font.family"] == [style.FONT]
    finally:
        plt.rcdefaults()


def test_map_labels():
    assert [style.lon_label(x) for x in (-120, 0, 60, 180)] == [
        "120°W",
        "0°",
        "60°E",
        "180°",
    ]
    assert [style.lat_label(y) for y in (-30, 0, 45)] == ["30°S", "0°", "45°N"]


def test_styled_figures_leave_global_settings_alone(tmp_path):
    """A notebook figure drawn after ours must keep matplotlib's own look."""
    from orbit_explorer.plots import plot_speeds

    plt.rcdefaults()
    before = dict(plt.rcParams)
    plot_speeds(tmp_path / "speeds.png")
    assert dict(plt.rcParams) == before
