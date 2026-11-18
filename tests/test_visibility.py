import numpy as np
import pytest

from orbit_explorer.elements import load_elements
from orbit_explorer.frames import geodetic_to_ecef
from orbit_explorer.kepler import time_grid_s
from orbit_explorer.visibility import (
    BELGRADE,
    CUSTOM_OPTIONS,
    ELEVATION_MASK_DEG,
    STATIONS,
    STATIONS_BY_SLUG,
    Station,
    build_summary,
    clock,
    enu_rotation,
    find_passes,
    look_angles,
    main,
    mask_label_hour,
    output_paths,
    parse_station,
    satellite_look_angles,
    with_mask_crossings,
)

T = time_grid_s()


def find_sat(name):
    return next(s for s in load_elements() if name.lower() in s["OBJECT_NAME"].lower())


# ---------- Look angles ----------
def test_enu_rotation_is_orthonormal():
    R = enu_rotation(BELGRADE.lat_deg, BELGRADE.lon_deg)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_point_straight_up_is_at_zenith():
    st = BELGRADE
    above = geodetic_to_ecef(st.lat_deg, st.lon_deg, st.height_km + 500)
    _, el, rng = look_angles(above, st)
    assert el[0] == pytest.approx(90, abs=1e-6)
    assert rng[0] == pytest.approx(500, abs=1e-6)


@pytest.mark.parametrize(
    "row, expected_az", [(1, 0.0), (0, 90.0)], ids=["north", "east"]
)
def test_local_horizontal_directions(row, expected_az):
    """A point 100 km along the local north/east axis: that azimuth, elevation 0."""
    st = Station("equator", 0.0, 0.0)
    site = geodetic_to_ecef(st.lat_deg, st.lon_deg, st.height_km)
    target = site + 100 * enu_rotation(st.lat_deg, st.lon_deg)[row]
    az, el, _ = look_angles(target, st)
    assert az[0] == pytest.approx(expected_az, abs=1e-6)
    assert el[0] == pytest.approx(0, abs=1e-6)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        satellite_look_angles(load_elements()[0], T[:3], BELGRADE, model="sgp8")


# ---------- Pass detection ----------
def test_find_passes_interpolates_crossings():
    t = np.arange(0, 1001, 10.0)
    el = 30 * np.sin(2 * np.pi * t / 1000)  # above 10° between ~54 s and ~446 s
    passes = find_passes(t, el, np.zeros_like(t), mask=10)
    assert len(passes) == 1
    expected_rise = np.arcsin(1 / 3) / (2 * np.pi) * 1000
    assert passes[0].rise_s == pytest.approx(expected_rise, abs=1)
    assert passes[0].set_s == pytest.approx(500 - expected_rise, abs=1)
    assert passes[0].max_el_deg == pytest.approx(30, abs=0.1)
    assert passes[0].rises_in_window and passes[0].sets_in_window


def test_find_passes_at_window_edges():
    t = np.arange(0, 101, 1.0)
    el = np.where((t < 20) | (t > 80), 45.0, -5.0)  # up at both ends
    passes = find_passes(t, el, np.zeros_like(t), mask=10)
    assert len(passes) == 2
    assert not passes[0].rises_in_window and passes[0].rise_s == 0
    assert not passes[1].sets_in_window and passes[1].set_s == 100


def test_never_above_mask_gives_no_passes():
    t = np.arange(0, 100, 1.0)
    assert find_passes(t, np.full_like(t, 5.0), np.zeros_like(t), mask=10) == []


# ---------- The four satellites from Belgrade ----------
def test_geostationary_stays_put_in_the_southern_sky():
    az, el, _ = satellite_look_angles(find_sat("SES-5"), T, BELGRADE)
    assert np.ptp(el) < 1 and np.ptp(az) < 1
    assert el.min() > ELEVATION_MASK_DEG
    assert 180 < az.mean() < 225  # south, a little west (SES-5 is at 5° E)


def test_leo_passes_are_short():
    az, el, _ = satellite_look_angles(find_sat("PULSAR"), T, BELGRADE)
    passes = find_passes(T, el, az)
    assert len(passes) >= 2
    assert all(p.set_s - p.rise_s < 15 * 60 for p in passes)


def test_qzss_is_regional():
    """QZS-2 serves Asia-Oceania: from Belgrade it never clears the mask."""
    _, el, _ = satellite_look_angles(find_sat("QZS-2"), T, BELGRADE)
    assert el.max() < ELEVATION_MASK_DEG


def test_qzss_is_high_over_tokyo():
    _, el, _ = satellite_look_angles(
        find_sat("QZS-2"), T, Station("Tokyo", 35.68, 139.77)
    )
    assert el.max() > 70


def test_clock_shows_window_end_as_24h():
    assert clock(0) == "00:00:00"
    assert clock(3661) == "01:01:01"
    assert clock(86400) == "24:00:00"


def test_summary_and_cli():
    station = parse_station(["--name", "Novi Sad", "--lat", "45.25", "--lon", "19.84"])
    assert station == Station("Novi Sad", 45.25, 19.84, BELGRADE.height_km)
    summary = build_summary(station)
    assert "NOVI SAD" in summary
    for sat in load_elements():
        assert sat["OBJECT_NAME"] in summary


# ---------- Choosing a station ----------
def test_no_options_means_belgrade():
    assert parse_station([]) is BELGRADE


@pytest.mark.parametrize("station", STATIONS, ids=[s.slug for s in STATIONS])
def test_station_option_picks_a_viewer_station(station):
    assert parse_station(["--station", station.slug]) is station
    assert parse_station(["--station", station.name.upper()]) is station


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--station", "paris"], "invalid choice: 'paris'"),
        (["--station", "tokyo", "--lat", "35"], "can't be combined with --lat"),
        (["--lat", "95"], "--lat must be between -90 and 90"),
        (["--lon", "-200"], "--lon must be between -180 and 180"),
        (["--name", "!!!", "--lat", "1"], "at least one letter or digit"),
        (["--name=---", "--lat", "1"], "at least one letter or digit"),
        (["--name", ""], "at least one letter or digit"),
        (["--height", "nan"], "--height must be between -500 and 9000 m, not nan"),
        (["--height", "inf"], "--height must be between"),
        (["--height", "-20000"], "--height must be between"),
    ],
    ids=[
        "unknown",
        "station-and-custom",
        "lat",
        "lon",
        "name",
        "hyphens",
        "empty-name",
        "height-nan",
        "height-inf",
        "height-low",
    ],
)
def test_bad_station_options_are_refused(argv, message, capsys):
    with pytest.raises(SystemExit) as exit_info:
        parse_station(argv)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_southern_and_western_coordinates_are_written_that_way():
    sydney, quito = (STATIONS_BY_SLUG[s] for s in ("sydney", "quito"))
    assert sydney.coordinates == "33.8688° S, 151.2093° E"
    assert quito.coordinates == "0.1807° S, 78.4678° W"
    assert f"Station: {quito.coordinates}, 2850 m" in build_summary(quito)


def test_slug_is_safe_in_a_file_name():
    assert Station("Novi Sad", 0, 0).slug == "novi-sad"
    assert Station("São Paulo / GRU", 0, 0).slug == "são-paulo-gru"
    assert Station("my_site", 0, 0).slug == "my-site"
    assert Station("Rio - Centro", 0, 0).slug == "rio-centro"  # one hyphen
    assert Station(" -Quito- ", 0, 0).slug == "quito"
    assert Station("---", 0, 0).slug == ""  # refused by the CLI


def test_viewer_station_slugs_are_unchanged():
    """They are CZML ids and URL hashes (#tokyo) already in use."""
    assert list(STATIONS_BY_SLUG) == ["belgrade", "tokyo", "sydney", "quito"]


def test_custom_station_options_are_all_covered():
    """Every custom option conflicts with --station, including any added later."""
    for option, _, _ in CUSTOM_OPTIONS:
        value = "X" if option == "name" else "1"
        with pytest.raises(SystemExit):
            parse_station(["--station", "tokyo", f"--{option}", value])


def test_belgrade_keeps_the_plain_output_names(tmp_path):
    names = {k: p.name for k, p in output_paths(BELGRADE, tmp_path).items()}
    assert names == {
        "summary": "visibility.txt",
        "elevation": "visibility_elevation.png",
        "skyplot": "visibility_skyplot.png",
    }
    # explicitly giving Belgrade's own values is still Belgrade
    assert parse_station(["--name", "Belgrade"]) == BELGRADE


@pytest.mark.parametrize(
    "station, stem",
    [
        (STATIONS_BY_SLUG["tokyo"], "visibility_tokyo"),
        (Station("Tokyo", 35.68, 139.77), "visibility_custom-tokyo"),
        (Station("Belgrade", 1, 1), "visibility_custom-belgrade"),
        (Station("Novi Sad", 45.25, 19.84), "visibility_custom-novi-sad"),
    ],
    ids=["viewer-station", "same-name-elsewhere", "belgrade-elsewhere", "custom"],
)
def test_custom_stations_never_share_a_viewer_stations_files(tmp_path, station, stem):
    assert output_paths(station, tmp_path)["summary"].name == f"{stem}.txt"


def test_main_writes_the_stations_own_files(tmp_path):
    paths = main(["--station", "tokyo"], output_dir=tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "visibility_tokyo.txt",
        "visibility_tokyo_elevation.png",
        "visibility_tokyo_skyplot.png",
    ]
    summary = paths["summary"].read_text()
    assert summary.startswith("VISIBILITY FROM TOKYO")
    qzs = next(line for line in summary.splitlines() if line.startswith("QZS-2"))
    assert "0 pass(es)" not in qzs  # the reason Tokyo is offered


def test_mask_crossings_are_inserted_on_the_line():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    el = np.array([0.0, 20.0, 20.0, 0.0])
    x_c, el_c = with_mask_crossings(x, el, 10)
    np.testing.assert_allclose(x_c, [0, 0.5, 1, 2, 2.5, 3])
    np.testing.assert_allclose(el_c, [0, 10, 20, 20, 10, 0])


def test_peak_just_below_mask_gets_no_visible_points():
    """The figure must agree with find_passes: 9.7° is not a pass."""
    x = np.arange(5.0)
    el = np.array([0, 5, 9.7, 5, 0])
    x_c, el_c = with_mask_crossings(x, el, 10)
    assert len(x_c) == len(x)
    assert not (el_c >= 10).any()
    assert find_passes(x, el, np.zeros_like(x), mask=10) == []


H = np.arange(0, 24, 0.01)
FLAT = np.full_like(H, -40.0)


def test_mask_label_moves_off_a_curve_near_the_mask():
    assert mask_label_hour(H, [FLAT], 1.4) == pytest.approx(6.2)  # the usual spot
    crossing = np.where((H > 6) & (H < 7), 15.0, -40.0)  # just above the line
    x = mask_label_hour(H, [FLAT, crossing], 1.4)
    assert x >= 7 + 0.3 or x + 1.4 + 0.3 <= 6  # clear of it, margins included
    assert x == pytest.approx(7.3)  # and the nearest clear spot


def test_mask_label_with_no_clear_spot_takes_the_least_crowded():
    """Curves near the line all day, but thinnest around 18 h."""
    crowded = np.where(np.abs(H - 18.5) < 1.5, -40.0, 12.0)
    other = np.where(np.abs(H - 18.5) < 0.6, -40.0, 15.0)
    x = mask_label_hour(H, [crowded, other], 1.4)
    assert 17.4 <= x <= 18.2  # around 18.5 h, not the usual 6.2 h


def test_mask_label_width_is_measured_from_the_text():
    """A wider label needs a wider gap, so it lands farther from the curve."""
    crossing = np.where((H > 6) & (H < 7), 15.0, -40.0)
    narrow = mask_label_hour(H, [crossing], 1.0, preferred_h=5.5)
    wide = mask_label_hour(H, [crossing], 3.0, preferred_h=5.5)
    assert narrow == pytest.approx(4.7)  # fits before the curve: 4.7 + 1.0 + 0.3
    assert wide == pytest.approx(7.3)  # would have to start at 2.7; after is nearer
