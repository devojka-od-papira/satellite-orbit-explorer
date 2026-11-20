"""SP3 precise orbits: parsing, time scales, interpolation and our satellites."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from conftest import FROZEN_START

from orbit_explorer.elements import MU_EARTH_KM3_S2
from orbit_explorer.frames import eci_to_ecef, gmst_rad
from orbit_explorer.sgp4_compare import propagate_sgp4
from orbit_explorer.sp3 import (
    PRECISE_ORBITS,
    SP3_IDS,
    Sp3,
    format_sp3,
    lagrange_interpolate,
    main,
    parse_sp3,
    positions_at,
    read_sp3,
)

# The stored SP3 file is for 2026-11-17, the day of the frozen GP data
# (conftest.py). SGP4 comparisons propagate those elements to that day, not
# today's: the current data moves on every day, the precise orbits don't.
START = FROZEN_START


@pytest.fixture(scope="module")
def sp3():
    return read_sp3(PRECISE_ORBITS)


def tiny_sp3(**changes) -> Sp3:
    epochs = [datetime(2026, 11, 17) + timedelta(minutes=5 * k) for k in range(3)]
    pos = {
        "E29": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [np.nan] * 3]),
        "J02": np.array([[-1.5, 0.25, 9.0], [7.0, 8.0, 9.0], [1.0, 1.0, 1.0]]),
    }
    return Sp3(epochs, pos, **{"agency": "TST", "comments": ["test"], **changes})


# ---------- Reading and writing ----------
def test_stored_file_is_for_the_frozen_gp_day(sp3):
    assert sp3.epochs[0].date() == FROZEN_START.date()


def test_stored_file_header(sp3):
    assert sp3.time_system == "GPS"
    assert sp3.coord_system == "IGc20"
    assert sp3.agency == "GFZ"
    assert (sp3.data_used, sp3.orbit_type) == ("u+U", "FIT")
    assert len(sp3.epochs) == 289  # one day every 5 minutes, both ends
    assert sp3.interval_s == 300
    assert sp3.epochs[0] == datetime(2026, 11, 17)
    assert list(sp3.positions) == ["E29", "J02"]
    for pos in sp3.positions.values():
        assert pos.shape == (289, 3) and not np.isnan(pos).any()


def test_write_then_read_gives_the_same_positions():
    original = tiny_sp3()
    back = parse_sp3(format_sp3(original))
    assert back.epochs == original.epochs
    assert (back.time_system, back.coord_system, back.agency) == ("GPS", "IGc20", "TST")
    for sat_id, pos in original.positions.items():
        np.testing.assert_array_equal(back.positions[sat_id], pos)  # NaN round-trips


def test_written_header_follows_the_sp3d_columns():
    first = format_sp3(tiny_sp3()).splitlines()[0]
    assert first[:31] == "#dP2026 11 17  0  0  0.00000000"
    assert first[32:39] == "      3"  # number of epochs
    assert first[40:45] == "  u+U"
    assert first[46:51] == "IGc20"
    assert first[56:60] == " TST"


def test_stored_file_is_a_faithful_subset(sp3):
    """It re-reads exactly as written (the subset was made by this module)."""
    again = parse_sp3(format_sp3(sp3))
    for sat_id in sp3.positions:
        np.testing.assert_array_equal(again.positions[sat_id], sp3.positions[sat_id])


def test_producer_fields_are_kept_not_relabelled():
    """A subset of an extrapolated (EXT) file must not call itself fitted."""
    text = format_sp3(tiny_sp3(data_used="ORBIT", orbit_type="EXT"))
    back = parse_sp3(text)
    assert (back.data_used, back.orbit_type) == ("ORBIT", "EXT")
    assert text.splitlines()[0][52:55] == "EXT"


def test_non_ascii_comments_are_read(tmp_path):
    text = format_sp3(tiny_sp3(comments=["Universität Bern, Zürich"]))
    path = tmp_path / "accents.sp3"
    path.write_bytes(text.encode("latin-1"))
    sp3 = read_sp3(path)
    assert sp3.comments[0] == "Universität Bern, Zürich"
    assert list(sp3.positions) == ["E29", "J02"]


def test_bad_positions_become_nan():
    text = format_sp3(tiny_sp3())
    assert "PE29      0.000000      0.000000      0.000000" in text
    assert np.isnan(parse_sp3(text).positions["E29"][2]).all()


@pytest.mark.parametrize(
    "first_line, message",
    [
        ("#aP2026 11 17  0  0  0.00000000", "SP3-c or SP3-d"),
        ("#dV2026 11 17  0  0  0.00000000", "position"),
        ("not sp3", "SP3-c or SP3-d"),
    ],
)
def test_unsupported_files_are_refused(first_line, message):
    with pytest.raises(ValueError, match=message):
        parse_sp3(first_line + "\nEOF\n")


def test_subset_and_missing_satellite():
    sp3 = tiny_sp3()
    assert list(sp3.subset(["J02"]).positions) == ["J02"]
    with pytest.raises(KeyError, match="G01"):
        sp3.subset(["J02", "G01"])


def test_cli_keeps_only_the_named_satellites(tmp_path):
    src = tmp_path / "in.sp3"
    src.write_text(format_sp3(tiny_sp3()))
    out = tmp_path / "out.sp3"
    main([str(src), "--keep", "J02", "-o", str(out)])
    kept = read_sp3(out)
    assert list(kept.positions) == ["J02"]
    assert kept.comments[0] == "Subset of in.sp3: J02"


# ---------- Time scales ----------
def test_gps_time_is_18_seconds_ahead_of_utc(sp3):
    assert sp3.utc_epochs()[0] == datetime(2026, 11, 16, 23, 59, 42, tzinfo=UTC)
    t = sp3.seconds_since(START)
    assert t[0] == -18 and t[-1] == 86400 - 18


@pytest.mark.parametrize(
    "system, offset_s", [("GPS", 18), ("GAL", 18), ("TAI", 37), ("UTC", 0)]
)
def test_time_scales(system, offset_s):
    sp3 = tiny_sp3(time_system=system)
    assert sp3.utc_epochs()[0] == datetime(2026, 11, 17, tzinfo=UTC) - timedelta(
        seconds=offset_s
    )


def test_unknown_time_scale_and_old_dates_are_refused():
    with pytest.raises(ValueError, match="GLO"):
        tiny_sp3(time_system="GLO").utc_epochs()
    old = tiny_sp3()
    old.epochs = [e.replace(year=2015) for e in old.epochs]
    with pytest.raises(ValueError, match="2017"):
        old.utc_epochs()


def test_epochs_past_the_checked_leap_second_table_are_refused():
    """A leap second announced later would make every offset 1 s wrong."""
    future = tiny_sp3()
    future.epochs = [e.replace(year=2027, month=3) for e in future.epochs]
    with pytest.raises(ValueError, match="Bulletin C"):
        future.utc_epochs()


# ---------- Interpolation ----------
def test_lagrange_is_exact_for_polynomials_up_to_its_order():
    t_nodes = np.arange(0, 3600.0, 300.0)
    t = np.linspace(0, 3300, 97)
    poly = np.polynomial.Polynomial([3.0, -2.0, 0.5, 1e-3, -1e-6, 2e-10])  # degree 5
    got = lagrange_interpolate(t_nodes, poly(t_nodes), t, order=5)
    np.testing.assert_allclose(got, poly(t), rtol=1e-9, atol=1e-9)


def test_lagrange_returns_the_nodes_and_nan_outside():
    t_nodes = np.arange(0, 3600.0, 300.0)
    values = np.sin(t_nodes / 1000)
    np.testing.assert_allclose(
        lagrange_interpolate(t_nodes, values, t_nodes, order=8), values, atol=1e-12
    )
    outside = lagrange_interpolate(t_nodes, values, np.array([-1.0, 3301.0]), order=8)
    assert np.isnan(outside).all()  # no extrapolation


def test_lagrange_needs_enough_nodes():
    with pytest.raises(ValueError, match="11 nodes"):
        lagrange_interpolate(np.arange(5.0), np.arange(5.0), np.array([1.0]), order=10)


def test_lagrange_on_an_orbit_is_far_below_a_centimetre():
    """A circular Galileo-like orbit sampled every 5 min, checked between nodes."""
    a = 29600.0
    n = np.sqrt(MU_EARTH_KM3_S2 / a**3)
    t_nodes = np.arange(0, 86401.0, 300.0)

    def orbit(t):
        return a * np.column_stack([np.cos(n * t), np.sin(n * t), np.zeros_like(t)])

    t = np.arange(0, 86400.0, 37.0)  # off the nodes
    got = lagrange_interpolate(t_nodes, orbit(t_nodes), t)
    assert np.linalg.norm(got - orbit(t), axis=1).max() < 1e-6  # km: a millimetre


@pytest.mark.parametrize("sat_id", list(SP3_IDS.values()))
def test_leaving_out_an_epoch_it_is_recovered_to_a_centimetre(sp3, sat_id):
    """Real data: drop every other epoch and interpolate the dropped ones."""
    t = sp3.seconds_since(START)
    pos = sp3.positions[sat_id]
    kept, dropped = slice(0, None, 2), slice(1, None, 2)  # 10-minute spacing
    got = lagrange_interpolate(t[kept], pos[kept], t[dropped])
    error = np.linalg.norm(got - pos[dropped], axis=1)
    assert np.nanmax(error) < 0.01  # km: 1 cm even at twice the spacing


# ---------- Our satellites in the file ----------
@pytest.mark.parametrize("norad", list(SP3_IDS), ids=lambda n: SP3_IDS[n])
def test_each_sp3_id_is_our_satellite(sp3, frozen_sats, norad):
    """At 01:00 UTC the matching SP3 satellite is within a few km of SGP4 in
    the Earth-fixed frame, and every other one is thousands of km away."""
    t = np.array([3600.0])
    r_teme, _ = propagate_sgp4(frozen_sats[norad], t, START)
    ecef = eci_to_ecef(r_teme, gmst_rad(t, START))[0]
    distance = {
        sat_id: np.linalg.norm(positions_at(sp3, sat_id, t, START)[0] - ecef)
        for sat_id in sp3.positions
    }
    assert distance[SP3_IDS[norad]] < 10
    assert all(d > 1000 for i, d in distance.items() if i != SP3_IDS[norad])


@pytest.mark.parametrize("norad", list(SP3_IDS), ids=lambda n: SP3_IDS[n])
def test_ignoring_gps_time_costs_tens_of_kilometres(sp3, frozen_sats, norad):
    """The classic mistake: reading SP3 epochs as UTC. Against SGP4 the error
    then grows from a few km to tens of km."""
    t = np.arange(0.0, 86400 - 18, 600.0)
    r_teme, _ = propagate_sgp4(frozen_sats[norad], t, START)
    ecef = eci_to_ecef(r_teme, gmst_rad(t, START))
    sat_id = SP3_IDS[norad]
    right = positions_at(sp3, sat_id, t, START)
    as_if_utc = lagrange_interpolate(
        sp3.seconds_since(START) + 18, sp3.positions[sat_id], t
    )
    right_km = np.linalg.norm(right - ecef, axis=1).mean()
    wrong_km = np.nanmean(np.linalg.norm(as_if_utc - ecef, axis=1))
    assert right_km < 10
    assert wrong_km > 20 and wrong_km > 4 * right_km


@pytest.mark.parametrize(
    "sat_id, radius_km", [("E29", 29600), ("J02", 42164)], ids=["E29", "J02"]
)
def test_orbit_radius(sp3, sat_id, radius_km):
    r = np.linalg.norm(sp3.positions[sat_id], axis=1)
    assert abs(r.mean() - radius_km) < 400  # QZS-2 swings ±3,200 km (e = 0.076)


def test_positions_on_our_grid_are_nan_only_after_the_file_ends(sp3):
    t = np.arange(0, 86401.0, 5.0)
    pos = positions_at(sp3, "E29", t, START)
    missing = np.isnan(pos).any(axis=1)
    assert not missing[t <= 86400 - 18].any()
    assert missing[t > 86400 - 18].all()  # last 18 s: past the GPS-time file
