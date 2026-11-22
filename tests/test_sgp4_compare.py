"""Kepler vs SGP4.

Propagation and the comparison machinery run on the current GP data. The
"how far apart" tests use the frozen GP data of 2026-11-17 (conftest.py): how
far Kepler and SGP4/SDP4 are apart depends on the element set (QZS-2's
lunisolar terms, the satellite's phase), and the refresh of 2026-11-22 failed
on a 50 km threshold that a new QZS-2 element set crossed (61.8 km).
"""

from datetime import timedelta

import numpy as np
import pytest
from conftest import FROZEN_GP_PATH, FROZEN_START

from orbit_explorer.elements import load_elements
from orbit_explorer.kepler import START, time_grid_s
from orbit_explorer.sgp4_compare import (
    build_summary,
    compare,
    epoch_age_days,
    propagate_sgp4,
    ric_components,
)

SATS = load_elements()
FROZEN_SATS = load_elements(FROZEN_GP_PATH)
T = time_grid_s()


def find_sat(name, sats=SATS):
    return next(s for s in sats if name.lower() in s["OBJECT_NAME"].lower())


def at_epoch_s(sat, start=START):
    """Seconds after ``start`` of the satellite's GP epoch, as a 1-element array."""
    return np.array([-epoch_age_days(sat, start) * 86400])


# ---------- SGP4 propagation ----------
@pytest.mark.parametrize("sat", SATS, ids=lambda s: s["OBJECT_NAME"])
def test_sgp4_propagates_whole_day(sat):
    r, v = propagate_sgp4(sat, T)
    assert r.shape == v.shape == (len(T), 3)
    assert np.all(np.isfinite(r))


@pytest.mark.parametrize("sat", SATS, ids=lambda s: s["OBJECT_NAME"])
def test_sgp4_radius_close_to_kepler_semi_major_axis(sat):
    from orbit_explorer.elements import mean_motion_rad_s, semi_major_axis_km

    a = semi_major_axis_km(mean_motion_rad_s(sat["MEAN_MOTION"]))
    r, _ = propagate_sgp4(sat, T)
    assert np.linalg.norm(r, axis=1).mean() == pytest.approx(a, rel=0.01)


# ---------- RIC frame ----------
def test_ric_unit_directions():
    r = np.array([[7000.0, 0.0, 0.0]])
    v = np.array([[0.0, 7.5, 0.0]])
    for diff, expected in [
        ([1.0, 0, 0], [1, 0, 0]),  # along r: radial
        ([0, 1.0, 0], [0, 1, 0]),  # along v: in-track
        ([0, 0, 1.0], [0, 0, 1]),  # along r x v: cross-track
    ]:
        np.testing.assert_allclose(ric_components(np.array([diff]), r, v), [expected])


def test_ric_keeps_length():
    rng = np.random.default_rng(0)
    r, v = propagate_sgp4(SATS[0], T[:100])
    diff = rng.normal(size=(100, 3))
    ric = ric_components(diff, r, v)
    np.testing.assert_allclose(
        np.linalg.norm(ric, axis=1), np.linalg.norm(diff, axis=1)
    )


# ---------- Kepler vs SGP4 (frozen 2026-11-17 data) ----------
@pytest.mark.parametrize("sat", FROZEN_SATS, ids=lambda s: s["OBJECT_NAME"])
def test_models_agree_at_epoch(sat):
    """At the GP epoch only mean vs osculating elements (and, for the deep-space
    orbits, SDP4's lunisolar terms) differ: 8-32 km on 2026-11-17."""
    c = compare(sat, at_epoch_s(sat, FROZEN_START), FROZEN_START)
    assert c["total"][0] < 50


def test_leo_drifts_apart_in_track():
    """J2 makes the LEO orbit drift away from two-body, mostly along the track."""
    sat = find_sat("PULSAR", FROZEN_SATS)
    t = at_epoch_s(sat, FROZEN_START) + np.array([0.0, 86400.0])
    c = compare(sat, t, FROZEN_START)
    assert c["total"][1] > 10 * c["total"][0]
    radial, in_track, cross = np.abs(c["ric"][1])
    assert in_track > cross > radial


def test_geo_stays_close_over_the_day():
    c = compare(find_sat("SES-5", FROZEN_SATS), T, FROZEN_START)
    assert c["total"].max() < 50


def test_comparison_runs_on_the_current_data():
    """What holds every day: the comparison runs and is finite. No distance
    bound: with elements a week old Kepler and SGP4 are legitimately far apart
    in LEO, and stale data is flagged by the refresh, not failed. (That SGP4
    itself stays on a sane orbit is test_sgp4_radius_close_to_kepler_...)"""
    for sat in SATS:
        c = compare(sat, T)
        assert c["total"].shape == (len(T),) and c["ric"].shape == (len(T), 3)
        assert np.isfinite(c["total"]).all() and np.isfinite(c["ric"]).all()


def test_epoch_age_sign():
    def sat_at(epoch):
        return {"EPOCH": f"{epoch:%Y-%m-%dT%H:%M:%S}.000000"}

    assert epoch_age_days(sat_at(START - timedelta(days=2))) == pytest.approx(2)
    assert epoch_age_days(sat_at(START + timedelta(hours=6))) == pytest.approx(-0.25)


def test_summary_lists_every_satellite():
    summary = build_summary()
    for sat in SATS:
        assert sat["OBJECT_NAME"] in summary
