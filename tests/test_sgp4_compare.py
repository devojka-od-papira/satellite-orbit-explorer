from datetime import timedelta

import numpy as np
import pytest

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
T = time_grid_s()


def find_sat(name):
    return next(s for s in SATS if name.lower() in s["OBJECT_NAME"].lower())


def at_epoch_s(sat):
    """Seconds after START of the satellite's GP epoch, as a 1-element array."""
    return np.array([-epoch_age_days(sat) * 86400])


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


# ---------- Kepler vs SGP4 ----------
@pytest.mark.parametrize("sat", SATS, ids=lambda s: s["OBJECT_NAME"])
def test_models_agree_at_epoch(sat):
    """At the GP epoch only mean vs osculating elements differ: tens of km."""
    assert compare(sat, at_epoch_s(sat))["total"][0] < 50


def test_leo_drifts_apart_in_track():
    """J2 makes the LEO orbit drift away from two-body, mostly along the track."""
    sat = find_sat("PULSAR")
    t = at_epoch_s(sat) + np.array([0.0, 86400.0])
    c = compare(sat, t)
    assert c["total"][1] > 10 * c["total"][0]
    radial, in_track, cross = np.abs(c["ric"][1])
    assert in_track > cross > radial


def test_geo_stays_close_over_the_day():
    c = compare(find_sat("SES-5"), T)
    assert c["total"].max() < 50


def test_epoch_age_sign():
    def sat_at(epoch):
        return {"EPOCH": f"{epoch:%Y-%m-%dT%H:%M:%S}.000000"}

    assert epoch_age_days(sat_at(START - timedelta(days=2))) == pytest.approx(2)
    assert epoch_age_days(sat_at(START + timedelta(hours=6))) == pytest.approx(-0.25)


def test_summary_lists_every_satellite():
    summary = build_summary()
    for sat in SATS:
        assert sat["OBJECT_NAME"] in summary
