"""Orbit determination: the fit recovers what it should, and the real day."""

import numpy as np
import pytest

from orbit_explorer import gravity, j2, od
from orbit_explorer.sp3 import PRECISE_ORBITS, read_sp3

GALILEO, QZS2 = 59598, 42738


@pytest.fixture(scope="module")
def results():
    return od.analyse()


@pytest.fixture(scope="module")
def e29_start():
    t = np.array([0.0])
    r, v = od.observed(read_sp3(PRECISE_ORBITS), "E29", t)
    return np.concatenate([r[0], v[0]])


# ---------- The machinery ----------
def test_with_only_j2_it_is_the_j2_propagator(e29_start):
    """Same RK4, the field cut to J2 (EGM2008's constants), the Moon and the
    Sun: the two propagators agree to rounding."""
    t = np.arange(0.0, 6 * 3600 + 1, 300.0)
    field = gravity.load_field(degree=2)
    field.C[2, 1:] = 0.0
    field.S[2, 1:] = 0.0
    r_od, _ = od.propagate(e29_start, t, field=field, substeps=1)
    earth = j2.Earth("EGM2008", field.gm, field.radius_km, -field.C[2, 0])
    # j2.propagate's frame is any one sharing the rotation axis; od's rotates
    # into ITRF and back, where polar motion tilts the axis by 0.35"
    r_j2, _ = j2.propagate(
        e29_start[:3], e29_start[3:], t, earth=earth, third_body_start=od.PRECISE_START
    )
    no_pole = od.EarthOrientation(0.0, 0.0, od.PRECISE_EOP.ut1_minus_utc_s, "test")
    r_flat, _ = od.propagate(e29_start, t, field=field, substeps=1, eop=no_pole)
    assert np.abs(r_flat[:, 0] - r_j2).max() < 1e-8
    assert np.abs(r_od[:, 0] - r_j2).max() < 1e-3  # the pole's tilt: under 1 m


def test_a_60_s_step_is_far_below_a_metre(e29_start):
    t = np.arange(0.0, 86000.0, 300.0)
    field = gravity.load_field(degree=4)
    r60, _ = od.propagate(e29_start, t, field=field, substeps=5)
    r30, _ = od.propagate(e29_start, t, field=field, substeps=10)
    assert np.abs(r60 - r30).max() < 5e-5  # 5 cm in a day


def test_the_fit_recovers_a_known_orbit(e29_start):
    """Observations made by the model itself, with a known SRP: the fit finds
    the state and the SRP from a guess 100 m and 1 cm/s off."""
    t = np.arange(0.0, 12 * 3600 + 1, 900.0)
    field = gravity.load_field(degree=2)
    truth = np.array([0.13])
    r_obs, _ = od.propagate(e29_start, t, truth, "cannonball", field, substeps=15)
    guess = e29_start + np.array([0.1, -0.05, 0.05, 1e-5, 0, -1e-5])
    f = od.fit(t, r_obs[:, 0], guess, "cannonball", field, substeps=15)
    np.testing.assert_allclose(f.state[:3], e29_start[:3], atol=1e-5)
    np.testing.assert_allclose(f.srp_params, truth, atol=1e-4)
    assert f.rms_km < 1e-5
    assert f.iterations <= 4


def test_the_reported_rms_is_that_of_the_returned_state(results):
    """Fit.rms_km is the post-fit RMS of Fit.state and Fit.srp_params, not of
    the step before the last correction."""
    res = results[QZS2]
    f = res["fits"]["dyb"]
    t, in_fit = res["t"][res["in_fit"]], res["in_fit"]
    r_true, _ = od.observed(read_sp3(PRECISE_ORBITS), "J02", res["t"])
    r, _ = od.propagate(f.state, t, f.srp_params, "dyb")
    rms = np.sqrt(np.mean(np.sum((r[:, 0] - r_true[in_fit]) ** 2, axis=1)))
    assert f.rms_km == pytest.approx(rms, rel=1e-9)


def test_without_the_parameter_in_the_model_the_fit_says_so(e29_start):
    """The same observations fitted without SRP leave metres of residual."""
    t = np.arange(0.0, 12 * 3600 + 1, 900.0)
    field = gravity.load_field(degree=2)
    r_obs, _ = od.propagate(
        e29_start, t, np.array([0.13]), "cannonball", field, substeps=15
    )
    f = od.fit(t, r_obs[:, 0], e29_start, "none", field, substeps=15)
    assert f.rms_km > 1e-3
    assert f.srp_params.size == 0


# ---------- The real day ----------
def pred_rms(res, key):
    return od.rms(res[key]["total"][~res["in_fit"]])


def test_the_fit_sees_the_first_half_only(results):
    for res in results.values():
        h = res["t"] / 3600
        assert h[res["in_fit"]].max() == od.FIT_HOURS
        assert h[~res["in_fit"]].min() > od.FIT_HOURS
        assert h[-1] > 23.9


def test_fitted_srp_makes_the_prediction_several_times_better(results):
    for res in results.values():
        assert pred_rms(res, "fit_cannonball") * 5 < pred_rms(res, "fit_none")
        assert res["fits"]["cannonball"].rms_km < 0.003  # post-fit, 3 m
    assert pred_rms(results[GALILEO], "fit_cannonball") < 0.015
    assert pred_rms(results[QZS2], "fit_cannonball") < 0.06


def test_the_field_beyond_j2_matters_at_these_heights(results):
    for res in results.values():
        assert pred_rms(res, "egm") < pred_rms(res, "j2")


def test_galileos_fitted_srp_is_the_physical_size(results):
    """Fitted, the push is what public area and mass predict (within 25%)."""
    d = results[GALILEO]["fits"]["cannonball"].srp_params[0]
    assert d == pytest.approx(od.galileo_srp_estimate(), rel=0.25)
    for res in results.values():  # away from the Sun, the same size for both
        assert 0.08 < res["fits"]["cannonball"].srp_params[0] < 0.2
        dyb = res["fits"]["dyb"]
        assert (
            abs(dyb.srp_params[0] / res["fits"]["cannonball"].srp_params[0] - 1) < 0.1
        )
        assert np.all(np.abs(dyb.srp_params[1:]) < 0.3 * dyb.srp_params[0])


def test_without_the_field_srp_soaks_up_gravity(results, e29_start):
    """Fitted with J2 alone, the SRP parameter takes up the missing C22/S22
    pull: it comes out far too large and the prediction gets worse, not
    better, than the full model's."""
    res = results[GALILEO]
    t, in_fit = res["t"], res["in_fit"]
    field = gravity.load_field(degree=2)
    field.C[2, 1:] = 0.0
    field.S[2, 1:] = 0.0
    r_true, _ = od.observed(read_sp3(PRECISE_ORBITS), "E29", t)
    f = od.fit(t[in_fit], r_true[in_fit], e29_start, "cannonball", field)
    r, _ = od.propagate(f.state, t, f.srp_params, "cannonball", field)
    predicted = od.rms(np.linalg.norm(r[:, 0] - r_true, axis=1)[~in_fit])
    assert f.srp_params[0] > 1.6 * od.galileo_srp_estimate()
    assert predicted > pred_rms(res, "fit_none")
    assert predicted == pytest.approx(0.155, abs=0.005)


def test_summary_and_figure(results, tmp_path, monkeypatch):
    monkeypatch.setattr(od, "analyse", lambda: results)
    od.main(tmp_path)
    text = (tmp_path / "od.txt").read_text()
    assert "FIT 00-12 h, PREDICT 12-24 h" in text
    assert "Fitted to 00-12 h, cannonball SRP" in text
    assert (tmp_path / "od.png").stat().st_size > 50_000


def test_readme_numbers_are_the_analysis(results):
    """The README's orbit determination table is what od.py gives."""
    import re
    from pathlib import Path

    text = Path(__file__).parents[1].joinpath("README.md").read_text()
    for key, label in od.README_ROWS.items():
        row = re.search(rf"^\| {re.escape(label)} \|(.*)\|$", text, re.MULTILINE)
        assert row, label
        cells = [c.strip().strip("*") for c in row[1].split("|")]
        assert cells == [
            f"{pred_rms(results[n], key) * 1000:.0f} m" for n in (GALILEO, QZS2)
        ], label
    flat = " ".join(text.split())
    old = [pred_rms(results[n], "j2") for n in (GALILEO, QZS2)]
    assert (
        f"is {old[0]:.2f} km (GALILEO 29) and {old[1]:.2f} km (QZS-2)" in flat
    )  # the footnote
    assert f"is {old[0] * 1000:.0f} m and {old[1] * 1000:.0f} m over the same" in flat
    top = re.search(r"^\| Orbit determination: .*\|(.*)\|(.*)\|$", text, re.MULTILINE)
    assert [c.strip().strip("*") for c in top.groups()] == [
        f"{pred_rms(results[n], 'fit_cannonball'):.3f} km" for n in (GALILEO, QZS2)
    ]
