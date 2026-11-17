import json

import numpy as np
import pytest

from orbit_explorer.czml import CZML_STEP, build_czml
from orbit_explorer.elements import load_elements
from orbit_explorer.frames import ecef_to_geodetic, eci_to_ecef, gmst_rad
from orbit_explorer.kepler import SPAN_S, propagate_eci, time_grid_s

SATS = load_elements()
T = time_grid_s()[::CZML_STEP]


@pytest.fixture(scope="module")
def czml():
    return build_czml()


def packet(czml, packet_id):
    return next(p for p in czml if p["id"] == packet_id)


def samples(pkt):
    """CZML cartesian samples as (times [s], positions [km])."""
    a = np.array(pkt["position"]["cartesian"], dtype=float).reshape(-1, 4)
    return a[:, 0], a[:, 1:] / 1000


def test_document_packet_comes_first(czml):
    doc = czml[0]
    assert doc["id"] == "document"
    assert doc["version"] == "1.0"
    assert doc["clock"]["interval"] == "2026-11-10T00:00:00Z/2026-11-11T00:00:00Z"


def test_three_packets_per_satellite_with_unique_ids(czml):
    ids = [p["id"] for p in czml]
    assert len(ids) == 1 + 3 * len(SATS)
    assert len(set(ids)) == len(ids)


def test_is_plain_json(czml):
    """No numpy types may leak in, or json.dumps fails."""
    assert json.loads(json.dumps(czml)) == czml


@pytest.mark.parametrize("sat", SATS, ids=lambda s: s["OBJECT_NAME"])
def test_sample_times_cover_the_day(czml, sat):
    t, _ = samples(packet(czml, f"sat-{sat['NORAD_CAT_ID']}"))
    assert t[0] == 0 and t[-1] == SPAN_S
    assert np.all(np.diff(t) == 60)


@pytest.mark.parametrize("sat", SATS, ids=lambda s: s["OBJECT_NAME"])
def test_fixed_positions_match_frames(czml, sat):
    """Satellite positions are our own ECEF, to within the 1 m rounding."""
    pos_eci, _ = propagate_eci(sat, T)
    expected = eci_to_ecef(pos_eci, gmst_rad(T))
    pkt = packet(czml, f"sat-{sat['NORAD_CAT_ID']}")
    assert pkt["position"]["referenceFrame"] == "FIXED"
    _, pos = samples(pkt)
    np.testing.assert_allclose(pos, expected, atol=1e-3)


@pytest.mark.parametrize("sat", SATS, ids=lambda s: s["OBJECT_NAME"])
def test_inertial_positions_match_kepler(czml, sat):
    pos_eci, _ = propagate_eci(sat, T)
    pkt = packet(czml, f"orbit-{sat['NORAD_CAT_ID']}")
    assert pkt["position"]["referenceFrame"] == "INERTIAL"
    _, pos = samples(pkt)
    np.testing.assert_allclose(pos, pos_eci, atol=1e-3)


@pytest.mark.parametrize("sat", SATS, ids=lambda s: s["OBJECT_NAME"])
def test_ground_track_matches_geodetic(czml, sat):
    pos_eci, _ = propagate_eci(sat, T)
    lat, lon, _ = ecef_to_geodetic(eci_to_ecef(pos_eci, gmst_rad(T)))
    pkt = packet(czml, f"track-{sat['NORAD_CAT_ID']}")
    coords = np.array(pkt["polyline"]["positions"]["cartographicDegrees"])
    coords = coords.reshape(-1, 3)
    np.testing.assert_allclose(coords[:, 0], lon, atol=1e-4)
    np.testing.assert_allclose(coords[:, 1], lat, atol=1e-4)
    assert np.all(coords[:, 2] == 0)


def test_geostationary_satellite_barely_moves(czml):
    _, pos = samples(packet(czml, "sat-38652"))  # SES-5
    drift_km = np.linalg.norm(pos - pos.mean(axis=0), axis=1).max()
    assert drift_km < 50
