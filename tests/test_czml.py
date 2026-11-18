import json
from datetime import timedelta

import numpy as np
import pytest

from orbit_explorer.czml import CZML_STEP, build_czml, visibility_radius_deg
from orbit_explorer.elements import load_elements
from orbit_explorer.frames import ecef_to_geodetic, eci_to_ecef, gmst_rad
from orbit_explorer.kepler import SPAN_S, START, propagate_eci, time_grid_s
from orbit_explorer.sgp4_compare import compare, propagate_sgp4
from orbit_explorer.visibility import (
    BELGRADE,
    STATIONS,
    find_passes,
    satellite_look_angles,
)

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
    end = START + timedelta(seconds=SPAN_S)
    assert doc["clock"]["interval"] == (
        f"{START:%Y-%m-%dT%H:%M:%SZ}/{end:%Y-%m-%dT%H:%M:%SZ}"
    )


def test_packets_are_complete_with_unique_ids(czml):
    """Document, five packets per satellite, the stations, their rings and
    links, and nothing else."""
    ids = [p["id"] for p in czml]
    assert len(set(ids)) == len(ids)
    for sat in SATS:
        norad = sat["NORAD_CAT_ID"]
        for prefix in ("sat", "track", "orbit", "sgp4", "sgp4orbit"):
            assert f"{prefix}-{norad}" in ids
    rings = [i for i in ids if i.startswith("ring-")]
    links = [i for i in ids if i.startswith("link-")]
    assert len(ids) == 1 + 5 * len(SATS) + len(STATIONS) + len(rings) + len(links)


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


@pytest.mark.parametrize("sat", SATS, ids=lambda s: s["OBJECT_NAME"])
def test_sgp4_packets_match_sgp4(czml, sat):
    r_teme, _ = propagate_sgp4(sat, T)
    norad = sat["NORAD_CAT_ID"]

    fixed = packet(czml, f"sgp4-{norad}")
    assert fixed["position"]["referenceFrame"] == "FIXED"
    np.testing.assert_allclose(
        samples(fixed)[1], eci_to_ecef(r_teme, gmst_rad(T)), atol=1e-3
    )

    inertial = packet(czml, f"sgp4orbit-{norad}")
    assert inertial["position"]["referenceFrame"] == "INERTIAL"
    np.testing.assert_allclose(samples(inertial)[1], r_teme, atol=1e-3)


@pytest.mark.parametrize("sat", SATS, ids=lambda s: s["OBJECT_NAME"])
def test_kepler_sgp4_distance_matches_comparison(czml, sat):
    """The gap between the two dots in the viewer is the one in sgp4_compare."""
    norad = sat["NORAD_CAT_ID"]
    _, kepler = samples(packet(czml, f"sat-{norad}"))
    _, sgp4 = samples(packet(czml, f"sgp4-{norad}"))
    np.testing.assert_allclose(
        np.linalg.norm(kepler - sgp4, axis=1), compare(sat, T)["total"], atol=2e-3
    )


def test_geostationary_satellite_barely_moves(czml):
    _, pos = samples(packet(czml, "sat-38652"))  # SES-5
    drift_km = np.linalg.norm(pos - pos.mean(axis=0), axis=1).max()
    assert drift_km < 50


# ---------- Ground stations ----------
def station_passes(czml, station=BELGRADE):
    return json.loads(packet(czml, f"station-{station.slug}")["properties"]["passes"])


STATION_IDS = [st.slug for st in STATIONS]


def test_stations_in_viewer_order(czml):
    stations = [p for p in czml if p["id"].startswith("station-")]
    assert [p["id"] for p in stations] == [f"station-{s}" for s in STATION_IDS]
    assert [p["properties"]["order"] for p in stations] == list(range(len(STATIONS)))
    assert STATIONS[0] == BELGRADE  # the viewer opens on the first one


@pytest.mark.parametrize("station", STATIONS, ids=STATION_IDS)
def test_station_position(czml, station):
    pkt = packet(czml, f"station-{station.slug}")
    lon, lat, h = pkt["position"]["cartographicDegrees"]
    assert (lat, lon) == (station.lat_deg, station.lon_deg)
    assert h == pytest.approx(station.height_km * 1000)


@pytest.mark.parametrize("station", STATIONS, ids=STATION_IDS)
def test_station_passes_match_visibility(czml, station):
    """The panel lists exactly the passes visibility.py finds for that station."""
    t = time_grid_s()
    expected = []
    for sat in SATS:
        az, el, _ = satellite_look_angles(sat, t, station)
        expected += [(sat["NORAD_CAT_ID"], p) for p in find_passes(t, el, az)]
    passes = station_passes(czml, station)
    assert len(passes) == len(expected)
    assert [p["rise"] for p in passes] == sorted(p["rise"] for p in passes)
    for norad, p in expected:
        rise = START + timedelta(seconds=round(p.rise_s))
        match = [
            q
            for q in passes
            if q["norad"] == norad and q["rise"] == f"{rise:%Y-%m-%dT%H:%M:%SZ}"
        ]
        assert len(match) == 1
        assert match[0]["maxEl"] == pytest.approx(p.max_el_deg, abs=0.05)


@pytest.mark.parametrize("station", STATIONS, ids=STATION_IDS)
def test_links_point_from_station_to_sgp4_during_passes(czml, station):
    ids = {p["id"] for p in czml}
    passes = station_passes(czml, station)
    prefix = f"link-{station.slug}-"
    links = [p for p in czml if p["id"].startswith(prefix)]
    assert {int(lk["id"][len(prefix) :]) for lk in links} == {
        p["norad"] for p in passes
    }
    for link in links:
        norad = int(link["id"][len(prefix) :])
        refs = link["polyline"]["positions"]["references"]
        assert refs == [f"station-{station.slug}#position", f"sgp4-{norad}#position"]
        assert refs[1].split("#")[0] in ids
        intervals = [f"{p['rise']}/{p['set']}" for p in passes if p["norad"] == norad]
        assert link["availability"] == intervals  # same second as the pass list


def test_qzss_is_seen_from_tokyo_but_not_belgrade(czml):
    """The reason to offer more than one station."""
    ids = {p["id"] for p in czml}
    assert "link-belgrade-42738" not in ids
    assert "link-tokyo-42738" in ids


def test_visibility_ring_only_for_leo(czml):
    rings = [p["id"] for p in czml if p["id"].startswith("ring-")]
    assert rings == [f"ring-{s}-64541" for s in STATION_IDS]  # PULSAR-0


def test_visibility_radius():
    r = 6378.137
    # a 90° mask leaves only the point straight overhead; 0° is the horizon
    assert visibility_radius_deg(r + 500, 90) == pytest.approx(0, abs=1e-9)
    assert visibility_radius_deg(r + 500, 0) == pytest.approx(
        np.degrees(np.arccos(r / (r + 500))), rel=1e-9
    )
    # PULSAR-0 at ~505 km and a 10° mask: about 14° of arc, ~1,570 km
    assert np.radians(visibility_radius_deg(r + 505, 10)) * r == pytest.approx(
        1570, abs=20
    )


@pytest.mark.parametrize("sat", SATS, ids=lambda s: s["OBJECT_NAME"])
def test_label_uses_the_short_name(czml, sat):
    """The globe label (and so the viewer's satellite list) uses the short name
    from style.py, e.g. 'GALILEO 29', not 'GSAT0225'."""
    from orbit_explorer.style import short_name

    label = packet(czml, f"sat-{sat['NORAD_CAT_ID']}")["label"]["text"]
    assert label == short_name(sat)


def test_pass_list_uses_the_short_names(czml):
    passes = station_passes(czml)
    assert passes  # the check below must have something to check
    labels = {
        int(p["id"].split("-")[1]): p["label"]["text"]
        for p in czml
        if p["id"].startswith("sat-")
    }
    for p in passes:
        assert p["name"] == labels[p["norad"]]


def test_station_packets_reject_positions_on_another_grid():
    from orbit_explorer.czml import station_packets

    sat = SATS[0]
    coarse = {sat["NORAD_CAT_ID"]: np.zeros((len(T), 3))}  # the CZML grid, not 5 s
    with pytest.raises(ValueError, match="full time grid"):
        station_packets(BELGRADE, [sat], coarse, 0)
