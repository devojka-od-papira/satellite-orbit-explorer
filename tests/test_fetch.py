"""fetch.py with CelesTrak replaced by canned responses: no network in tests."""

import json
import shutil
from datetime import UTC, datetime

import pytest

from orbit_explorer import fetch
from orbit_explorer.elements import DATA_PATH, MANIFEST_PATH, load_elements

NOW = datetime(2026, 12, 1, 9, 30, 15, tzinfo=UTC)


def fake_record(sat: dict, epoch: str) -> dict:
    return {**sat, "EPOCH": epoch}


def fake_tle(norad: int) -> str:
    return (
        "SAT NAME\n"
        f"1 {norad:05d}U 00000A   26335.00000000  .00000000  00000+0  00000+0 0  9990\n"
        f"2 {norad:05d}   0.0000   0.0000 0000000   0.0000   0.0000  1.00000000    00\n"
    )


@pytest.fixture
def data_copy(tmp_path):
    data = tmp_path / "orbital_elements.json"
    manifest = tmp_path / "data_manifest.json"
    shutil.copy(DATA_PATH, data)
    shutil.copy(MANIFEST_PATH, manifest)
    return data, manifest


@pytest.fixture
def celestrak(monkeypatch):
    """Serve a fresh record per satellite; tests can override per NORAD id."""
    sats = {s["NORAD_CAT_ID"]: s for s in load_elements()}
    responses = {}
    for norad, sat in sats.items():
        record = fake_record(sat, "2026-12-01T03:00:00.000000")
        responses[fetch.GP_URL.format(norad=norad, fmt="JSON")] = json.dumps([record])
        responses[fetch.GP_URL.format(norad=norad, fmt="TLE")] = fake_tle(norad)
    monkeypatch.setattr(fetch, "fetch_text", lambda url: responses[url])
    return responses


def test_refresh_writes_elements_and_manifest(data_copy, celestrak):
    data, manifest_path = data_copy
    before = [s["NORAD_CAT_ID"] for s in load_elements(data)]
    manifest = fetch.refresh(data, manifest_path, now=NOW)

    after = load_elements(data)
    assert [s["NORAD_CAT_ID"] for s in after] == before  # same satellites, same order
    assert all(s["EPOCH"] == "2026-12-01T03:00:00.000000" for s in after)
    assert json.loads(manifest_path.read_text()) == manifest
    assert manifest["reference_start_utc"] == "2026-12-01T00:00:00Z"
    assert manifest["retrieved_utc"] == "2026-12-01T09:30:15Z"
    assert len(manifest["source_records"]) == len(before)
    assert all(
        len(r["tle_lines_same_element_set"]) == 2 for r in manifest["source_records"]
    )
    assert not manifest["known_limitations"][0].startswith("Epoch more than")


def test_stale_epoch_is_flagged(data_copy, celestrak):
    data, manifest_path = data_copy
    sat = load_elements(data)[-1]
    url = fetch.GP_URL.format(norad=sat["NORAD_CAT_ID"], fmt="JSON")
    celestrak[url] = json.dumps([fake_record(sat, "2026-11-20T00:00:00.000000")])
    manifest = fetch.refresh(data, manifest_path, now=NOW)
    assert manifest["known_limitations"][0].startswith("Epoch more than 3 days")
    assert sat["OBJECT_NAME"] in manifest["known_limitations"][0]


@pytest.mark.parametrize(
    "bad_json",
    ["No GP data found", "[]", json.dumps([{"NORAD_CAT_ID": 1}])],
    ids=["plain text", "empty", "wrong satellite"],
)
def test_bad_response_leaves_files_untouched(data_copy, celestrak, bad_json):
    data, manifest_path = data_copy
    data_before, manifest_before = data.read_text(), manifest_path.read_text()
    norad = load_elements(data)[-1]["NORAD_CAT_ID"]  # the last one fails
    celestrak[fetch.GP_URL.format(norad=norad, fmt="JSON")] = bad_json
    with pytest.raises(RuntimeError):
        fetch.refresh(data, manifest_path, now=NOW)
    assert data.read_text() == data_before
    assert manifest_path.read_text() == manifest_before


def test_tle_for_another_satellite_is_rejected(data_copy, celestrak):
    data, manifest_path = data_copy
    norad = load_elements(data)[0]["NORAD_CAT_ID"]
    celestrak[fetch.GP_URL.format(norad=norad, fmt="TLE")] = fake_tle(99999)
    with pytest.raises(RuntimeError, match="Unexpected TLE"):
        fetch.refresh(data, manifest_path, now=NOW)


def test_committed_manifest_matches_elements():
    """The manifest describes exactly the records in orbital_elements.json."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    records = load_elements()
    assert [r["norad_cat_id"] for r in manifest["source_records"]] == [
        s["NORAD_CAT_ID"] for s in records
    ]
    assert [r["epoch_utc"] for r in manifest["source_records"]] == [
        f"{s['EPOCH']}Z" for s in records
    ]
    start = datetime.fromisoformat(manifest["reference_start_utc"])
    assert (start.hour, start.minute, start.second) == (0, 0, 0)
