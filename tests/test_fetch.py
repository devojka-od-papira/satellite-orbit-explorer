"""fetch.py with CelesTrak replaced by canned responses: no network in tests."""

import json
import shutil
from datetime import UTC, datetime

import pytest

from orbit_explorer import archive, fetch
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


def write_retrieved(path, when: str):
    m = json.loads(path.read_text())
    m["retrieved_utc"] = when
    path.write_text(json.dumps(m))


@pytest.fixture
def data_copy(tmp_path):
    data = tmp_path / "orbital_elements.json"
    manifest = tmp_path / "data_manifest.json"
    shutil.copy(DATA_PATH, data)
    shutil.copy(MANIFEST_PATH, manifest)
    # well before NOW, whatever date the committed data has
    write_retrieved(manifest, "2026-11-01T00:00:00Z")
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


# ---------- CelesTrak rate limit ----------
def test_download_within_two_hours_is_refused(data_copy, celestrak):
    data, manifest_path = data_copy
    write_retrieved(manifest_path, "2026-12-01T08:00:00Z")  # 1.5 h before NOW
    before = data.read_text()
    with pytest.raises(fetch.TooSoon, match="1.5 h ago"):
        fetch.refresh(data, manifest_path, now=NOW)
    assert data.read_text() == before
    fetch.refresh(data, manifest_path, now=NOW, force=True)  # --force overrides
    assert data.read_text() != before


def test_download_on_an_unmerged_branch_counts(data_copy, celestrak, tmp_path):
    """The refresh branch's manifest is newer than main's until the PR merges."""
    data, manifest_path = data_copy
    branch = tmp_path / "branch_manifest.json"
    shutil.copy(manifest_path, branch)
    write_retrieved(branch, "2026-12-01T09:00:00Z")
    with pytest.raises(fetch.TooSoon):
        fetch.refresh(data, manifest_path, now=NOW, seen_manifests=(branch,))
    missing = tmp_path / "nope.json"  # a branch that doesn't exist is ignored
    fetch.refresh(data, manifest_path, now=NOW, seen_manifests=(missing,))


def test_cli_exit_code_when_too_soon(data_copy, celestrak, capsys):
    data, manifest_path = data_copy
    write_retrieved(manifest_path, datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
    assert fetch.main([], data, manifest_path) == fetch.TOO_SOON_EXIT_CODE
    assert "Nothing downloaded" in capsys.readouterr().out


def test_cli_archives_the_download(data_copy, celestrak, tmp_path, capsys):
    """python -m orbit_explorer.fetch keeps the history: the new element sets
    go into the archive, unless --no-archive."""
    data, manifest_path = data_copy
    skipped = tmp_path / "skipped"
    assert fetch.main(["--no-archive"], data, manifest_path, skipped) == 0
    assert not skipped.exists()

    write_retrieved(manifest_path, "2026-11-01T00:00:00Z")  # past the rate limit
    store = tmp_path / "archive"
    assert fetch.main([], data, manifest_path, store) == 0
    assert "Archive: added" in capsys.readouterr().out
    for sat in load_elements(data):
        (row,) = archive.load_history(sat["NORAD_CAT_ID"], store)
        assert row["gp"] == sat


# ---------- Several runs a day (the workflow runs every 6 hours) ----------
@pytest.mark.parametrize("hour", [0, 6, 12, 18])
def test_every_run_of_a_day_keeps_its_window(data_copy, celestrak, hour):
    data, manifest_path = data_copy
    now = NOW.replace(hour=hour, minute=17)
    manifest = fetch.refresh(data, manifest_path, now=now)
    assert manifest["reference_start_utc"] == "2026-12-01T00:00:00Z"


def test_a_run_with_nothing_new_is_the_same_refresh(data_copy, celestrak, tmp_path):
    data, manifest_path = data_copy
    fetch.refresh(data, manifest_path, now=NOW.replace(hour=6))
    old_data, old_manifest = tmp_path / "old.json", tmp_path / "old_manifest.json"
    shutil.copy(data, old_data)
    shutil.copy(manifest_path, old_manifest)

    # 6 hours later CelesTrak still serves the same sets: nothing to commit
    fetch.refresh(data, manifest_path, now=NOW.replace(hour=12))
    assert (
        json.loads(manifest_path.read_text())["retrieved_utc"]
        != json.loads(old_manifest.read_text())["retrieved_utc"]
    )
    assert fetch.same_refresh(old_data, old_manifest, data, manifest_path)

    # a new element set is worth a commit
    norad = load_elements(data)[0]["NORAD_CAT_ID"]
    url = fetch.GP_URL.format(norad=norad, fmt="JSON")
    record = json.loads(celestrak[url])[0]
    celestrak[url] = json.dumps([{**record, "EPOCH": "2026-12-01T15:00:00.000000"}])
    fetch.refresh(data, manifest_path, now=NOW.replace(hour=18))
    assert not fetch.same_refresh(old_data, old_manifest, data, manifest_path)


def test_the_same_sets_on_a_new_day_move_the_window(data_copy, celestrak, tmp_path):
    """After midnight the viewer must move to the new day, new sets or not."""
    data, manifest_path = data_copy
    fetch.refresh(data, manifest_path, now=NOW.replace(hour=18))
    old_data, old_manifest = tmp_path / "old.json", tmp_path / "old_manifest.json"
    shutil.copy(data, old_data)
    shutil.copy(manifest_path, old_manifest)
    fetch.refresh(data, manifest_path, now=datetime(2026, 12, 2, 0, 17, tzinfo=UTC))
    assert load_elements(data) == load_elements(old_data)
    assert not fetch.same_refresh(old_data, old_manifest, data, manifest_path)


def test_stale_records_from_manifest():
    manifest = {
        "reference_start_utc": "2026-12-01T00:00:00Z",
        "source_records": [
            {"object_name": "FRESH", "epoch_utc": "2026-11-30T12:00:00Z"},
            {"object_name": "OLD", "epoch_utc": "2026-11-20T00:00:00Z"},
        ],
    }
    assert fetch.stale_records(manifest) == ["OLD (+11.0 d)"]
