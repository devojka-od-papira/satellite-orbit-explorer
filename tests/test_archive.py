"""The element set archive: unit tests in a scratch folder, and checks of the
archive committed in data/archive/."""

import copy
import json
from datetime import UTC, datetime

import pytest
from conftest import FROZEN_GP_PATH
from sgp4.api import Satrec

from orbit_explorer import archive
from orbit_explorer.elements import DATA_PATH, MANIFEST_PATH, load_elements


@pytest.fixture
def rows():
    return archive.rows_from_download(DATA_PATH, MANIFEST_PATH)


def at_epoch(row: dict, epoch: str, retrieved: str | None = None) -> dict:
    """The same element set moved to another epoch (and retrieval time)."""
    new = copy.deepcopy(row)
    new["gp"]["EPOCH"] = epoch
    new["epoch_utc"] = epoch + "Z"
    if retrieved:
        new["retrieved_utc"] = retrieved
    return new


def epochs(path) -> list[str]:
    return [r["epoch_utc"] for r in archive.read_file(path)]


# ---------- Adding ----------
def test_one_file_per_satellite(tmp_path, rows):
    update = archive.add_rows(rows, tmp_path)
    assert update.added == rows and len(rows) == 4 and update.conflicts == []
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        f"{s['NORAD_CAT_ID']}.jsonl" for s in load_elements()
    )
    for row in rows:
        (stored,) = archive.read_file(
            archive.archive_path(row["norad_cat_id"], tmp_path)
        )
        assert stored == row


def test_same_download_twice_adds_nothing(tmp_path, rows):
    archive.add_rows(rows, tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert archive.add_rows(rows, tmp_path) == archive.Update()
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_kept_in_epoch_order(tmp_path, rows):
    row = rows[0]
    path = archive.archive_path(row["norad_cat_id"], tmp_path)
    archive.add_rows([at_epoch(row, "2026-11-20T00:00:00")], tmp_path)
    archive.add_rows([at_epoch(row, "2026-11-10T00:00:00")], tmp_path)
    archive.add_rows([at_epoch(row, "2026-11-15T00:00:00")], tmp_path)
    assert epochs(path) == [
        "2026-11-10T00:00:00Z",
        "2026-11-15T00:00:00Z",
        "2026-11-20T00:00:00Z",
    ]
    assert not list(tmp_path.glob("*.tmp"))


def test_first_retrieval_of_an_element_set_is_kept(tmp_path, rows):
    """CelesTrak serves an element set until a newer one: a later download of
    the same epoch is the same set, and the first retrieval stays."""
    epoch = "2026-11-16T09:41:16.391616"
    first = at_epoch(rows[0], epoch, "2026-11-17T18:20:03Z")
    later = at_epoch(rows[0], epoch, "2026-11-19T10:04:32Z")
    archive.add_rows([first], tmp_path)
    assert archive.add_rows([later], tmp_path) == archive.Update()
    # within one batch too, whatever the order
    other = tmp_path / "other"
    assert archive.add_rows([later, first], other) == archive.Update([first])


def test_a_different_set_with_a_known_epoch_is_reported(tmp_path, rows):
    """Same epoch, different elements (a re-issued set): the first stays,
    but not silently."""
    epoch = "2026-11-16T09:41:16.391616"
    first = at_epoch(rows[0], epoch, "2026-11-17T18:20:03Z")
    corrected = at_epoch(rows[0], epoch, "2026-11-19T10:04:32Z")
    corrected["gp"]["MEAN_MOTION"] += 1e-6
    archive.add_rows([first], tmp_path)

    update = archive.add_rows([corrected], tmp_path)
    assert update == archive.Update([], [(first, corrected)])
    path = archive.archive_path(first["norad_cat_id"], tmp_path)
    assert archive.read_file(path) == [first]
    text = archive.describe(update)
    assert "no new element sets" in text
    assert "Warning:" in text and "different elements" in text
    assert "retrieved 2026-11-19T10:04:32Z" in text


def test_merge_carries_another_archive_over(tmp_path, rows):
    branch, main = tmp_path / "branch", tmp_path / "main"
    archive.add_rows([at_epoch(r, "2026-11-20T00:00:00") for r in rows], branch)
    archive.add_rows(rows, main)
    assert len(archive.merge(branch, main).added) == 4
    assert archive.merge(branch, main) == archive.Update()
    for row in rows:
        assert (
            len(archive.read_file(archive.archive_path(row["norad_cat_id"], main))) == 2
        )


# ---------- What is refused ----------
@pytest.mark.parametrize(
    "spoil, message",
    [
        (lambda r: r["gp"].pop("MEAN_MOTION"), "without MEAN_MOTION"),
        (lambda r: r.update(norad_cat_id=1), "NORAD id 1"),
        (lambda r: r.update(epoch_utc="2026-01-01T00:00:00Z"), "epoch"),
        (lambda r: r.update(tle=r["tle"][:1]), "TLE"),
        (lambda r: r.update(tle=[r["tle"][1], r["tle"][0]]), "TLE"),
        (lambda r: r.update(retrieved_utc="yesterday"), "incomplete"),
        (lambda r: r.pop("gp"), "incomplete"),
    ],
)
def test_bad_rows_are_refused(tmp_path, rows, spoil, message):
    row = copy.deepcopy(rows[0])
    spoil(row)
    with pytest.raises(archive.ArchiveError, match=message):
        archive.add_rows([row], tmp_path)
    assert not tmp_path.exists() or not list(tmp_path.iterdir())


def test_a_bad_line_stops_a_write_but_not_an_analysis(tmp_path, rows):
    row = rows[0]
    path = archive.archive_path(row["norad_cat_id"], tmp_path)
    archive.add_rows([row], tmp_path)
    with path.open("a") as f:
        f.write('{"truncated": \n')
    damaged = path.read_bytes()

    # a rewrite would drop the line: refused, file left as it was
    with pytest.raises(archive.ArchiveError, match=f"{path.name}:2"):
        archive.add_rows([at_epoch(row, "2026-12-01T00:00:00")], tmp_path)
    assert path.read_bytes() == damaged

    with pytest.warns(UserWarning, match=f"skipped .*{path.name}:2"):
        assert archive.load_history(row["norad_cat_id"], tmp_path) == [row]


def test_a_row_in_the_wrong_file_is_refused(tmp_path, rows):
    (tmp_path / "1.jsonl").write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(archive.ArchiveError, match="satellite"):
        archive.read_file(tmp_path / "1.jsonl")


def test_download_with_a_satellite_missing_from_the_manifest(tmp_path):
    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest["source_records"] = manifest["source_records"][1:]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(archive.ArchiveError, match="not in the manifest"):
        archive.rows_from_download(DATA_PATH, path)


# ---------- Command line ----------
def test_cli_add_and_summary(tmp_path, capsys):
    assert archive.main(["--archive", str(tmp_path), "add"]) == 0
    assert "Archive: added" in capsys.readouterr().out
    assert archive.main(["--archive", str(tmp_path), "add"]) == 0
    assert "no new element sets" in capsys.readouterr().out

    assert archive.main(["--archive", str(tmp_path), "summary"]) == 0
    out = capsys.readouterr().out
    assert out.count("1 element sets") == 4


def test_cli_reports_a_damaged_archive(tmp_path, rows, capsys):
    path = archive.archive_path(rows[0]["norad_cat_id"], tmp_path)
    tmp_path.joinpath(path.name).write_text("not json\n")
    assert archive.main(["--archive", str(tmp_path), "add"]) == 1
    assert f"{path.name}:1" in capsys.readouterr().err


# ---------- The committed archive ----------
COMMITTED = sorted(archive.ARCHIVE_DIR.glob("*.jsonl"))


def test_committed_archive_covers_every_satellite():
    assert {int(p.stem) for p in COMMITTED} == {
        s["NORAD_CAT_ID"] for s in load_elements()
    }


@pytest.mark.parametrize("path", COMMITTED, ids=lambda p: p.stem)
def test_committed_archive_is_valid_and_ordered(path):
    rows = archive.read_file(path, strict=True)  # raises on any bad line
    stamps = [archive.epoch_of(r) for r in rows]
    assert stamps == sorted(set(stamps)), "out of order or repeated epochs"
    for row in rows:
        # the TLE is the same element set as the GP record (TLE epochs have
        # 1e-8 day, under a millisecond)
        tle_epoch = Satrec.twoline2rv(*row["tle"])
        jd = tle_epoch.jdsatepoch + tle_epoch.jdsatepochF
        gp_jd = archive.epoch_of(row).timestamp() / 86400 + 2440587.5
        assert abs(jd - gp_jd) * 86400 < 1e-3
        assert archive.epoch_of(row) <= datetime.fromisoformat(row["retrieved_utc"])


def test_current_data_is_archived(rows):
    """The refresh archives each download before the tests run."""
    for row in rows:
        stored = archive.load_history(row["norad_cat_id"])
        assert row["epoch_utc"] in [r["epoch_utc"] for r in stored], (
            f"{row['object_name']} {row['epoch_utc']} is not in the archive: "
            "run python -m orbit_explorer.archive add"
        )


def test_archive_holds_the_frozen_day():
    """The GP data the README figures use (2026-11-17) is in the history."""
    for gp in load_elements(FROZEN_GP_PATH):
        stored = {
            r["epoch_utc"]: r["gp"] for r in archive.load_history(gp["NORAD_CAT_ID"])
        }
        assert stored[gp["EPOCH"] + "Z"] == gp


def test_history_starts_before_the_refresh():
    """Back-filled from the first committed data (2026-11-10)."""
    first = min(
        archive.epoch_of(archive.load_history(int(p.stem))[0]) for p in COMMITTED
    )
    assert first < datetime(2026, 11, 11, tzinfo=UTC)
