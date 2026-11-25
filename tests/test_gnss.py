"""The GNSS constellation snapshot: checks on a download, storing, loading."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from orbit_explorer import gnss


@pytest.fixture
def downloads():
    systems, _ = gnss.load_snapshot()
    return systems


def test_round_trip(tmp_path, downloads):
    when = datetime(2026, 11, 25, 11, 7, 49, tzinfo=UTC)
    gnss.write_snapshot(downloads, when, tmp_path)
    systems, start = gnss.load_snapshot(tmp_path)
    assert systems == downloads
    assert start == datetime(2026, 11, 25, tzinfo=UTC)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == [
        "gp_galileo_2026-11-25.json",
        "gp_gps-ops_2026-11-25.json",
        "manifest.json",
    ]
    assert "2026-11-25 11:07" in gnss.summary(tmp_path).replace("T", " ")


def test_a_new_snapshot_replaces_the_old_files(tmp_path, downloads):
    gnss.write_snapshot(downloads, datetime(2026, 11, 25, 12, tzinfo=UTC), tmp_path)
    gnss.write_snapshot(downloads, datetime(2026, 12, 2, 12, tzinfo=UTC), tmp_path)
    assert sorted(p.name for p in tmp_path.glob("gp_*.json")) == [
        "gp_galileo_2026-12-02.json",
        "gp_gps-ops_2026-12-02.json",
    ]


@pytest.mark.parametrize(
    "spoil, message",
    [
        (lambda s: s[:5], "5 records"),
        (lambda s: s + s[:1], "listed twice"),
        (lambda s: [{**s[0], "MEAN_MOTION": 15.2}] + s[1:], "mean motion"),
        (lambda s: {"error": "no data"}, "no records"),
    ],
)
def test_a_broken_download_is_refused(downloads, spoil, message):
    with pytest.raises(gnss.SnapshotError, match=message):
        gnss.check_group("GPS", spoil(downloads["GPS"]))


def test_galileos_eccentric_pair_is_accepted(downloads):
    """GSAT0201/0202, launched into the wrong orbit, orbit 1.86 times a day."""
    names = {s["OBJECT_NAME"]: s["MEAN_MOTION"] for s in downloads["Galileo"]}
    eccentric = [n for n in names if n.startswith(("GSAT0201", "GSAT0202"))]
    assert len(eccentric) == 2
    assert all(names[n] > 1.8 for n in eccentric)


def test_no_download_within_two_hours(tmp_path, downloads, monkeypatch):
    when = datetime(2026, 11, 25, 11, tzinfo=UTC)
    gnss.write_snapshot(downloads, when, tmp_path)

    def no_network(url):
        raise AssertionError("downloaded although too soon")

    monkeypatch.setattr(gnss, "fetch_text", no_network)
    with pytest.raises(gnss.SnapshotError, match="CelesTrak asks"):
        gnss.fetch_snapshot(tmp_path, when + timedelta(hours=1, minutes=59))


def test_fetch_checks_before_writing(tmp_path, downloads, monkeypatch):
    """A bad group leaves the stored snapshot as it was."""
    when = datetime(2026, 11, 25, 11, tzinfo=UTC)
    gnss.write_snapshot(downloads, when, tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    answers = iter([json.dumps(downloads["GPS"]), "No GP data found"])
    monkeypatch.setattr(gnss, "fetch_text", lambda url: next(answers))
    with pytest.raises(gnss.SnapshotError, match="No GP data"):
        gnss.fetch_snapshot(tmp_path, when + timedelta(hours=3))
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_committed_snapshot_epochs_are_near_its_day():
    systems, start = gnss.load_snapshot()
    for sats in systems.values():
        for s in sats:
            epoch = datetime.fromisoformat(s["EPOCH"] + "+00:00")
            assert abs((epoch - start).total_seconds()) < 4 * 86400, s["OBJECT_NAME"]
