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


# ---------- Satellites set unusable ----------
def test_unusable_satellites_are_left_out():
    systems, _ = gnss.load_snapshot()
    kept, _, excluded = gnss.usable()
    names = {e["name"] for e in excluded["Galileo"]}
    assert names == {"GSAT0201", "GSAT0202", "GSAT0204", "GSAT0209", "GSAT0210"}
    assert len(kept["Galileo"]) == len(systems["Galileo"]) - 5
    assert not {s["OBJECT_NAME"].split()[0] for s in kept["Galileo"]} & names
    assert kept["GPS"] == systems["GPS"]
    for entry in excluded["Galileo"]:  # each says why, where from and when
        assert entry["reason"].startswith("NOT USABLE, NAGU ")
        assert entry["source"].startswith("https://www.gsc-europa.eu/")
        datetime.fromisoformat(entry["checked"])


def test_a_new_snapshot_keeps_the_exclusions(tmp_path, downloads):
    excluded = {
        "Galileo": [
            {
                "name": "GSAT0204",
                "sv": "E22",
                "reason": "x",
                "source": "y",
                "checked": "2026-11-25",
            }
        ]
    }
    gnss.write_snapshot(
        downloads, datetime(2026, 11, 25, tzinfo=UTC), tmp_path, excluded
    )
    gnss.write_snapshot(downloads, datetime(2026, 12, 2, tzinfo=UTC), tmp_path)
    kept, _, kept_excluded = gnss.usable(tmp_path)
    assert kept_excluded == excluded
    assert len(kept["Galileo"]) == len(downloads["Galileo"]) - 1
    assert "1 excluded" in gnss.summary(tmp_path)
    assert "checked on 2026-11-25" in gnss.summary(tmp_path)


def test_an_exclusion_must_name_one_satellite(tmp_path, downloads):
    """A typo, or a satellite gone from the list, is caught."""
    excluded = {
        "Galileo": [
            {
                "name": "GSAT9999",
                "sv": "E99",
                "reason": "x",
                "source": "y",
                "checked": "2026-11-25",
            }
        ]
    }
    with pytest.raises(gnss.SnapshotError, match="GSAT9999 matches 0"):
        gnss.write_snapshot(
            downloads, datetime(2026, 11, 25, tzinfo=UTC), tmp_path, excluded
        )
    assert not tmp_path.exists() or not list(tmp_path.iterdir())


def test_an_interrupted_write_leaves_the_old_snapshot(tmp_path, downloads, monkeypatch):
    old_day = datetime(2026, 11, 25, 11, tzinfo=UTC)
    gnss.write_snapshot(downloads, old_day, tmp_path)
    real_replace = type(tmp_path).replace

    def fail_on_manifest(self, target):
        if self.name == "manifest.json.tmp":
            raise OSError("disk full")
        return real_replace(self, target)

    monkeypatch.setattr(type(tmp_path), "replace", fail_on_manifest)
    with pytest.raises(OSError, match="disk full"):
        gnss.write_snapshot(downloads, old_day + timedelta(days=7), tmp_path)
    monkeypatch.undo()
    systems, start = gnss.load_snapshot(tmp_path)  # the old one, whole
    assert start == datetime(2026, 11, 25, tzinfo=UTC)
    assert systems == downloads
