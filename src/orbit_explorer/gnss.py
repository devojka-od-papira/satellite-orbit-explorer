"""A snapshot of whole GNSS constellations: CelesTrak GP data for GPS and Galileo.

The rest of the project follows four satellites; DOP (orbit_explorer.dop)
needs every satellite a receiver could track. This keeps one download of
each constellation in data/inputs/gnss/, pinned like the precise orbits in
data/inputs/precise/, so the DOP figures don't change with every refresh:

    gp_gps-ops_<date>.json, gp_galileo_<date>.json   CelesTrak GP JSON (OMM)
    manifest.json                                      source, times, counts

The analysis window starts at midnight UTC of the retrieval day.

    python -m orbit_explorer.gnss fetch     # a new snapshot (replaces the old)
    python -m orbit_explorer.gnss summary

The CelesTrak groups list the satellites in orbit, not their health: a
satellite in testing or set unusable still counts, so the DOP is what the
full listed constellation would give.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from orbit_explorer.fetch import MIN_HOURS_BETWEEN_DOWNLOADS, fetch_text

GNSS_DIR = Path("data/inputs/gnss")
GROUP_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=json"
# System name: (CelesTrak group, mean motion range in rev/day). GPS orbits
# twice per sidereal day (2.006 rev/day); Galileo 17 times in 10 days (1.70),
# except GSAT0201 and GSAT0202, left in an eccentric orbit by their 2014
# launch (1.86 rev/day, raised later; they broadcast and count).
SYSTEMS = {
    "GPS": ("gps-ops", (1.95, 2.05)),
    "Galileo": ("galileo", (1.65, 1.90)),
}
MIN_SATELLITES = 20  # fewer means a broken download, not a constellation


class SnapshotError(ValueError):
    """A download or a stored snapshot that can't be used."""


def check_group(system: str, records: object) -> list[dict]:
    """The records if they look like the whole constellation, else raise."""
    _, (low, high) = SYSTEMS[system]
    if not isinstance(records, list) or len(records) < MIN_SATELLITES:
        count = len(records) if isinstance(records, list) else "no"
        raise SnapshotError(f"{system}: {count} records, expected {MIN_SATELLITES}+")
    ids = [r.get("NORAD_CAT_ID") for r in records]
    if len(set(ids)) != len(ids):
        raise SnapshotError(f"{system}: a satellite is listed twice")
    for r in records:
        if not low <= r.get("MEAN_MOTION", 0) <= high:
            raise SnapshotError(
                f"{system}: {r.get('OBJECT_NAME')} has mean motion "
                f"{r.get('MEAN_MOTION')} rev/day, not {low}-{high}"
            )
    return records


def fetch_snapshot(folder: Path = GNSS_DIR, now: datetime | None = None) -> Path:
    """Download every system, check it, then replace the snapshot in
    ``folder``. Refuses within MIN_HOURS_BETWEEN_DOWNLOADS of the last one."""
    now = now or datetime.now(UTC)
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        last = datetime.fromisoformat(
            json.loads(manifest_path.read_text())["retrieved_utc"]
        )
        hours = (now - last).total_seconds() / 3600
        if hours < MIN_HOURS_BETWEEN_DOWNLOADS:
            raise SnapshotError(
                f"the last snapshot is {hours:.1f} h old; CelesTrak asks for "
                f"at least {MIN_HOURS_BETWEEN_DOWNLOADS} h between downloads"
            )
    downloads = {}
    for system, (group, _) in SYSTEMS.items():
        text = fetch_text(GROUP_URL.format(group=group))
        try:
            records = json.loads(text)
        except json.JSONDecodeError:
            raise SnapshotError(f"CelesTrak: {text.strip()[:80]!r}") from None
        downloads[system] = check_group(system, records)
    return write_snapshot(downloads, now, folder)


def write_snapshot(
    downloads: dict[str, list[dict]], retrieved: datetime, folder: Path = GNSS_DIR
) -> Path:
    """Replace the snapshot in ``folder`` with checked downloads per system."""
    downloads = {s: check_group(s, downloads[s]) for s in SYSTEMS}
    manifest_path = folder / "manifest.json"
    day = f"{retrieved:%Y-%m-%d}"
    files = {s: f"gp_{SYSTEMS[s][0]}_{day}.json" for s in SYSTEMS}
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("gp_*.json"):
        old.unlink()
    for system, records in downloads.items():
        (folder / files[system]).write_text(json.dumps(records, indent=1) + "\n")
    manifest = {
        "retrieved_utc": f"{retrieved:%Y-%m-%dT%H:%M:%SZ}",
        "reference_start_utc": f"{retrieved:%Y-%m-%d}T00:00:00+00:00",
        "source": GROUP_URL.format(group="<group>"),
        "systems": {
            system: {
                "group": SYSTEMS[system][0],
                "file": files[system],
                "satellites": len(records),
                "epochs_utc": [
                    min(r["EPOCH"] for r in records) + "Z",
                    max(r["EPOCH"] for r in records) + "Z",
                ],
            }
            for system, records in downloads.items()
        },
        "note": (
            "All satellites CelesTrak lists in each group, without health "
            "flags: a satellite in testing or set unusable still counts."
        ),
        "made_with": "python -m orbit_explorer.gnss fetch",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def load_snapshot(folder: Path = GNSS_DIR) -> tuple[dict[str, list[dict]], datetime]:
    """GP records per system ("GPS", "Galileo") and the window start."""
    manifest = json.loads((folder / "manifest.json").read_text())
    systems = {
        system: check_group(
            system,
            json.loads((folder / manifest["systems"][system]["file"]).read_text()),
        )
        for system in SYSTEMS
    }
    return systems, datetime.fromisoformat(manifest["reference_start_utc"])


def summary(folder: Path = GNSS_DIR) -> str:
    manifest = json.loads((folder / "manifest.json").read_text())
    lines = [f"GNSS snapshot retrieved {manifest['retrieved_utc']}"]
    for system, info in manifest["systems"].items():
        first, last = (e[:16].replace("T", " ") for e in info["epochs_utc"])
        lines.append(
            f"{system:8s} {info['satellites']:3d} satellites, epochs {first} to {last}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["fetch", "summary"])
    parser.add_argument("--dir", type=Path, default=GNSS_DIR)
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            print(f"Written {fetch_snapshot(args.dir)}")
        print(summary(args.dir))
    except SnapshotError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
