"""An archive of every GP element set the daily refresh has downloaded.

The refresh replaces data/inputs/orbital_elements.json each day; this keeps
the history. One JSON Lines file per satellite, data/archive/<NORAD id>.jsonl,
one element set per line, ordered by epoch:

    {"norad_cat_id": 64541, "object_name": "PULSAR-0",
     "epoch_utc": "2026-11-23T03:12:52.902432Z",
     "retrieved_utc": "2026-11-23T12:57:13Z",
     "gp": {...the CelesTrak GP record, unchanged...},
     "tle": ["1 64541U ...", "2 64541 ..."]}

An element set is identified by its satellite and epoch: CelesTrak serves the
same set until a new one is published, so a download that finds no new set
adds nothing, and the first retrieval of a set is the one kept. (A set that
comes back with the same epoch but different elements, a correction, is
reported, not silently dropped.)

A series of element sets is what shows what a satellite does: station-keeping
manoeuvres as jumps that no force model explains, drag lowering a LEO orbit,
and how quickly an element set goes stale.

    python -m orbit_explorer.archive add             # data/inputs -> archive
                                                     # (fetch.py does it itself)
    python -m orbit_explorer.archive merge OTHER_DIR # another archive's sets
    python -m orbit_explorer.archive summary
"""

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from orbit_explorer.elements import DATA_PATH, MANIFEST_PATH

ARCHIVE_DIR = Path("data/archive")
GP_FIELDS = (
    "OBJECT_NAME",
    "NORAD_CAT_ID",
    "EPOCH",
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
    "BSTAR",
)


class ArchiveError(ValueError):
    """A record or an archive file that can't be trusted."""


def archive_path(norad: int, archive_dir: Path = ARCHIVE_DIR) -> Path:
    return archive_dir / f"{norad}.jsonl"


def epoch_of(row: dict) -> datetime:
    return datetime.fromisoformat(row["epoch_utc"])


def check_row(row: dict) -> None:
    """Raise ArchiveError unless the row is a complete, consistent entry."""
    try:
        gp = row["gp"]
        missing = [f for f in GP_FIELDS if f not in gp]
        if missing:
            raise ArchiveError(f"GP record without {', '.join(missing)}")
        if gp["NORAD_CAT_ID"] != row["norad_cat_id"]:
            raise ArchiveError(
                f"NORAD id {row['norad_cat_id']} but GP record {gp['NORAD_CAT_ID']}"
            )
        if epoch_of(row) != datetime.fromisoformat(gp["EPOCH"] + "Z"):
            raise ArchiveError(f"epoch {row['epoch_utc']} but GP record {gp['EPOCH']}")
        datetime.fromisoformat(row["retrieved_utc"])
        tle = row["tle"]
        if len(tle) != 2 or not all(
            line[:2] == f"{k} " and line[2:7] == f"{row['norad_cat_id']:05d}"
            for k, line in enumerate(tle, 1)
        ):
            raise ArchiveError(f"not the two TLE lines of {row['norad_cat_id']}")
    except (KeyError, TypeError, ValueError) as err:
        if isinstance(err, ArchiveError):
            raise
        raise ArchiveError(f"incomplete row ({type(err).__name__}: {err})") from None


def rows_from_download(
    data_path: Path = DATA_PATH, manifest_path: Path = MANIFEST_PATH
) -> list[dict]:
    """Archive rows for one download: data/inputs as fetch.py writes it."""
    records = json.loads(data_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    tles = {
        r["norad_cat_id"]: r["tle_lines_same_element_set"]
        for r in manifest["source_records"]
    }
    rows = []
    for gp in records:
        norad = gp["NORAD_CAT_ID"]
        if norad not in tles:
            raise ArchiveError(f"{norad} is not in the manifest {manifest_path}")
        row = {
            "norad_cat_id": norad,
            "object_name": gp["OBJECT_NAME"],
            "epoch_utc": gp["EPOCH"] + "Z",
            "retrieved_utc": manifest["retrieved_utc"],
            "gp": gp,
            "tle": tles[norad],
        }
        check_row(row)
        rows.append(row)
    return rows


def read_file(path: Path, strict: bool = True) -> list[dict]:
    """Rows of one archive file, in file order.

    strict: raise ArchiveError on a bad line (before writing, so a rewrite
    can never drop it). Otherwise warn and skip it, for analyses that should
    go on with the rest of the history.
    """
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            check_row(row)
            if path.stem != str(row["norad_cat_id"]):
                raise ArchiveError(f"satellite {row['norad_cat_id']} in {path.name}")
        except (json.JSONDecodeError, ArchiveError) as err:
            message = f"{path}:{number}: {err}"
            if strict:
                raise ArchiveError(message) from None
            warnings.warn(f"skipped {message}", stacklevel=2)
            continue
        rows.append(row)
    return rows


def load_history(norad: int, archive_dir: Path = ARCHIVE_DIR) -> list[dict]:
    """Every archived element set of one satellite, oldest epoch first;
    bad lines are skipped with a warning."""
    return sorted(
        read_file(archive_path(norad, archive_dir), strict=False), key=epoch_of
    )


@dataclass
class Update:
    """What adding element sets did: the new ones, and the ones refused
    because the archive already holds a *different* set with that epoch
    (archived, refused): normally the same epoch means the same set, so a
    conflict is worth a look."""

    added: list[dict] = field(default_factory=list)
    conflicts: list[tuple[dict, dict]] = field(default_factory=list)


def add_rows(rows: list[dict], archive_dir: Path = ARCHIVE_DIR) -> Update:
    """Add the rows whose (satellite, epoch) isn't archived yet.

    Of two rows with one epoch the first retrieved is kept; if their GP
    records differ, that is reported as a conflict. Each touched file is
    rewritten sorted by epoch, through a temporary file, so an interrupted
    write leaves the old file in place.
    """
    by_norad: dict[int, list[dict]] = {}
    for row in rows:
        check_row(row)
        by_norad.setdefault(row["norad_cat_id"], []).append(row)

    update = Update()
    for norad, new in by_norad.items():
        path = archive_path(norad, archive_dir)
        kept = {epoch_of(r): r for r in read_file(path, strict=True)}
        fresh = []
        for row in sorted(new, key=lambda r: (epoch_of(r), r["retrieved_utc"])):
            known = kept.get(epoch_of(row))
            if known is None:
                kept[epoch_of(row)] = row
                fresh.append(row)
            elif known["gp"] != row["gp"]:
                update.conflicts.append((known, row))
        if not fresh:
            continue
        archive_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".jsonl.tmp")  # ignored by git, if left behind
        tmp.write_text(
            "".join(
                json.dumps(kept[epoch], separators=(",", ":")) + "\n"
                for epoch in sorted(kept)
            )
        )
        tmp.replace(path)
        update.added.extend(fresh)
    return update


def merge(other_dir: Path, archive_dir: Path = ARCHIVE_DIR) -> Update:
    """Add another archive's element sets (e.g. an unmerged refresh branch's)."""
    rows = [
        row
        for path in sorted(other_dir.glob("*.jsonl"))
        for row in read_file(path, strict=True)
    ]
    return add_rows(rows, archive_dir)


def summary(archive_dir: Path = ARCHIVE_DIR) -> str:
    lines = []
    for path in sorted(archive_dir.glob("*.jsonl"), key=lambda p: int(p.stem)):
        rows = load_history(int(path.stem), archive_dir)
        if not rows:
            continue
        lines.append(
            f"{rows[-1]['object_name']:24s} {len(rows):4d} element sets  "
            f"{epoch_of(rows[0]):%Y-%m-%d %H:%M} to {epoch_of(rows[-1]):%Y-%m-%d %H:%M}"
        )
    return "\n".join(lines) if lines else f"No element sets in {archive_dir}"


def describe(update: Update) -> str:
    def name(r: dict) -> str:
        return f"{r['object_name']} {epoch_of(r):%Y-%m-%d %H:%M}"

    lines = [
        "Archive: added "
        + ", ".join(
            map(
                name,
                sorted(update.added, key=lambda r: (r["object_name"], epoch_of(r))),
            )
        )
        if update.added
        else "Archive: no new element sets."
    ]
    for known, refused in update.conflicts:
        lines.append(
            f"Warning: {name(known)} was downloaded again with different elements "
            f"(retrieved {refused['retrieved_utc']}); the archive keeps the one "
            f"retrieved {known['retrieved_utc']}."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, default=ARCHIVE_DIR)
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="archive a download (fetch.py does this)")
    add.add_argument("--elements", type=Path, default=DATA_PATH)
    add.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    other = commands.add_parser("merge", help="add another archive's element sets")
    other.add_argument("other", type=Path)
    commands.add_parser("summary", help="element sets per satellite")
    args = parser.parse_args(argv)

    try:
        if args.command == "summary":
            print(summary(args.archive))
        elif args.command == "merge":
            print(describe(merge(args.other, args.archive)))
        else:
            rows = rows_from_download(args.elements, args.manifest)
            print(describe(add_rows(rows, args.archive)))
    except ArchiveError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
