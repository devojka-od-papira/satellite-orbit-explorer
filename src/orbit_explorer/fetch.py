"""Download fresh GP records from CelesTrak and update the data manifest.

Refreshes the satellites already in data/inputs/orbital_elements.json, in the
same order, with one query per satellite (JSON for the elements, TLE for the
manifest's provenance record). The propagation window starts at midnight UTC
of the retrieval day. Nothing is written unless every satellite downloads and
validates, so a failed run leaves the data untouched. Run from the command
line, it then adds the new element sets to the archive in data/archive/
(orbit_explorer.archive), so the history is kept; --no-archive skips that.

CelesTrak asks clients not to download the same data more than once every two
hours; GP data is updated a few times a day at most.
"""

import argparse
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from orbit_explorer import archive
from orbit_explorer.elements import DATA_PATH, MANIFEST_PATH, load_elements

GP_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR={norad}&FORMAT={fmt}"
USER_AGENT = (
    "satellite-orbit-explorer "
    "(+https://github.com/devojka-od-papira/satellite-orbit-explorer)"
)
STALE_DAYS = 3  # flag records whose epoch is this far from the window start
MIN_HOURS_BETWEEN_DOWNLOADS = 2  # CelesTrak: the same data at most every 2 hours
TOO_SOON_EXIT_CODE = 3


class TooSoon(RuntimeError):
    """The last download was less than MIN_HOURS_BETWEEN_DOWNLOADS ago."""


def retrieved_utc(manifest: dict) -> datetime:
    return datetime.fromisoformat(manifest["retrieved_utc"])


def hours_since_download(manifest_paths: list[Path], now: datetime) -> float:
    """Hours since the most recent download recorded in any of the manifests
    (e.g. the committed one and the one on an unmerged refresh branch)."""
    latest = max(
        retrieved_utc(json.loads(path.read_text()))
        for path in manifest_paths
        if path.exists()
    )
    return (now - latest).total_seconds() / 3600


def stale_records(manifest: dict) -> list[str]:
    """'NAME (+x.x d)' for each record whose epoch is more than STALE_DAYS
    from the window start."""
    start = datetime.fromisoformat(manifest["reference_start_utc"])
    stale = []
    for rec in manifest["source_records"]:
        age = (start - datetime.fromisoformat(rec["epoch_utc"])).total_seconds() / 86400
        if abs(age) > STALE_DAYS:
            stale.append(f"{rec['object_name']} ({age:+.1f} d)")
    return stale


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode()


def fetch_record(norad: int) -> tuple[dict, list[str]]:
    """GP record (OMM JSON) and the two TLE lines of the same element set."""
    text = fetch_text(GP_URL.format(norad=norad, fmt="JSON"))
    try:
        records = json.loads(text)
    except json.JSONDecodeError:
        # CelesTrak answers e.g. "No GP data found" as plain text
        raise RuntimeError(f"CelesTrak: {text.strip()[:80]!r} for {norad}") from None
    if len(records) != 1 or records[0].get("NORAD_CAT_ID") != norad:
        raise RuntimeError(f"Expected one GP record for {norad}, got {len(records)}")

    tle_text = fetch_text(GP_URL.format(norad=norad, fmt="TLE"))
    tle = [line.strip() for line in tle_text.splitlines() if line[:2] in ("1 ", "2 ")]
    if len(tle) != 2 or not all(f" {norad:05d}" in line[:8] for line in tle):
        raise RuntimeError(f"Unexpected TLE for {norad}: {tle_text[:80]!r}")
    return records[0], tle


def epoch_utc(record: dict) -> datetime:
    return datetime.fromisoformat(record["EPOCH"]).replace(tzinfo=UTC)


def build_manifest(
    records: list[dict], tles: list[list[str]], retrieved: datetime
) -> dict:
    start = retrieved.replace(hour=0, minute=0, second=0, microsecond=0)
    ages = {
        r["OBJECT_NAME"]: (start - epoch_utc(r)).total_seconds() / 86400
        for r in records
    }
    stale = [
        f"{name} ({age:+.1f} d)" for name, age in ages.items() if abs(age) > STALE_DAYS
    ]
    limitations = [
        "Kepler propagation of GP mean elements is an educational approximation.",
        (
            "The JSON eccentricity has more digits than the seven-digit TLE field; "
            "the two encodings describe the same element set."
        ),
    ]
    if stale:
        limitations.insert(
            0,
            f"Epoch more than {STALE_DAYS} days from the window start: "
            + ", ".join(stale),
        )
    return {
        "assembled_on_utc": f"{retrieved:%Y-%m-%d}",
        "retrieved_utc": f"{retrieved:%Y-%m-%dT%H:%M:%SZ}",
        "format": "GP JSON using OMM field names "
        f"(native CelesTrak API download, {len(records)} satellites)",
        "provenance_note": "Native CelesTrak GP JSON, one query per satellite "
        f"(gp.php?CATNR=<id>&FORMAT=JSON), retrieved on {retrieved:%Y-%m-%d}. "
        "Epochs range from "
        f"{min(map(epoch_utc, records)):%Y-%m-%d %H:%M} to "
        f"{max(map(epoch_utc, records)):%Y-%m-%d %H:%M} UTC. "
        "No orbital model conversion was performed.",
        "reference_start_utc": f"{start:%Y-%m-%dT%H:%M:%SZ}",
        "source_records": [
            {
                "norad_cat_id": r["NORAD_CAT_ID"],
                "object_name": r["OBJECT_NAME"],
                "epoch_utc": f"{r['EPOCH']}Z",
                "source_url": GP_URL.format(norad=r["NORAD_CAT_ID"], fmt="JSON"),
                "source_representation": "Native CelesTrak GP JSON (OMM field names)",
                "tle_lines_same_element_set": tle,
            }
            for r, tle in zip(records, tles)
        ],
        "refresh_url_template": GP_URL.format(norad="{NORAD_CAT_ID}", fmt="JSON"),
        "known_limitations": limitations,
    }


def refresh(
    data_path: Path = DATA_PATH,
    manifest_path: Path = MANIFEST_PATH,
    now: datetime | None = None,
    force: bool = False,
    seen_manifests: tuple[Path, ...] = (),
) -> dict:
    """Download every satellite, then write elements and manifest together.

    Raises TooSoon instead of downloading if this manifest or any of
    ``seen_manifests`` records a download less than 2 hours ago, unless
    ``force`` is set.
    """
    retrieved = (now or datetime.now(UTC)).replace(microsecond=0)
    if not force:
        hours = hours_since_download([manifest_path, *seen_manifests], retrieved)
        if hours < MIN_HOURS_BETWEEN_DOWNLOADS:
            raise TooSoon(
                f"Last download was {hours:.1f} h ago; CelesTrak asks for at most "
                f"one every {MIN_HOURS_BETWEEN_DOWNLOADS} h. Nothing downloaded "
                "(use --force to override)."
            )
    norads = [sat["NORAD_CAT_ID"] for sat in load_elements(data_path)]
    fetched = [fetch_record(norad) for norad in norads]
    records = [record for record, _ in fetched]
    manifest = build_manifest(records, [tle for _, tle in fetched], retrieved)
    data_path.write_text(json.dumps(records, indent=2) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(
    argv: list[str] | None = None,
    data_path: Path = DATA_PATH,
    manifest_path: Path = MANIFEST_PATH,
    archive_dir: Path = archive.ARCHIVE_DIR,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--force",
        action="store_true",
        help=f"download even if the last one was under {MIN_HOURS_BETWEEN_DOWNLOADS} h ago",
    )
    parser.add_argument(
        "--seen-manifest",
        type=Path,
        action="append",
        default=[],
        help="another manifest whose download time counts for the rate limit",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="don't add the new element sets to the archive in data/archive/",
    )
    args = parser.parse_args(argv)
    try:
        manifest = refresh(
            data_path,
            manifest_path,
            force=args.force,
            seen_manifests=tuple(args.seen_manifest),
        )
    except TooSoon as err:
        print(err)
        return TOO_SOON_EXIT_CODE
    start = datetime.fromisoformat(manifest["reference_start_utc"])
    print(f"Window start: {start:%Y-%m-%d %H:%M} UTC")
    for rec in manifest["source_records"]:
        epoch = datetime.fromisoformat(rec["epoch_utc"])
        age = (start - epoch).total_seconds() / 86400
        print(
            f"  {rec['object_name']:24s} epoch {epoch:%Y-%m-%d %H:%M}  ({age:+.2f} d)"
        )
    stale = stale_records(manifest)
    if stale:
        print(f"Warning: epoch more than {STALE_DAYS} days old: {', '.join(stale)}")
    print(f"Written to {data_path} and {manifest_path}")
    if args.no_archive:
        return 0
    try:
        rows = archive.rows_from_download(data_path, manifest_path)
        print(archive.describe(archive.add_rows(rows, archive_dir)))
    except archive.ArchiveError as err:
        print(f"Error: the data is written but not archived: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
