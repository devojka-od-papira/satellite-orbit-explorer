"""Precise GNSS orbits in SP3 format: read, write a subset, interpolate.

SP3 files from the IGS analysis centres give satellite positions every few
minutes, accurate to a few centimetres, in an Earth-fixed frame (ITRF, e.g.
IGc20). They are the "ground truth" the propagated orbits can be checked
against. Two details matter more than the format itself:

- **Time scale.** SP3 epochs are in GPS time (or another GNSS time), not UTC.
  GPS time is 18 s ahead of UTC since the leap second of 2017-01-01; ignoring
  that shifts GALILEO 29 by about 50 km in the Earth-fixed frame (and QZS-2
  by about 25 km), many times the real model errors.
- **Sampling.** Positions are given every 5 or 15 minutes. In between they are
  interpolated with a Lagrange polynomial over the nearest epochs, the usual
  way for GNSS orbits; at 5 min and order 10 the error is far below a
  centimetre.

Supports SP3-c and SP3-d position files (velocity records are skipped).

    python -m orbit_explorer.sp3 SRC.SP3.gz --keep E29 J02 -o OUT.sp3
"""

import argparse
import gzip
import math
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

PRECISE_DIR = Path("data/inputs/precise")
PRECISE_ORBITS = PRECISE_DIR / "GBM0MGXRAP_20263210000_01D_05M_ORB_E29_J02.sp3"

# Our satellites in the SP3 files, by NORAD ID. Found by matching positions:
# each is within a few km of its SGP4 position and thousands of km from any
# other satellite in the file (see the tests). PULSAR-0 and SES-5 have no
# public precise orbits.
SP3_IDS = {59598: "E29", 42738: "J02"}  # GALILEO 29 (GSAT0225), QZS-2

# Seconds to subtract from each time scale to get UTC. Valid from 2017-01-01,
# the last leap second; IERS has announced none since.
TIME_SCALE_MINUS_UTC_S = {"GPS": 18, "GAL": 18, "QZS": 18, "TAI": 37, "UTC": 0}
LEAP_SECONDS_VALID_FROM = datetime(2017, 1, 1, tzinfo=UTC)

GPS_EPOCH = datetime(1980, 1, 6, tzinfo=UTC)
MJD_EPOCH = datetime(1858, 11, 17, tzinfo=UTC)
BAD_POSITION = 0.0  # SP3: all three coordinates 0.000000 means "no position"
DEFAULT_ORDER = 10  # Lagrange interpolation over 11 epochs


@dataclass
class Sp3:
    """The positions of an SP3 file. ``epochs`` are naive datetimes in
    ``time_system``; ``positions[id]`` is (len(epochs), 3) km in
    ``coord_system``, NaN where the file has no position."""

    epochs: list[datetime]
    positions: dict[str, np.ndarray]
    time_system: str = "GPS"
    coord_system: str = "IGc20"
    agency: str = ""
    comments: list[str] = field(default_factory=list)

    @property
    def interval_s(self) -> float:
        return (self.epochs[1] - self.epochs[0]).total_seconds()

    def utc_epochs(self) -> list[datetime]:
        """The epochs in UTC (timezone-aware)."""
        try:
            offset = TIME_SCALE_MINUS_UTC_S[self.time_system]
        except KeyError:
            raise ValueError(f"unsupported SP3 time system {self.time_system!r}")
        utc = [e.replace(tzinfo=UTC) - timedelta(seconds=offset) for e in self.epochs]
        if utc[0] < LEAP_SECONDS_VALID_FROM:
            raise ValueError("leap second table starts at 2017-01-01")
        return utc

    def seconds_since(self, start_utc: datetime) -> np.ndarray:
        """Epochs as seconds after a UTC instant, e.g. the window START."""
        return np.array([(e - start_utc).total_seconds() for e in self.utc_epochs()])

    def subset(self, ids: list[str]) -> "Sp3":
        missing = [i for i in ids if i not in self.positions]
        if missing:
            raise KeyError(f"not in the SP3 file: {', '.join(missing)}")
        return replace(self, positions={i: self.positions[i] for i in ids})


# ---------- Reading ----------
def parse_sp3(text: str) -> Sp3:
    lines = text.splitlines()
    if not lines or not lines[0].startswith("#") or lines[0][1] not in "cd":
        raise ValueError("not an SP3-c or SP3-d file")
    if lines[0][2] != "P":
        raise ValueError("only position (P) SP3 files are supported")
    header = lines[0]
    coord_system = header[46:51].strip()
    agency = header[56:60].strip()
    time_system = "GPS"  # SP3-a/b default; %c names it in c/d
    comments = []
    epochs: list[datetime] = []
    records: dict[str, dict[int, np.ndarray]] = {}
    for line in lines[1:]:
        if (
            line.startswith("%c")
            and not epochs
            and line[9:12].strip() not in ("", "ccc")
        ):
            time_system = line[9:12].strip()
        elif line.startswith("/*") and not epochs:
            comments.append(line[3:].rstrip())
        elif line.startswith("* "):
            fields = line[2:].split()
            whole = datetime(*(int(f) for f in fields[:5]))
            epochs.append(whole + timedelta(seconds=float(fields[5])))
        elif line.startswith("P") and epochs:
            xyz = np.array([float(line[4:18]), float(line[18:32]), float(line[32:46])])
            if np.all(xyz == BAD_POSITION):
                xyz = np.full(3, np.nan)
            records.setdefault(line[1:4], {})[len(epochs) - 1] = xyz
        elif line.startswith("EOF"):
            break
    if not epochs:
        raise ValueError("SP3 file has no epochs")
    positions = {}
    for sat_id, by_epoch in records.items():
        pos = np.full((len(epochs), 3), np.nan)
        for k, xyz in by_epoch.items():
            pos[k] = xyz
        positions[sat_id] = pos
    return Sp3(epochs, positions, time_system, coord_system, agency, comments)


def read_sp3(path: Path) -> Sp3:
    """Read a .sp3 file, gzipped or not."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="ascii") as f:
        return parse_sp3(f.read())


# ---------- Writing (to keep a small subset in the repository) ----------
def _epoch_fields(t: datetime) -> str:
    """'2026 11 17  0  5  0.00000000': year, then space-padded fields."""
    sec = t.second + t.microsecond / 1e6
    return f"{t.year:4d} {t.month:2d} {t.day:2d} {t.hour:2d} {t.minute:2d} {sec:11.8f}"


def format_sp3(sp3: Sp3) -> str:
    """SP3-d text for ``sp3`` (positions only, clocks left as 999999.999999)."""
    ids = list(sp3.positions)
    first = sp3.epochs[0]
    aware = first.replace(tzinfo=UTC)
    gps_s = (aware - GPS_EPOCH).total_seconds()
    week, sow = divmod(gps_s, 7 * 86400)
    mjd_float = (aware - MJD_EPOCH).total_seconds() / 86400
    mjd = math.floor(mjd_float)
    lines = [
        # fixed columns: 4-31 first epoch, 33-39 epochs, 41-45 data used,
        # 47-51 coordinate system, 53-55 orbit type, 57-60 agency
        (
            f"#dP{_epoch_fields(first)} {len(sp3.epochs):7d} {'u+U':>5s} "
            f"{sp3.coord_system:>5s} FIT {sp3.agency:>4s}"
        ),
        (
            f"## {int(week):4d} {sow:15.8f} {sp3.interval_s:14.8f} {mjd:5d} "
            f"{mjd_float - mjd:15.13f}"
        ),
    ]
    rows = [ids[i : i + 17] for i in range(0, len(ids), 17)]
    rows += [[]] * max(0, 5 - len(rows))
    for k, row in enumerate(rows):
        lead = f"+  {len(ids):3d}   " if k == 0 else "+        "
        cells = "".join(row) + "  0" * (17 - len(row))
        lines.append(lead + cells)
    for row in rows:
        lines.append("++       " + "  0" * 17)
    lines += [
        f"%c M  cc {sp3.time_system:3s} ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc",
        "%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc",
        "%f  1.2500000  1.025000000  0.00000000000  0.000000000000000",
        "%f  0.0000000  0.000000000  0.00000000000  0.000000000000000",
        "%i    0    0    0    0      0      0      0      0         0",
        "%i    0    0    0    0      0      0      0      0         0",
    ]
    comments = sp3.comments + [""] * max(0, 4 - len(sp3.comments))
    lines += [f"/* {c}".rstrip() if c else "/*" for c in comments]
    for k, epoch in enumerate(sp3.epochs):
        lines.append(f"*  {_epoch_fields(epoch)}")
        for sat_id in ids:
            xyz = sp3.positions[sat_id][k]
            if np.any(np.isnan(xyz)):
                xyz = np.zeros(3)
            lines.append(
                f"P{sat_id}{xyz[0]:14.6f}{xyz[1]:14.6f}{xyz[2]:14.6f} 999999.999999"
            )
    lines.append("EOF")
    return "\n".join(lines) + "\n"


# ---------- Interpolation ----------
def lagrange_interpolate(
    t_nodes: np.ndarray, values: np.ndarray, t: np.ndarray, order: int = DEFAULT_ORDER
) -> np.ndarray:
    """Values at times t from samples at t_nodes, with a Lagrange polynomial of
    degree ``order`` through the nearest order + 1 nodes.

    The window is centred on t where possible and slides at the ends. Outside
    [t_nodes[0], t_nodes[-1]] (no extrapolation) or where a window has a
    missing value, the result is NaN. ``values`` may be (n,) or (n, k).
    """
    t_nodes = np.asarray(t_nodes, dtype=float)
    values = np.asarray(values, dtype=float)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n = len(t_nodes)
    if n < order + 1:
        raise ValueError(f"need at least {order + 1} nodes for order {order}")
    # first node of each window: centred on t, clipped to the ends
    centre = np.searchsorted(t_nodes, t)
    start = np.clip(centre - (order + 1) // 2, 0, n - order - 1)
    window = start[:, None] + np.arange(order + 1)  # (len(t), order + 1)
    tw = t_nodes[window]
    # Lagrange basis l_j(t) = prod_{m != j} (t - t_m) / (t_j - t_m)
    diff = t[:, None] - tw  # (len(t), order + 1)
    basis = np.ones_like(tw)
    for j in range(order + 1):
        for m in range(order + 1):
            if m != j:
                basis[:, j] *= diff[:, m] / (tw[:, j] - tw[:, m])
    vw = values[window]  # (len(t), order + 1, ...)
    out = np.einsum("tj,tj...->t...", basis, vw)
    outside = (t < t_nodes[0]) | (t > t_nodes[-1])
    out[outside] = np.nan
    return out


def positions_at(
    sp3: Sp3,
    sat_id: str,
    t_s: np.ndarray,
    start_utc: datetime,
    order: int = DEFAULT_ORDER,
) -> np.ndarray:
    """Earth-fixed positions [km] of ``sat_id`` at t_s seconds after the UTC
    instant ``start_utc``, interpolated from the SP3 epochs (converted to
    UTC). NaN outside the file's time span."""
    return lagrange_interpolate(
        sp3.seconds_since(start_utc), sp3.positions[sat_id], t_s, order
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Keep some satellites of an SP3 file")
    parser.add_argument("source", type=Path, help=".sp3 or .sp3.gz")
    parser.add_argument("--keep", nargs="+", required=True, help="e.g. E29 J02")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)
    sp3 = read_sp3(args.source).subset(args.keep)
    sp3.comments = [
        f"Subset of {args.source.name}: {' '.join(args.keep)}",
        *sp3.comments,
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_sp3(sp3))
    utc = sp3.utc_epochs()
    print(
        f"Written {args.output}: {len(sp3.epochs)} epochs every "
        f"{sp3.interval_s:.0f} s, {sp3.time_system} time "
        f"({utc[0]:%Y-%m-%d %H:%M:%S} to {utc[-1]:%Y-%m-%d %H:%M:%S} UTC), "
        f"{sp3.coord_system}"
    )


if __name__ == "__main__":
    main()
