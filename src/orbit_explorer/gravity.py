"""The Earth's gravity field as spherical harmonics (EGM2008, low degrees).

j2.py keeps only J2, the equatorial bulge. Above about 20,000 km the rest
of the field still matters: at Galileo's radius the terms beyond J2 together
pull with up to about 4e-10 km/s², more than sunlight does (orbit
determination in od.py needs both). The largest of them is C22/S22, the equator's slight
ellipticity, not J3.

The coefficients are EGM2008 (Pavlis et al. 2012), fully normalised, tide
free, from ICGEM. Only the first degrees are stored, in
data/inputs/gravity/. od.py uses degree 8: degrees 9-12 add at most 6e-15
km/s² at Galileo's radius, a few centimetres in a day (see the tests). To make the stored subset from
the full file:

    python -m orbit_explorer.gravity EGM2008.gfc --max-degree 12 -o data/inputs/gravity/EGM2008_n12.gfc

The acceleration follows Montenbruck & Gill, *Satellite Orbits* (2000),
section 3.2.4: the V/W recursions in Cartesian coordinates, which have no
singularity at the poles. The field is Earth-fixed: positions go in and
accelerations come out in ITRF.
"""

import argparse
from dataclasses import dataclass
from math import factorial
from pathlib import Path

import numpy as np

GRAVITY_DIR = Path("data/inputs/gravity")
EGM2008 = GRAVITY_DIR / "EGM2008_n12.gfc"
DEFAULT_DEGREE = 8


@dataclass(frozen=True)
class GravityField:
    """Unnormalised coefficients C[n, m], S[n, m] up to ``degree``, with the
    model's own GM [km³/s²] and reference radius [km]."""

    name: str
    gm: float
    radius_km: float
    degree: int
    C: np.ndarray
    S: np.ndarray


def _header_value(lines: list[str], key: str) -> str:
    for line in lines:
        parts = line.split()
        if parts and parts[0] == key:
            return parts[1]
    raise ValueError(f"gfc header has no {key}")


def unnormalise(n: int, m: int) -> float:
    """Factor turning a fully normalised coefficient into an unnormalised one."""
    return np.sqrt((2 - (m == 0)) * (2 * n + 1) * factorial(n - m) / factorial(n + m))


def load_field(path: Path = EGM2008, degree: int = DEFAULT_DEGREE) -> GravityField:
    """Read an ICGEM .gfc file (fully normalised) up to ``degree``."""
    text = path.read_text().splitlines()
    end = next(i for i, line in enumerate(text) if line.startswith("end_of_head"))
    head = text[:end]
    if _header_value(head, "norm") != "fully_normalized":
        raise ValueError(f"{path}: expected fully normalised coefficients")
    available = int(_header_value(head, "max_degree"))
    C = np.zeros((degree + 1, degree + 1))
    S = np.zeros_like(C)
    found = set()
    for line in text[end + 1 :]:
        parts = line.replace("d", "e").replace("D", "e").split()
        if not parts or parts[0] != "gfc":
            continue
        n, m = int(parts[1]), int(parts[2])
        if n > degree:
            continue
        f = unnormalise(n, m)
        C[n, m], S[n, m] = float(parts[3]) * f, float(parts[4]) * f
        found.add((n, m))
    missing = [(n, m) for n in range(degree + 1) for m in range(n + 1)]
    missing = [
        nm for nm in missing if nm not in found and nm != (1, 0) and nm != (1, 1)
    ]
    if missing:
        raise ValueError(
            f"{path}: degree {degree} asked, the file has up to "
            f"{min(available, max(n for n, _ in found))}; missing {missing[:3]}"
        )
    return GravityField(
        name=_header_value(head, "modelname"),
        gm=float(_header_value(head, "earth_gravity_constant")) / 1e9,
        radius_km=float(_header_value(head, "radius")) / 1000,
        degree=degree,
        C=C,
        S=S,
    )


def acceleration(r: np.ndarray, field: GravityField) -> np.ndarray:
    """Gravitational acceleration [km/s²] at Earth-fixed positions r [km],
    shape (N, 3): the point mass and every harmonic up to the field's degree."""
    r = np.atleast_2d(r)
    nmax = field.degree
    R = field.radius_km
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    r2 = x * x + y * y + z * z
    rho = R * R / r2
    x0, y0, z0 = R * x / r2, R * y / r2, R * z / r2
    size = nmax + 2  # the acceleration of degree n needs V and W of degree n + 1
    V = np.zeros((size, size, len(r)))
    W = np.zeros_like(V)
    V[0, 0] = R / np.sqrt(r2)
    for m in range(size):
        if m > 0:  # the sectorial terms, from the previous diagonal
            V[m, m] = (2 * m - 1) * (x0 * V[m - 1, m - 1] - y0 * W[m - 1, m - 1])
            W[m, m] = (2 * m - 1) * (x0 * W[m - 1, m - 1] + y0 * V[m - 1, m - 1])
        if m + 1 < size:
            V[m + 1, m] = (2 * m + 1) * z0 * V[m, m]
            W[m + 1, m] = (2 * m + 1) * z0 * W[m, m]
        for n in range(m + 2, size):
            k = (n + m - 1) * rho
            V[n, m] = ((2 * n - 1) * z0 * V[n - 1, m] - k * V[n - 2, m]) / (n - m)
            W[n, m] = ((2 * n - 1) * z0 * W[n - 1, m] - k * W[n - 2, m]) / (n - m)
    ax = np.zeros(len(r))
    ay = np.zeros(len(r))
    az = np.zeros(len(r))
    for n in range(nmax + 1):
        for m in range(n + 1):
            c, s = field.C[n, m], field.S[n, m]
            if c == 0 and s == 0:
                continue
            if m == 0:
                ax -= c * V[n + 1, 1]
                ay -= c * W[n + 1, 1]
            else:
                f = 0.5 * (n - m + 1) * (n - m + 2)
                ax += 0.5 * (-c * V[n + 1, m + 1] - s * W[n + 1, m + 1]) + f * (
                    c * V[n + 1, m - 1] + s * W[n + 1, m - 1]
                )
                ay += 0.5 * (-c * W[n + 1, m + 1] + s * V[n + 1, m + 1]) + f * (
                    -c * W[n + 1, m - 1] + s * V[n + 1, m - 1]
                )
            az += (n - m + 1) * (-c * V[n + 1, m] - s * W[n + 1, m])
    return field.gm / R**2 * np.stack([ax, ay, az], axis=-1)


def potential(r: np.ndarray, field: GravityField) -> float:
    """Gravitational potential [km²/s²] at one Earth-fixed position r [km],
    straight from the series (associated Legendre functions by their explicit
    polynomial): slow, but independent of acceleration()'s recursions, so the
    tests can check one against the other."""
    x, y, z = r
    rn = np.sqrt(x * x + y * y + z * z)
    t, lon = z / rn, np.arctan2(y, x)
    total = 0.0
    for n in range(field.degree + 1):
        legendre = np.polynomial.legendre.Legendre.basis(n).convert(
            kind=np.polynomial.Polynomial
        )
        for m in range(n + 1):
            p = (1 - t * t) ** (m / 2) * legendre.deriv(m)(t)
            total += (
                (field.radius_km / rn) ** n
                * p
                * (field.C[n, m] * np.cos(m * lon) + field.S[n, m] * np.sin(m * lon))
            )
    return field.gm / rn * total


def subset(text: str, max_degree: int) -> str:
    """The .gfc file with only the coefficients up to max_degree, its header
    kept (max_degree updated, the original noted)."""
    out = []
    in_head = True
    for line in text.splitlines():
        parts = line.split()
        if in_head:
            if parts and parts[0] == "max_degree":
                out.append(f"max_degree                  {max_degree}")
                out.append(
                    f"comment                     subset of max_degree {parts[1]}"
                )
                continue
            in_head = not line.startswith("end_of_head")
            out.append(line)
        elif parts and parts[0] == "gfc" and int(parts[1]) <= max_degree:
            out.append(line)
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Keep the first degrees of a .gfc")
    parser.add_argument("gfc", type=Path)
    parser.add_argument("--max-degree", type=int, default=12)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(subset(args.gfc.read_text(), args.max_degree))
    print(f"Written {args.output}")


if __name__ == "__main__":
    main()
