# Satellite Orbit Explorer

Keplerian orbit propagation and visualization of four real navigation satellites
(LEO, MEO, GEO and IGSO) from CelesTrak GP data, in Python.

**[Open the 3D viewer](https://devojka-od-papira.github.io/satellite-orbit-explorer/)**:
the four satellites on a globe, animated over the day, in the Earth-fixed or
inertial frame, with the Kepler and SGP4 positions side by side.

![Ground tracks of four satellites over one day](docs/images/ground_tracks.png)

## What it does
- Reads GP (OMM) orbital elements and derives semi-major axis, period, perigee and apogee
- Propagates each orbit over one day at a 5 s step (Kepler's equation, Newton-Raphson)
- Transforms ECI positions to ECEF and to geodetic coordinates on WGS84
- Writes an orbit report, plots ground tracks and speed profiles, and exports KML for Google Earth
- Exports CZML for CesiumJS: animated satellites, ground tracks and inertial orbits,
  for both Kepler and SGP4
- Compares the two-body Kepler model with SGP4/SDP4 over the same day

## Data
`data/inputs/orbital_elements.json` holds the GP records and
`data/inputs/data_manifest.json` records where and when they were downloaded.
The propagation window starts at midnight UTC of the download day. To refresh
from CelesTrak:

    python -m orbit_explorer.fetch

This rewrites both files for the same four satellites, but only if every
download succeeds. It warns when an epoch is more than 3 days from the window
start. CelesTrak asks for no more than one download of the same data every two
hours.

## Validation
All periods, perigee/apogee altitudes and speed ranges matched the GNSS Academy
workshop reference values for the four satellites. That dataset (downloaded
2026-11-10) is kept in `tests/data/`, and the period test runs against it,
because fresh elements change: QZS-2's period moves after station-keeping.

## Kepler vs SGP4
GP elements are meant for SGP4, which adds Earth's oblateness (J2–J4), drag and,
for high orbits, lunar and solar gravity. Comparing positions over the day
(GP data of 2026-11-17) shows what the two-body model leaves out:

- At each satellite's GP epoch the two models differ by 8–32 km. GP elements
  are SGP4 *mean* elements, not osculating Keplerian ones.
- **LEO (PULSAR-0)** drifts apart fastest: about 340 km by the end of the day,
  mostly in-track, plus a growing cross-track swing from the J2 nodal drift of
  its sun-synchronous orbit.
- **MEO, GEO and IGSO** stay within about 30–55 km all day.
- Element age matters as much as the model: with the earlier QZS-2 record,
  23 days old, the difference was about 200 km from the start of the day. With
  a fresh record it stays below 55 km.

![Kepler minus SGP4 position difference over the day](docs/images/sgp4_differences.png)

## Run it
    pip install -e .
    python -m orbit_explorer.report
    python -m orbit_explorer.plots
    python -m orbit_explorer.kml
    python -m orbit_explorer.czml
    python -m orbit_explorer.sgp4_compare

## 3D viewer
A static CesiumJS page in `web/` that loads `orbits.czml`. GitHub Actions
builds and publishes it to GitHub Pages on every push to `main`. To preview it
locally:

    python -m orbit_explorer.czml
    mkdir -p _site && cp web/* outputs/orbits.czml _site/
    python -m http.server 8000 -d _site    # then open http://localhost:8000

## Development
    pip install -e ".[dev]"
    pre-commit install
    pytest

Built during GNSS Academy's Space Career Week 2026.
