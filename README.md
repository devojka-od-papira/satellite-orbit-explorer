# Satellite Orbit Explorer

Keplerian orbit propagation and visualization of four real navigation satellites
(LEO, MEO, GEO and IGSO) from CelesTrak GP data, in Python.

**[Open the 3D viewer](https://devojka-od-papira.github.io/satellite-orbit-explorer/)**:
the four satellites on a globe, animated over the day, in the Earth-fixed or
inertial frame.

![Ground tracks of four satellites over one day](docs/images/ground_tracks.png)

## What it does
- Reads GP (OMM) orbital elements and derives semi-major axis, period, perigee and apogee
- Propagates each orbit over one day at a 5 s step (Kepler's equation, Newton-Raphson)
- Transforms ECI positions to ECEF and to geodetic coordinates on WGS84
- Writes an orbit report, plots ground tracks and speed profiles, and exports KML for Google Earth
- Exports CZML for CesiumJS: animated satellites, ground tracks and inertial orbits

## Validation
All periods, perigee/apogee altitudes and speed ranges match the GNSS Academy
workshop reference values for the four satellites.

## Run it
    pip install -e .
    python -m orbit_explorer.report
    python -m orbit_explorer.plots
    python -m orbit_explorer.kml
    python -m orbit_explorer.czml

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

Built during GNSS Academy's Space Career Week 2026. Work in progress: an SGP4
comparison is next.
