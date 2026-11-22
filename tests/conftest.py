"""Shared test data.

Most tests run on the current GP data in data/inputs/, which the daily refresh
replaces every day; there they check what must hold for any day's data. Tests
that check what one particular day shows (e.g. how much J2 helps each
satellite, or agreement with that day's precise orbits) use a frozen copy of
the GP data of 2026-11-17 instead, propagated to that day, so they can't break
when the data moves on.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from orbit_explorer.elements import load_elements

FROZEN_GP_PATH = Path(__file__).parent / "data" / "gp_2026-11-17.json"
FROZEN_START = datetime(2026, 11, 17, tzinfo=UTC)  # its window start


@pytest.fixture(scope="session")
def frozen_sats() -> dict[int, dict]:
    """The GP records of 2026-11-17, by NORAD ID."""
    return {s["NORAD_CAT_ID"]: s for s in load_elements(FROZEN_GP_PATH)}
