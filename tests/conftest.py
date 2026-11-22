"""Shared test data.

Most tests run on the current GP data in data/inputs/, which the daily refresh
replaces every day; there they check what must hold for any day's data. Tests
that check what one particular day shows (e.g. how much J2 helps each
satellite, or agreement with that day's precise orbits) use a frozen copy of
the GP data of 2026-11-17 instead, propagated to that day, so they can't break
when the data moves on.
"""

import pytest

from orbit_explorer.elements import load_elements
from orbit_explorer.validation import PRECISE_GP, PRECISE_START

# The precise day in data/inputs/precise/ (its manifest is the one source)
FROZEN_GP_PATH = PRECISE_GP
FROZEN_START = PRECISE_START


@pytest.fixture(scope="session")
def frozen_sats() -> dict[int, dict]:
    """The GP records of 2026-11-17, by NORAD ID."""
    return {s["NORAD_CAT_ID"]: s for s in load_elements(FROZEN_GP_PATH)}
