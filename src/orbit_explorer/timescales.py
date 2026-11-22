"""Time scales: the one leap-second table the project uses.

GPS, Galileo and QZSS time run 18 s ahead of UTC, TAI 37 s, and TT is TAI +
32.184 s, all since the last leap second (2017-01-01). IERS announces each
leap second in Bulletin C about six months ahead; after checking a new
bulletin, move LEAP_SECONDS_CHECKED_UNTIL forward (or add the leap second
here). Times outside the checked range are refused rather than silently
off by a second.
"""

from datetime import UTC, datetime

# Seconds to subtract from each time scale to get UTC.
TIME_SCALE_MINUS_UTC_S = {"GPS": 18, "GAL": 18, "QZS": 18, "TAI": 37, "UTC": 0}
TT_MINUS_TAI_S = 32.184
TT_MINUS_UTC_S = TIME_SCALE_MINUS_UTC_S["TAI"] + TT_MINUS_TAI_S  # 69.184 s
LEAP_SECONDS_VALID_FROM = datetime(2017, 1, 1, tzinfo=UTC)
LEAP_SECONDS_CHECKED_UNTIL = datetime(2027, 1, 1, tzinfo=UTC)


def check_leap_second_range(first_utc: datetime, last_utc: datetime) -> None:
    """Refuse UTC times the leap-second table isn't valid for."""
    if first_utc < LEAP_SECONDS_VALID_FROM:
        raise ValueError("leap second table starts at 2017-01-01")
    if last_utc >= LEAP_SECONDS_CHECKED_UNTIL:
        raise ValueError(
            f"leap seconds checked only until {LEAP_SECONDS_CHECKED_UNTIL:%Y-%m-%d}"
            "; see IERS Bulletin C and update the table"
        )
