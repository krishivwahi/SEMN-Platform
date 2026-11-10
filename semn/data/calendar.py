"""
semn/data/calendar.py
==========================
NYSE trading sessions and rebalance schedules (Phase 1).

Daily data is keyed by **session date** (the exchange-local trading day), never
by UTC midnight.  Staleness and ADV windows are counted in sessions.
"""

from __future__ import annotations

from datetime import date, timedelta

import exchange_calendars as xcals
import polars as pl

EXCHANGE = "XNYS"


def sessions(start: str | date, end: str | date, exchange: str = EXCHANGE) -> pl.Series:
    """NYSE session dates in ``[start, end]`` (both inclusive), as a sorted ``Date`` series."""
    start, end = date.fromisoformat(str(start)), date.fromisoformat(str(end))
    # Pad the calendar's own bounds: exchange_calendars rejects bounds that fall on a holiday.
    cal = xcals.get_calendar(exchange, start=str(start - timedelta(days=14)), end=str(end + timedelta(days=14)))
    return pl.Series("date", cal.sessions_in_range(str(start), str(end)).date.tolist(), dtype=pl.Date)


def rebalance_dates(session_dates: pl.Series, every: str = "1mo") -> pl.Series:
    """
    Last session of each period (``"1w"``, ``"1mo"``, ``"1q"``, …; Polars duration strings).

    Uses the last *actual* session, so holidays and half-periods are handled by the calendar.
    """
    return (
        session_dates.to_frame("date")
        .group_by(pl.col("date").dt.truncate(every).alias("period"), maintain_order=True)
        .agg(pl.col("date").max())
        .sort("date")["date"]
    )
