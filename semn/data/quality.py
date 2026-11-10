"""
semn/data/quality.py
==========================
Data-quality report for the Phase 1 data layer.

One long table, one row per finding: ``check, date, security_id, value``.

Checks
------
coverage            per session: share of index members with a price row (``security_id`` null)
missing_bar         index member with no price row on a session
duplicate_row       more than one price row for the same (date, security_id)
off_session         price row on a date that is not an NYSE session (the universe ignores it)
bad_price           raw close is zero, negative or NaN (``value`` = close)
suspicious_return   ``|ret| > 50%`` with no split, dividend or delisting that day (``value`` = ret)
split_volume        split day where share volume did not move by about the split ratio:
                    ``value`` = (volume_t / volume_{t-1}) / ratio, flagged outside [0.2, 5]
                    (catches vendors that already split-adjusted the volume history)
membership_gap      a security re-enters the same index within 7 days of leaving, or its
                    intervals overlap (``value`` = gap in days, negative = overlap)
stale               excluded from the universe for a stale price (from ``UniverseBuilder.exclusions``)

``report.group_by("check").len()`` gives the summary.
"""

from __future__ import annotations

import polars as pl

_KEY = "security_id"
SUSPICIOUS_RETURN = 0.5
MEMBERSHIP_GAP_DAYS = 7


def quality_report(
    prices: pl.DataFrame,
    membership: pl.DataFrame,
    sessions: pl.Series,
    returns: pl.DataFrame,
    splits: pl.DataFrame,
    exclusions: pl.DataFrame,
) -> pl.DataFrame:
    """
    Parameters
    ----------
    prices:      raw ``date, security_id, close, volume``
    membership:  ``security_id, index, entry_date, exit_date``
    sessions:    NYSE session dates to check (``calendar.sessions``)
    returns:     ``corporate_actions.total_returns`` output
    splits:      ``security_id, ex_date, ratio``
    exclusions:  ``UniverseBuilder.exclusions`` output
    """
    key_type = prices.schema[_KEY]

    def rows(df: pl.DataFrame, check: str, value: pl.Expr) -> pl.DataFrame:
        return df.select(
            pl.lit(check).alias("check"),
            pl.col("date"),
            pl.col(_KEY).cast(key_type),
            value.cast(pl.Float64).alias("value"),
        )

    active = (
        sessions.to_frame("date")
        .join(membership.select(_KEY, "entry_date", "exit_date").unique(), how="cross")
        .filter((pl.col("entry_date") <= pl.col("date")) & (pl.col("date") < pl.col("exit_date")))
        .select("date", _KEY)
        .unique()
        .join(prices.select("date", _KEY, pl.lit(True).alias("priced")).unique(), on=["date", _KEY], how="left")
    )
    coverage = (
        active.group_by("date")
        .agg(pl.col("priced").is_not_null().mean().alias("value"))
        .select(pl.lit("coverage").alias("check"), "date", pl.lit(None, dtype=key_type).alias(_KEY), "value")
    )

    by_day = prices.sort(_KEY, "date").with_columns(pl.col("volume").shift().over(_KEY).alias("prev_volume"))
    split_days = splits.select(_KEY, pl.col("ex_date").alias("date"), "ratio").join(by_day, on=[_KEY, "date"])
    split_move = pl.col("volume") / pl.col("prev_volume") / pl.col("ratio")

    gaps = (
        membership.sort(_KEY, "index", "entry_date")
        .with_columns(pl.col("exit_date").shift().over(_KEY, "index").alias("prev_exit"))
        .with_columns((pl.col("entry_date") - pl.col("prev_exit")).dt.total_days().alias("gap"))
        .filter(pl.col("gap") <= MEMBERSHIP_GAP_DAYS)
        .rename({"entry_date": "date"})
    )

    return pl.concat([
        coverage,
        rows(active.filter(pl.col("priced").is_null()), "missing_bar", pl.lit(None)),
        rows(prices.filter(pl.struct("date", _KEY).is_duplicated()), "duplicate_row", pl.lit(None)),
        rows(prices.filter(pl.col("date").is_between(sessions.min(), sessions.max())
                           & ~pl.col("date").is_in(sessions.implode())), "off_session", pl.lit(None)),
        rows(prices.filter(pl.col("close").is_nan() | (pl.col("close") <= 0)), "bad_price", pl.col("close")),
        rows(returns.filter((pl.col("ret").abs() > SUSPICIOUS_RETURN) & ~pl.col("has_action")),
             "suspicious_return", pl.col("ret")),
        rows(split_days.filter((split_move < 0.2) | (split_move > 5)), "split_volume", split_move),
        rows(gaps, "membership_gap", pl.col("gap")),
        rows(exclusions.filter(pl.col("exclusion") == "stale"), "stale", pl.lit(None)),
    ]).sort("check", "date", _KEY, nulls_last=True)
