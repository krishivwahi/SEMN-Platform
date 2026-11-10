"""
semn/data/corporate_actions.py
==========================
Total returns from raw prices + splits + dividends, with delisting returns (Phase 1).

Returns are computed from **raw** closes and action factors; adjusted price series
are derived from these returns, never used as inputs.

Input schemas
-------------
prices:     ``date, security_id, close``  (raw, unadjusted)
splits:     ``security_id, ex_date, ratio``  (new shares per old share: 10.0 for 10-for-1, 0.1 for 1-for-10)
dividends:  ``security_id, ex_date, amount``  (cash per share, per share outstanding after any same-day split)
delistings: ``security_id, delist_date, delist_return (nullable), performance_related (Boolean)``

Return on session t (ex-date actions land on the first priced session on/after ex_date):

    r_t = (close_t + dividend_t) * split_ratio_t / close_{t-1} - 1

On ``delist_date`` the delisting return is compounded in exactly once:
``(1 + r_t)(1 + dlret) - 1``, adding a row if there is no price that day.  Rows
after ``delist_date`` are dropped.  When the vendor has no delisting return, the
fallback is ``PERFORMANCE_DELISTING_FALLBACK`` (−30%, Shumway 1997) for
performance-related delistings and 0% otherwise (e.g. cash mergers).
"""

from __future__ import annotations

import polars as pl

PERFORMANCE_DELISTING_FALLBACK = -0.30

_KEY = "security_id"


def total_returns(
    prices: pl.DataFrame,
    splits: pl.DataFrame,
    dividends: pl.DataFrame,
    delistings: pl.DataFrame,
    performance_fallback: float = PERFORMANCE_DELISTING_FALLBACK,
) -> pl.DataFrame:
    """
    Daily total-return panel.

    Returns
    -------
    pl.DataFrame
        ``date, security_id, ret, has_action`` sorted by ``security_id, date``.
        ``ret`` is null on a security's first priced session.  ``has_action``
        marks a split, dividend or delisting on that row.
    """
    if delistings[_KEY].is_duplicated().any():
        raise ValueError("delistings must have at most one row per security_id")

    px = (
        prices.select("date", _KEY, pl.col("close").fill_nan(None))
        .sort(_KEY, "date")
    )
    split_f = _on_priced_session(px, splits, "ratio").group_by(_KEY, "date").agg(pl.col("ratio").product())
    divs = _on_priced_session(px, dividends, "amount").group_by(_KEY, "date").agg(pl.col("amount").sum())

    rets = (
        px.join(split_f, on=[_KEY, "date"], how="left")
        .join(divs, on=[_KEY, "date"], how="left")
        .with_columns(
            ((pl.col("close") + pl.col("amount").fill_null(0.0)) * pl.col("ratio").fill_null(1.0)
             / pl.col("close").shift().over(_KEY) - 1).alias("ret"),
            (pl.col("ratio").is_not_null() | pl.col("amount").is_not_null()).alias("has_action"),
        )
        .select("date", _KEY, "ret", "has_action")
    )

    dl = delistings.select(
        _KEY,
        pl.col("delist_date").alias("date"),
        pl.coalesce(
            pl.col("delist_return").fill_nan(None),
            pl.when(pl.col("performance_related")).then(performance_fallback).otherwise(0.0),
        ).alias("dlret"),
    )
    return (
        rets.join(dl, on=[_KEY, "date"], how="full", coalesce=True)
        .join(dl.select(_KEY, pl.col("date").alias("delist_date")), on=_KEY, how="left")
        .filter(pl.col("delist_date").is_null() | (pl.col("date") <= pl.col("delist_date")))
        .with_columns(
            pl.when(pl.col("dlret").is_null())
            .then(pl.col("ret"))
            .otherwise((1 + pl.col("ret").fill_null(0.0)) * (1 + pl.col("dlret")) - 1)
            .alias("ret"),
            (pl.col("has_action").fill_null(False) | pl.col("dlret").is_not_null()).alias("has_action"),
        )
        .select("date", _KEY, "ret", "has_action")
        .sort(_KEY, "date")
    )


def _on_priced_session(px: pl.DataFrame, actions: pl.DataFrame, value: str) -> pl.DataFrame:
    """Move each action to the first priced session on/after its ex_date, so none is lost on a gap day."""
    sessions = px.select(_KEY, "date", pl.col("date").alias("session")).sort("date")
    return (
        actions.select(_KEY, pl.col("ex_date").alias("date"), value)
        .sort("date")
        .join_asof(sessions, on="date", by=_KEY, strategy="forward", check_sortedness=False)
        .filter(pl.col("session").is_not_null())
        .select(_KEY, pl.col("session").alias("date"), value)
    )
