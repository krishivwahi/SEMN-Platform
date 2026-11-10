"""
semn/data/security_master.py
==========================
Permanent ``security_id`` with dated ticker and GICS history (Phase 1).

Tickers are display labels: they change (FB → META) and are reused by unrelated
companies.  Vendor rows keyed by ticker are mapped to ``security_id`` here, by
date, before anything else touches them.  All intervals are ``[valid_from, valid_to)``
with ``9999-12-31`` for "still valid".

Input schemas
-------------
tickers: ``security_id, ticker (Utf8), valid_from (Date), valid_to (Date)``
gics:    ``security_id, gics_sector (Utf8), gics_industry (Utf8), valid_from (Date), valid_to (Date)``
"""

from __future__ import annotations

import polars as pl

_INTERVAL = ["valid_from", "valid_to"]


class SecurityMaster:
    def __init__(self, tickers: pl.DataFrame, gics: pl.DataFrame) -> None:
        _require(tickers, {"security_id", "ticker", *_INTERVAL}, "tickers")
        _require(gics, {"security_id", "gics_sector", "gics_industry", *_INTERVAL}, "gics")
        # A security has one ticker at a time, and a ticker names one security at a time.
        _check_no_overlap(tickers, "security_id", "tickers")
        _check_no_overlap(tickers, "ticker", "tickers")
        _check_no_overlap(gics, "security_id", "gics")
        self.tickers = tickers
        self.gics = gics

    def map_tickers(self, df: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
        """
        Replace ``ticker`` with the ``security_id`` it named on each row's date.

        Rows whose ticker is unknown on that date are dropped; find them with
        :meth:`unmapped`.
        """
        return _as_of(df, self.tickers, "ticker", ["security_id"], date_col).drop("ticker")

    def unmapped(self, df: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
        """Rows of *df* whose ticker names no security on that date."""
        rows = df.with_row_index("_row")
        mapped = _as_of(rows, self.tickers, "ticker", [], date_col)
        return rows.join(mapped.select("_row"), on="_row", how="anti").drop("_row")

    def attach_gics(self, df: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
        """Add the point-in-time ``gics_sector`` and ``gics_industry`` (null if unknown on that date)."""
        rows = df.with_row_index("_row")
        cols = ["gics_sector", "gics_industry"]
        known = _as_of(rows, self.gics, "security_id", cols, date_col).select("_row", *cols)
        return rows.join(known, on="_row", how="left").drop("_row")


def _as_of(df: pl.DataFrame, history: pl.DataFrame, key: str, value_cols: list[str], date_col: str) -> pl.DataFrame:
    """Inner-join *df* to the *history* row whose interval contains the row's date."""
    return (
        df.join(history.select(key, *value_cols, *_INTERVAL), on=key, how="inner", maintain_order="left")
        .filter((pl.col("valid_from") <= pl.col(date_col)) & (pl.col(date_col) < pl.col("valid_to")))
        .drop(_INTERVAL)
    )


def _check_no_overlap(history: pl.DataFrame, key: str, name: str) -> None:
    """Raise if two intervals for the same *key* overlap."""
    clash = (
        history.sort(key, "valid_from")
        .with_columns(pl.col("valid_to").shift().over(key).alias("prev_to"))
        .filter(pl.col("valid_from") < pl.col("prev_to"))
    )
    if not clash.is_empty():
        raise ValueError(f"{name}: overlapping intervals for {key}: {clash[key].unique().to_list()[:10]}")


def _require(df: pl.DataFrame, cols: set[str], name: str) -> None:
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
