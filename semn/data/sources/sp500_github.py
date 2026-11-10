"""
semn/data/sources/sp500_github.py
==========================
Free historical S&P 500 membership from github.com/fja05680/sp500 (MIT licence).

The file has one row per index-change date: ``date, tickers`` where ``tickers``
is the comma-separated list in force from that date until the next row.  Tickers
are the symbols in use *on that date* (FB before 2022-06-09, META after).

Free-data limits (decision D1): compiled from public sources, not S&P; no
security identifiers, so renames are linked later by
:func:`semn.data.sources.tiingo.build_reference`.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

#: Pinned to a commit so every build reads the same file (recorded in manifests).
COMMIT = "a2430f2af0c79ddf0748e91de11bdeb1616ab5a7"
URL = (
    f"https://raw.githubusercontent.com/fja05680/sp500/{COMMIT}/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
FAR = date(9999, 12, 31)


def normalise_ticker(col: str = "ticker") -> pl.Expr:
    """``BRK.B`` → ``BRK-B`` (the vendor's class-share format), upper case."""
    return pl.col(col).str.strip_chars().str.to_uppercase().str.replace_all(r"\.", "-")


def load_membership(path: str | Path) -> pl.DataFrame:
    """
    Membership intervals from the change-date file.

    Returns
    -------
    pl.DataFrame
        ``ticker, index, entry_date, exit_date`` with ``[entry, exit)`` intervals;
        still-listed members have ``exit_date = 9999-12-31``.
    """
    rows = (
        pl.read_csv(path, schema_overrides={"date": pl.Utf8, "tickers": pl.Utf8})
        .select(pl.col("date").str.to_date(), "tickers")
        .sort("date")
        .with_columns(pl.col("date").shift(-1).alias("next_date"))  # null on the last row: still a member
        .with_row_index("row")
        .with_columns(pl.col("row").cast(pl.Int64))
    )
    return (
        rows.with_columns(pl.col("tickers").str.split(","))
        .explode("tickers")
        .rename({"tickers": "ticker"})
        .filter(pl.col("ticker").str.strip_chars() != "")
        .with_columns(normalise_ticker())
        .unique(["row", "ticker"])
        .sort("ticker", "row")
        # A new run starts whenever the ticker is missing from the previous change row.
        .with_columns(
            (pl.col("row") != pl.col("row").shift().over("ticker") + 1)
            .fill_null(True).cum_sum().over("ticker").alias("run")
        )
        .group_by("ticker", "run")
        .agg(pl.col("date").min().alias("entry_date"), pl.col("next_date").last().alias("exit_date"))
        .select("ticker", pl.lit("SP500").alias("index"), "entry_date", pl.col("exit_date").fill_null(FAR))
        .sort("ticker", "entry_date")
    )
