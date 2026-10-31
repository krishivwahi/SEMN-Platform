"""
semn/data/universe.py
==========================
Point-in-time universe construction for the SEMN platform (Phase 1).

Design principles
-----------------
* **No survivorship bias** — the membership table carries both ``entry_date``
  and ``exit_date`` for every index constituent.  A symbol is eligible on
  ``as_of_date`` only when ``entry_date <= as_of_date < exit_date``.
  Delisted symbols have their ``exit_date`` set to the actual last trading
  date; still-active symbols carry the far-future sentinel ``9999-12-31``.
* **Point-in-time liquidity** — ADV, price, and market-cap data are sliced to
  the last available observation *on or before* ``as_of_date`` using
  :meth:`_pit_slice`, so no forward-filled data from a future date leaks in.
* **Polars-native** — all inputs and outputs are :class:`polars.DataFrame`.
  External callers inject DataFrames; this class contains no file IO.
* **Testable without fixtures** — all data dependencies are passed in via the
  constructor, allowing unit tests to use tiny synthetic DataFrames.

Expected input schemas
----------------------
membership_df:
    ``symbol (Utf8), index (Utf8), entry_date (Date), exit_date (Date)``

price_df:
    ``date (Date), symbol (Utf8), close (Float64)``
    One row per (date, symbol).  Close price used for the price floor filter.

adv_df:
    ``date (Date), symbol (Utf8), adv_20d (Float64)``
    20-day average daily dollar volume in USD.

mcap_df:
    ``date (Date), symbol (Utf8), market_cap (Float64)``
    Market capitalisation in USD.

Usage
-----
>>> import polars as pl
>>> from semn.data import UniverseBuilder
>>> builder = UniverseBuilder(
...     membership_df=membership,
...     price_df=prices,
...     adv_df=adv,
...     mcap_df=mcap,
... )
>>> universe = builder.build(
...     as_of_date="2024-03-01",
...     index="SP500",
...     adv_floor=1_000_000,
...     price_floor=5.0,
...     mcap_floor=300_000_000,
... )
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional, Sequence

import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

#: Output schema of :meth:`UniverseBuilder.build`.
UNIVERSE_SCHEMA: dict[str, pl.PolarsDataType] = {
    "date": pl.Date,
    "symbol": pl.Utf8,
    "index": pl.Utf8,
    "close": pl.Float64,
    "adv_20d": pl.Float64,
    "market_cap": pl.Float64,
}

#: Required columns in the membership input DataFrame.
_MEMBERSHIP_REQUIRED = {"symbol", "index", "entry_date", "exit_date"}

#: Required columns in the price input DataFrame.
_PRICE_REQUIRED = {"date", "symbol", "close"}

#: Required columns in the ADV input DataFrame.
_ADV_REQUIRED = {"date", "symbol", "adv_20d"}

#: Required columns in the market-cap input DataFrame.
_MCAP_REQUIRED = {"date", "symbol", "market_cap"}

# Sentinel far-future date used for still-active symbols.
_FAR_FUTURE: date = date(9999, 12, 31)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class UniverseBuilder:
    """
    Constructs a point-in-time eligible universe for the SEMN platform.

    Parameters
    ----------
    membership_df:
        Index constituent membership with ``symbol, index, entry_date,
        exit_date`` columns.  ``exit_date`` must be ``9999-12-31`` for
        still-active members.
    price_df:
        Daily close price panel: ``date, symbol, close``.
    adv_df:
        20-day average daily dollar volume panel: ``date, symbol, adv_20d``.
    mcap_df:
        Daily market capitalisation panel: ``date, symbol, market_cap``.
    """

    def __init__(
        self,
        membership_df: pl.DataFrame,
        price_df: pl.DataFrame,
        adv_df: pl.DataFrame,
        mcap_df: pl.DataFrame,
    ) -> None:
        _validate_schema(membership_df, _MEMBERSHIP_REQUIRED, "membership_df")
        _validate_schema(price_df, _PRICE_REQUIRED, "price_df")
        _validate_schema(adv_df, _ADV_REQUIRED, "adv_df")
        _validate_schema(mcap_df, _MCAP_REQUIRED, "mcap_df")

        # Ensure date columns have the right type.
        # Polars 2.x removed string→Date casting via .cast(); use str.to_date().
        self._membership = _coerce_date_cols(membership_df, ["entry_date", "exit_date"])
        self._price = _coerce_date_cols(price_df, ["date"])
        self._adv = _coerce_date_cols(adv_df, ["date"])
        self._mcap = _coerce_date_cols(mcap_df, ["date"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        as_of_date: str | date,
        index: Optional[str] = None,
        adv_floor: float = 0.0,
        price_floor: float = 0.0,
        mcap_floor: float = 0.0,
    ) -> pl.DataFrame:
        """
        Return the eligible universe on *as_of_date*.

        Parameters
        ----------
        as_of_date:
            The point-in-time "today".  Only data available *on or before*
            this date is used.
        index:
            Index filter (e.g. ``"SP500"``, ``"R1000"``).  When ``None``,
            all indices in the membership table are considered.
        adv_floor:
            Minimum 20-day average daily dollar volume in USD.  Symbols
            below this threshold are excluded.
        price_floor:
            Minimum close price in USD.  Symbols below this threshold are
            excluded (removes micro-caps and near-delisted names).
        mcap_floor:
            Minimum market capitalisation in USD.

        Returns
        -------
        pl.DataFrame
            Columns: ``date, symbol, index, close, adv_20d, market_cap``.
            One row per eligible symbol.  Empty DataFrame if no symbols pass
            all filters on *as_of_date*.
        """
        pit_date = _parse_date(as_of_date)

        # 1. PIT membership filter: entry_date <= pit_date < exit_date
        members = self._membership.filter(
            (pl.col("entry_date") <= pit_date)
            & (pl.col("exit_date") > pit_date)
        )
        if index is not None:
            members = members.filter(pl.col("index") == index)

        if members.is_empty():
            logger.debug(
                "build(%s, index=%s): no members pass PIT filter", pit_date, index
            )
            return pl.DataFrame(schema=UNIVERSE_SCHEMA)

        # 2. PIT-slice each panel to as_of_date (last available <= pit_date)
        price_slice = self._pit_slice(self._price, pit_date, "date").select(
            ["symbol", "close"]
        )
        adv_slice = self._pit_slice(self._adv, pit_date, "date").select(
            ["symbol", "adv_20d"]
        )
        mcap_slice = self._pit_slice(self._mcap, pit_date, "date").select(
            ["symbol", "market_cap"]
        )

        # 3. Join panels onto membership
        universe = (
            members.select(["symbol", "index"])
            .join(price_slice, on="symbol", how="left")
            .join(adv_slice, on="symbol", how="left")
            .join(mcap_slice, on="symbol", how="left")
        )

        # 4. Apply numeric liquidity floors (nulls are treated as failures)
        universe = universe.filter(
            pl.col("close").is_not_null()
            & pl.col("adv_20d").is_not_null()
            & pl.col("market_cap").is_not_null()
            & (pl.col("close") >= price_floor)
            & (pl.col("adv_20d") >= adv_floor)
            & (pl.col("market_cap") >= mcap_floor)
        )

        # 5. Attach the as_of_date column and reorder
        universe = universe.with_columns(
            pl.lit(pit_date).cast(pl.Date).alias("date")
        ).select(["date", "symbol", "index", "close", "adv_20d", "market_cap"])

        logger.debug(
            "build(%s, index=%s): %d symbols eligible "
            "(adv_floor=%s, price_floor=%s, mcap_floor=%s)",
            pit_date, index, len(universe), adv_floor, price_floor, mcap_floor,
        )
        return universe

    def build_panel(
        self,
        dates: Sequence[str | date],
        index: Optional[str] = None,
        adv_floor: float = 0.0,
        price_floor: float = 0.0,
        mcap_floor: float = 0.0,
    ) -> pl.DataFrame:
        """
        Build the eligible universe across multiple dates.

        Calls :meth:`build` for each date in *dates* and returns a
        long-format panel DataFrame.

        Parameters
        ----------
        dates:
            Sequence of as-of dates (ISO-8601 strings or :class:`datetime.date`).
        index:
            Index filter passed through to :meth:`build`.
        adv_floor, price_floor, mcap_floor:
            Liquidity filter thresholds passed through to :meth:`build`.

        Returns
        -------
        pl.DataFrame
            Long-format: ``date, symbol, index, close, adv_20d, market_cap``.
        """
        slices: list[pl.DataFrame] = []
        for d in dates:
            daily = self.build(
                as_of_date=d,
                index=index,
                adv_floor=adv_floor,
                price_floor=price_floor,
                mcap_floor=mcap_floor,
            )
            if not daily.is_empty():
                slices.append(daily)

        if not slices:
            return pl.DataFrame(schema=UNIVERSE_SCHEMA)

        return pl.concat(slices, rechunk=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pit_slice(
        df: pl.DataFrame,
        as_of_date: date,
        ts_col: str,
    ) -> pl.DataFrame:
        """
        Return the last available row per symbol on or before *as_of_date*.

        This prevents forward-filling from future dates and guarantees strict
        point-in-time correctness for price, ADV, and market-cap panels.

        Parameters
        ----------
        df:
            Panel DataFrame with a ``symbol`` column and a date column
            identified by *ts_col*.
        as_of_date:
            The point-in-time ceiling (inclusive).
        ts_col:
            Name of the date/timestamp column in *df*.

        Returns
        -------
        pl.DataFrame
            One row per symbol: the last observation on or before
            *as_of_date*.  Symbols with no data on or before this date are
            absent from the result.
        """
        return (
            df.filter(pl.col(ts_col) <= as_of_date)
            .sort(ts_col)
            .group_by("symbol")
            .last()
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_date(value: str | date) -> date:
    """Parse an ISO-8601 date string or passthrough a :class:`datetime.date`."""
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _coerce_date_cols(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """
    Coerce *cols* in *df* to :class:`polars.Date` dtype.

    Polars 2.x removed implicit string→Date casting via ``.cast(pl.Date)``.
    This helper uses ``str.to_date()`` for string columns and ``cast(pl.Date)``
    for integer or already-Date columns.
    """
    exprs = []
    for col in cols:
        if df[col].dtype == pl.Utf8:
            exprs.append(pl.col(col).str.to_date())
        elif df[col].dtype != pl.Date:
            exprs.append(pl.col(col).cast(pl.Date))
        # if already Date, no-op
    if exprs:
        return df.with_columns(exprs)
    return df


def _validate_schema(
    df: pl.DataFrame,
    required_cols: set[str],
    name: str,
) -> None:
    """Raise :class:`ValueError` if *df* is missing any *required_cols*."""
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {sorted(missing)}. "
            f"Got: {df.columns}"
        )


# ---------------------------------------------------------------------------
# Convenience filter dataclass
# ---------------------------------------------------------------------------

class LiquidityFilter:
    """
    A lightweight container for liquidity filter parameters.

    Designed for use with :meth:`UniverseBuilder.build` and
    :meth:`UniverseBuilder.build_panel` to avoid long keyword argument lists.

    Parameters
    ----------
    adv_floor:
        Minimum 20-day ADV in USD (default 1 million).
    price_floor:
        Minimum close price in USD (default $5).
    mcap_floor:
        Minimum market cap in USD (default $300 million).

    Examples
    --------
    >>> filt = LiquidityFilter(adv_floor=2e6, price_floor=10.0)
    >>> universe = builder.build("2024-01-15", index="SP500", **filt.as_kwargs())
    """

    def __init__(
        self,
        adv_floor: float = 1_000_000.0,
        price_floor: float = 5.0,
        mcap_floor: float = 300_000_000.0,
    ) -> None:
        self.adv_floor = adv_floor
        self.price_floor = price_floor
        self.mcap_floor = mcap_floor

    def as_kwargs(self) -> dict[str, float]:
        """Return filter thresholds as a keyword-argument dict."""
        return {
            "adv_floor": self.adv_floor,
            "price_floor": self.price_floor,
            "mcap_floor": self.mcap_floor,
        }

    def __repr__(self) -> str:
        return (
            f"LiquidityFilter(adv_floor={self.adv_floor:,.0f}, "
            f"price_floor={self.price_floor}, "
            f"mcap_floor={self.mcap_floor:,.0f})"
        )
