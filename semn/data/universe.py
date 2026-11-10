"""
semn/data/universe.py
==========================
Point-in-time universe construction for the SEMN platform (Phase 1).

Design principles
-----------------
* **Keyed by ``security_id``** — a permanent internal identifier.  Tickers
  change (FB → META) and get reused by unrelated companies, so they never
  appear in joins here.
* **No survivorship bias** — a security is eligible on rebalance date ``t``
  only when ``entry_date <= t < exit_date``.  Still-active members carry the
  far-future sentinel ``9999-12-31``.
* **Raw prices only** — ``close`` and ``volume`` must be **raw, unadjusted**
  vendor values.  The $ price floor and ADV are defined on what actually
  traded that day; split-adjusted history would let a pre-split stock fail
  the price floor it really passed.  ADV₂₀ is computed here from the same raw
  panel rather than accepted as an input, so the two can never disagree.
* **Point-in-time** — every value used at ``t`` is the last observation on or
  before ``t`` (``join_asof`` backward).  Ages are counted in NYSE sessions
  (:func:`semn.data.calendar.sessions`).  A price older than ``max_staleness``
  sessions excludes the security (reported by :meth:`UniverseBuilder.exclusions`);
  a market cap that old is treated as missing.
* **Clean inputs** — NaN close/volume are treated as missing, duplicate
  ``(date, security_id)`` rows are rejected, and rows on non-session dates
  are ignored (``semn.data.quality`` reports them).
* **Vectorized** — a whole panel is one cross-join + ``join_asof`` pass, not a
  per-date loop.

Expected input schemas
----------------------
membership_df:
    ``security_id, index (Utf8), entry_date (Date), exit_date (Date)``

price_df:
    ``date (Date), security_id, close (Float64), volume (Float64)``
    One row per (session date, security), raw unadjusted close and share volume.

mcap_df:
    ``date (Date), security_id, market_cap (Float64)``, daily.

sessions:
    NYSE session dates covering the price history and every rebalance date.

Usage
-----
>>> from semn.data import LiquidityFilter, UniverseBuilder, calendar
>>> sessions = calendar.sessions("2010-01-01", "2024-12-31")
>>> rebalance_dates = calendar.rebalance_dates(sessions, "1mo")
>>> builder = UniverseBuilder(membership, prices, mcap, sessions)
>>> panel = builder.build_panel(rebalance_dates, index="SP500", **LiquidityFilter().as_kwargs())
>>> report = builder.exclusions(rebalance_dates, index="SP500", **LiquidityFilter().as_kwargs())
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Optional, Sequence

import polars as pl

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

#: Output schema of :meth:`UniverseBuilder.build_panel` (``security_id`` keeps the input dtype).
UNIVERSE_SCHEMA: dict[str, pl.PolarsDataType] = {
    "date": pl.Date,
    "security_id": pl.Utf8,
    "index": pl.Utf8,
    "close": pl.Float64,
    "adv_20d": pl.Float64,
    "market_cap": pl.Float64,
}

_KEY = "security_id"
_MEMBERSHIP_REQUIRED = {_KEY, "index", "entry_date", "exit_date"}
_PRICE_REQUIRED = {"date", _KEY, "close", "volume"}
_MCAP_REQUIRED = {"date", _KEY, "market_cap"}

#: ADV window in observations; also the minimum history required.
ADV_WINDOW = 20


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class UniverseBuilder:
    """
    Constructs a point-in-time eligible universe for the SEMN platform.

    Parameters
    ----------
    membership_df:
        Index membership: ``security_id, index, entry_date, exit_date``.
    price_df:
        Raw daily panel: ``date, security_id, close, volume``.
    mcap_df:
        Market capitalisation panel: ``date, security_id, market_cap``.
    sessions:
        NYSE session dates (:func:`semn.data.calendar.sessions`).
    """

    def __init__(
        self,
        membership_df: pl.DataFrame,
        price_df: pl.DataFrame,
        mcap_df: pl.DataFrame,
        sessions: pl.Series,
    ) -> None:
        _validate_schema(membership_df, _MEMBERSHIP_REQUIRED, "membership_df")
        _validate_schema(price_df, _PRICE_REQUIRED, "price_df")
        _validate_schema(mcap_df, _MCAP_REQUIRED, "mcap_df")

        self._membership = _coerce_date_cols(membership_df, ["entry_date", "exit_date"])

        price = _coerce_date_cols(price_df, ["date"]).with_columns(
            pl.col("close", "volume").fill_nan(None)
        )
        dupes = price.filter(pl.struct("date", _KEY).is_duplicated())
        if not dupes.is_empty():
            raise ValueError(
                f"price_df has {dupes.height} duplicate (date, security_id) rows, e.g. {dupes.row(0)}"
            )
        self._sessions = (
            sessions.cast(pl.Date).unique().sort().to_frame("date")
            .with_row_index("session")
            .with_columns(pl.col("session").cast(pl.Int64))
        )
        self._price = (
            price.join(self._sessions, on="date")  # rows on non-session dates are dropped
            .sort(_KEY, "date")
            .with_columns(
                (pl.col("close") * pl.col("volume"))
                .rolling_mean(ADV_WINDOW)  # min_samples defaults to the window
                .over(_KEY)
                .alias("adv_20d")
            )
            .select("date", _KEY, "close", "adv_20d", "session")
            .sort("date")
        )
        self._mcap = (
            _coerce_date_cols(mcap_df, ["date"])
            .select("date", _KEY, pl.col("market_cap").fill_nan(None))
            .sort("date")
            .join_asof(self._sessions.rename({"session": "mcap_session"}), on="date")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, as_of_date: str | date, **kwargs) -> pl.DataFrame:
        """Eligible universe on a single date; same keyword arguments as :meth:`build_panel`."""
        return self.build_panel([as_of_date], **kwargs)

    def build_panel(
        self,
        dates: Sequence[str | date],
        index: Optional[str] = None,
        adv_floor: float = 0.0,
        price_floor: float = 0.0,
        mcap_floor: float = 0.0,
        max_staleness: int = 5,
    ) -> pl.DataFrame:
        """
        Eligible universe on every date in *dates*, long format.

        Parameters
        ----------
        dates:
            Rebalance dates (ISO-8601 strings or :class:`datetime.date`).
        index:
            Restrict to one index (e.g. ``"SP500"``).  ``None`` keeps all
            indexes, one row per security (``index`` = first label alphabetically).
        adv_floor:
            Minimum ADV₂₀ in USD (raw close × raw volume).
        price_floor:
            Minimum raw close in USD.
        mcap_floor:
            Minimum market cap in USD.  ``0`` disables the filter, so a missing
            market cap does not exclude a security.
        max_staleness:
            Maximum age, in sessions, of the last price observation.

        Returns
        -------
        pl.DataFrame
            ``date, security_id, index, close, adv_20d, market_cap``, sorted by
            ``date, security_id``.
        """
        return (
            self._candidates(dates, index, adv_floor, price_floor, mcap_floor, max_staleness)
            .filter(pl.col("exclusion").is_null())
            .drop("exclusion")
        )

    def exclusions(self, dates: Sequence[str | date], **kwargs) -> pl.DataFrame:
        """
        Index members that were dropped, with an ``exclusion`` reason.

        Reasons: ``no_data``, ``adv_history`` (fewer than ``ADV_WINDOW``
        observations), ``stale``, ``price_floor``, ``adv_floor``, ``mcap_floor``.
        Same keyword arguments as :meth:`build_panel`.
        """
        return self._candidates(dates, **kwargs).filter(pl.col("exclusion").is_not_null())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _candidates(
        self,
        dates: Sequence[str | date],
        index: Optional[str] = None,
        adv_floor: float = 0.0,
        price_floor: float = 0.0,
        mcap_floor: float = 0.0,
        max_staleness: int = 5,
    ) -> pl.DataFrame:
        """Every PIT index member on every date, with its exclusion reason (null = eligible)."""
        rebalance = pl.DataFrame({"date": [_parse_date(d) for d in dates]}, schema={"date": pl.Date})
        last_session = self._sessions["date"].max()
        if rebalance.height and (last_session is None or rebalance["date"].max() > last_session):
            raise ValueError(f"rebalance dates go past the last session ({last_session}); extend `sessions`")
        rebalance = (
            rebalance
            .unique()
            .sort("date")
            .join_asof(self._sessions.rename({"session": "as_of_session"}), on="date")
        )
        members = self._membership
        if index is not None:
            members = members.filter(pl.col("index") == index)

        panel = (
            rebalance.join(members.select(_KEY, "index", "entry_date", "exit_date"), how="cross")
            .filter((pl.col("entry_date") <= pl.col("date")) & (pl.col("date") < pl.col("exit_date")))
            .sort("date")
            .join_asof(self._price, on="date", by=_KEY, check_sortedness=False)
            .join_asof(self._mcap, on="date", by=_KEY, check_sortedness=False)
            .with_columns(
                pl.when(pl.col("as_of_session") - pl.col("mcap_session") <= max_staleness)
                .then(pl.col("market_cap"))
                .alias("market_cap")
            )
        )
        if index is None:
            # A security in several indexes would otherwise appear once per index.
            panel = panel.sort("date", _KEY, "index").unique(subset=["date", _KEY], keep="first", maintain_order=True)

        mcap_fails = (
            pl.col("market_cap").is_null() | (pl.col("market_cap") < mcap_floor)
            if mcap_floor > 0
            else pl.lit(False)
        )
        reason = (
            pl.when(pl.col("close").is_null()).then(pl.lit("no_data"))
            .when(pl.col("as_of_session") - pl.col("session") > max_staleness).then(pl.lit("stale"))
            .when(pl.col("adv_20d").is_null()).then(pl.lit("adv_history"))
            .when(pl.col("close") < price_floor).then(pl.lit("price_floor"))
            .when(pl.col("adv_20d") < adv_floor).then(pl.lit("adv_floor"))
            .when(mcap_fails).then(pl.lit("mcap_floor"))
        )
        return (
            panel.with_columns(reason.alias("exclusion"))
            .select("date", _KEY, "index", "close", "adv_20d", "market_cap", "exclusion")
            .sort("date", _KEY)
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
    """Coerce *cols* to :class:`polars.Date` (strings via ``str.to_date``)."""
    exprs = [
        pl.col(c).str.to_date() if df[c].dtype == pl.Utf8 else pl.col(c).cast(pl.Date)
        for c in cols
        if df[c].dtype != pl.Date
    ]
    return df.with_columns(exprs) if exprs else df


def _validate_schema(df: pl.DataFrame, required_cols: set[str], name: str) -> None:
    """Raise :class:`ValueError` if *df* is missing any *required_cols*."""
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {sorted(missing)}. Got: {df.columns}"
        )


# ---------------------------------------------------------------------------
# Filter parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LiquidityFilter:
    """
    Roadmap universe filter defaults, for ``build_panel(**f.as_kwargs())``.

    ADV₂₀ ≥ $25M, raw close ≥ $5, no market-cap floor, last price at most
    5 sessions old.
    """

    adv_floor: float = 25_000_000.0
    price_floor: float = 5.0
    mcap_floor: float = 0.0
    max_staleness: int = 5

    def as_kwargs(self) -> dict[str, float]:
        """Return filter thresholds as a keyword-argument dict."""
        return asdict(self)
