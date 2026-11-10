"""
semn/data/clickhouse_loader.py
====================================
Columnar tick & bar pipeline for the SEMN platform (Phase 1).

Design principles
-----------------
* **Point-in-time (PIT) safety** — every query injects ``ts < pit_cutoff`` so
  no future data can leak into research or backtesting contexts.
* **Polars-native** — all results are returned as :class:`polars.DataFrame`
  with proper dtypes.  Downstream consumers never need pandas for data IO.
* **Resample in Python** — bar construction from raw ticks is done in Polars
  so it works identically whether the data comes from ClickHouse or a local
  Parquet fixture, keeping the resampling logic testable without a live DB.
* **Dependency-injectable client** — the ``client`` constructor argument
  accepts any object that exposes ``query_df(sql, params) -> pd.DataFrame``,
  making unit tests trivially mockable without a real ClickHouse instance.

Usage
-----
>>> from semn.data import ClickHouseLoader
>>> loader = ClickHouseLoader(host="localhost", port=8123,
...                           database="market", user="default",
...                           password="")
>>> bars = loader.get_bar_data(
...     symbols=["AAPL", "MSFT"],
...     start="2024-01-02",
...     end="2024-06-30",
...     freq="1d",
...     pit_cutoff="2024-07-01T00:00:00",
... )
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional, Protocol, Sequence

import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

#: Expected column names and dtypes for the raw ticks table.
TICK_SCHEMA: dict[str, pl.PolarsDataType] = {
    "ts": pl.Datetime("us", "UTC"),
    "symbol": pl.Utf8,
    "price": pl.Float64,
    "size": pl.Float64,
    "side": pl.Int8,  # 1=buy, -1=sell, 0=unknown
}

#: Expected column names and dtypes for the bars table / resampled output.
BAR_SCHEMA: dict[str, pl.PolarsDataType] = {
    "ts": pl.Datetime("us", "UTC"),
    "symbol": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "vwap": pl.Float64,
    "session_date": pl.Date,  # exchange-local (New York) date of the bar
}

#: Exchange time zone; daily bars and session dates follow its local calendar.
EXCHANGE_TZ = "America/New_York"

# Polars truncate aliases for common bar frequencies.
_FREQ_ALIASES: dict[str, str] = {
    "1t": "1m",
    "1min": "1m",
    "5t": "5m",
    "5min": "5m",
    "15t": "15m",
    "15min": "15m",
    "30t": "30m",
    "30min": "30m",
}


# ---------------------------------------------------------------------------
# Client protocol — allows injection of mock clients in tests
# ---------------------------------------------------------------------------

class ClickHouseClientProtocol(Protocol):
    """Minimal interface required from a ClickHouse client."""

    def query_df(self, query: str, parameters: Optional[dict] = None):
        """Execute *query* and return a pandas DataFrame."""
        ...


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class ClickHouseLoader:
    """
    High-performance ClickHouse data loader for the SEMN platform.

    Parameters
    ----------
    host:
        ClickHouse server hostname.
    port:
        ClickHouse HTTP port (default 8123).
    database:
        Database name that contains the ``ticks`` and ``bars`` tables.
    user:
        Database username.
    password:
        Database password.
    client:
        Optional pre-built client object that satisfies
        :class:`ClickHouseClientProtocol`.  When supplied, ``host``,
        ``port``, ``database``, ``user``, and ``password`` are ignored.
        This is the primary injection point for unit tests.
    ticks_table:
        Name of the raw ticks table (default ``"ticks"``).
    bars_table:
        Name of the pre-aggregated bars table (default ``"bars"``).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8123,
        database: str = "market",
        user: str = "default",
        password: str = "",
        *,
        client: Optional[ClickHouseClientProtocol] = None,
        ticks_table: str = "ticks",
        bars_table: str = "bars",
    ) -> None:
        self._ticks_table = ticks_table
        self._bars_table = bars_table

        if client is not None:
            self._client = client
        else:
            try:
                import clickhouse_connect  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "clickhouse-connect is required for a live connection.  "
                    "Install it with: pip install 'semn[clickhouse]'"
                ) from exc

            self._client: ClickHouseClientProtocol = (
                clickhouse_connect.get_client(  # type: ignore[assignment]
                    host=host,
                    port=port,
                    database=database,
                    username=user,
                    password=password,
                )
            )
            logger.info(
                "ClickHouseLoader connected to %s:%d/%s", host, port, database
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tick_data(
        self,
        symbols: Sequence[str],
        start: str | datetime,
        end: str | datetime,
        pit_cutoff: Optional[str | datetime] = None,
    ) -> pl.DataFrame:
        """
        Fetch raw tick data for *symbols* in [*start*, *end*).

        Parameters
        ----------
        symbols:
            Sequence of ticker symbols to fetch.
        start:
            Inclusive lower bound for ``ts`` (ISO-8601 string or datetime).
        end:
            Exclusive upper bound for ``ts``.
        pit_cutoff:
            Hard point-in-time ceiling.  Any tick with ``ts >= pit_cutoff``
            is excluded.  Defaults to ``end`` when not specified.

        Returns
        -------
        pl.DataFrame
            Schema: ``ts, symbol, price, size, side``.
        """
        start_dt = _parse_dt(start)
        end_dt = _parse_dt(end)
        cutoff_dt = _parse_dt(pit_cutoff) if pit_cutoff is not None else end_dt

        # PIT: effective end is the earlier of end and pit_cutoff
        effective_end = min(end_dt, cutoff_dt)

        sql = f"""
            SELECT
                ts,
                symbol,
                price,
                size,
                side
            FROM {self._ticks_table}
            WHERE has({{symbols:Array(String)}}, symbol)
              AND ts >= {{start:DateTime64(6, 'UTC')}}
              AND ts  < {{end:DateTime64(6, 'UTC')}}
            ORDER BY symbol, ts
        """
        params = {
            "symbols": list(symbols),
            "start": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "end": effective_end.strftime("%Y-%m-%d %H:%M:%S"),
        }
        logger.debug(
            "get_tick_data: symbols=%s start=%s end=%s (pit=%s)",
            symbols, start_dt, effective_end, cutoff_dt,
        )
        return self._run_query(sql, params, TICK_SCHEMA)

    def get_bar_data(
        self,
        symbols: Sequence[str],
        start: str | datetime,
        end: str | datetime,
        freq: str = "1d",
        pit_cutoff: Optional[str | datetime] = None,
        *,
        from_ticks: bool = False,
    ) -> pl.DataFrame:
        """
        Fetch OHLCV bar data for *symbols* in [*start*, *end*).

        Parameters
        ----------
        symbols:
            Sequence of ticker symbols to fetch.
        start:
            Inclusive lower bound for bar ``ts``.
        end:
            Exclusive upper bound for bar ``ts``.
        freq:
            Bar frequency string (e.g. ``"1d"``, ``"1h"``, ``"5min"``).
        pit_cutoff:
            Point-in-time ceiling; bars with ``ts >= pit_cutoff`` are
            excluded.  Defaults to ``end``.
        from_ticks:
            When ``True``, fetch raw ticks and resample in Polars rather
            than reading from the pre-aggregated bars table.  Useful when
            the bars table does not carry the desired frequency.

        Returns
        -------
        pl.DataFrame
            Schema: ``ts, symbol, open, high, low, close, volume, vwap,
            session_date``.  ``ts`` is a UTC instant inside the bar;
            ``session_date`` is its New York local date, the key for daily data.
        """
        start_dt = _parse_dt(start)
        end_dt = _parse_dt(end)
        cutoff_dt = _parse_dt(pit_cutoff) if pit_cutoff is not None else end_dt
        effective_end = min(end_dt, cutoff_dt)

        if from_ticks:
            tick_df = self.get_tick_data(
                symbols, start_dt, effective_end, pit_cutoff=effective_end
            )
            return self.resample_ticks_to_bars(tick_df, freq)

        norm_freq = _normalise_freq(freq)
        sql = f"""
            SELECT
                ts,
                symbol,
                open,
                high,
                low,
                close,
                volume,
                vwap,
                toDate(ts, '{EXCHANGE_TZ}') AS session_date
            FROM {self._bars_table}
            WHERE has({{symbols:Array(String)}}, symbol)
              AND freq   = {{freq:String}}
              AND ts    >= {{start:DateTime64(6, 'UTC')}}
              AND ts     < {{end:DateTime64(6, 'UTC')}}
            ORDER BY symbol, ts
        """
        params = {
            "symbols": list(symbols),
            "freq": norm_freq,
            "start": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "end": effective_end.strftime("%Y-%m-%d %H:%M:%S"),
        }
        logger.debug(
            "get_bar_data: symbols=%s freq=%s start=%s end=%s (pit=%s)",
            symbols, norm_freq, start_dt, effective_end, cutoff_dt,
        )
        return self._run_query(sql, params, BAR_SCHEMA)

    @staticmethod
    def resample_ticks_to_bars(tick_df: pl.DataFrame, freq: str) -> pl.DataFrame:
        """
        Resample a tick :class:`polars.DataFrame` into OHLCV bars.

        This is a **pure Polars** operation — no database connection needed.
        Useful for testing with synthetic data or for on-the-fly resampling
        after a ``get_tick_data`` call.

        Parameters
        ----------
        tick_df:
            DataFrame with columns ``ts, symbol, price, size`` at minimum.
            The ``ts`` column must be ``Datetime`` dtype (UTC).
        freq:
            Target bar frequency (e.g. ``"1d"``, ``"1h"``, ``"5m"``).

        Returns
        -------
        pl.DataFrame
            Schema: :data:`BAR_SCHEMA`.  Bars are truncated in New York local
            time.  Only periods with at least one tick are returned (no empty bars).
        """
        # Validate required columns are present (before empty check so error always fires)
        required = {"ts", "symbol", "price", "size"}
        missing = required - set(tick_df.columns)
        if missing:
            raise ValueError(
                f"tick_df is missing required columns: {missing}"
            )

        if tick_df.is_empty():
            return pl.DataFrame(schema=BAR_SCHEMA)

        # Truncate in exchange-local time so a "1d" bar is one New York session date,
        # not a UTC midnight-to-midnight window that splits the evening.
        # shortcut: includes pre/post-market ticks; clip to calendar open/close when tick data (D3) is in use.
        bar_local = (
            pl.col("ts")
            .dt.convert_time_zone(EXCHANGE_TZ)
            .dt.truncate(_normalise_freq(freq))
        )
        bars = (
            tick_df
            .sort("ts")
            .with_columns(
                bar_local.dt.convert_time_zone("UTC").alias("bar_ts"),
                bar_local.dt.date().alias("session_date"),
            )
            .group_by(["symbol", "bar_ts", "session_date"])
            .agg(
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
                pl.col("size").sum().alias("volume"),
                # VWAP = sum(price * size) / sum(size)
                (
                    (pl.col("price") * pl.col("size")).sum()
                    / pl.col("size").sum()
                ).alias("vwap"),
            )
            .rename({"bar_ts": "ts"})
            .select(list(BAR_SCHEMA))
            .sort(["symbol", "ts"])
        )
        return bars

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_query(
        self,
        sql: str,
        params: dict,
        schema: dict[str, pl.PolarsDataType],
    ) -> pl.DataFrame:
        """Execute *sql* via the injected client and return a polars DataFrame."""
        try:
            pandas_df = self._client.query_df(sql, parameters=params)
        except Exception as exc:
            logger.error("ClickHouse query failed: %s", exc, exc_info=True)
            raise

        if pandas_df.empty:
            return pl.DataFrame(schema=schema)

        df = pl.from_pandas(pandas_df)
        df = _cast_to_schema(df, schema)
        return df


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_dt(value: str | datetime) -> datetime:
    """
    Parse to a UTC datetime.  Values without a time zone are New York time, so
    ``"2024-01-02"`` means the start of that session date, not 19:00 the day before.
    """
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(EXCHANGE_TZ))
    return dt.astimezone(timezone.utc)


def _normalise_freq(freq: str) -> str:
    """Normalise a user-supplied frequency string to the canonical form stored in ClickHouse."""
    return _FREQ_ALIASES.get(freq.lower(), freq.lower())



def _cast_to_schema(
    df: pl.DataFrame,
    schema: dict[str, pl.PolarsDataType],
) -> pl.DataFrame:
    """
    Cast a DataFrame to the target *schema*, ignoring extra columns.

    Raises
    ------
    ValueError
        If a required column is absent from *df*.
    """
    missing = [col for col in schema if col not in df.columns]
    if missing:
        raise ValueError(
            f"Query result is missing expected columns: {missing}. "
            f"Got: {df.columns}"
        )
    cast_exprs = [
        pl.col(col).cast(dtype) for col, dtype in schema.items()
    ]
    return df.select(cast_exprs)
