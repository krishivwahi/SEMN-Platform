"""
tests/test_data_pipeline.py
============================
Unit tests for Phase 1: ClickHouseLoader and UniverseBuilder.

All tests use synthetic in-memory data — no live ClickHouse connection is
required.  The ClickHouseLoader is exercised via a mock client that satisfies
the ClickHouseClientProtocol interface.
"""

from __future__ import annotations

import sys
import os
from datetime import date, datetime, timezone
from typing import Optional
from unittest.mock import MagicMock

import pandas as pd
import polars as pl
import polars.testing as plt
import pytest

# Ensure the repo root is on sys.path so `platform` is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from semn.data.clickhouse_loader import (
    BAR_SCHEMA,
    TICK_SCHEMA,
    ClickHouseLoader,
    _normalise_freq,
    _parse_dt,
    _to_polars_duration,
)
from semn.data.universe import (
    UNIVERSE_SCHEMA,
    LiquidityFilter,
    UniverseBuilder,
    _parse_date,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_client(pandas_df: pd.DataFrame) -> MagicMock:
    """Return a mock ClickHouse client that always returns *pandas_df*."""
    client = MagicMock()
    client.query_df.return_value = pandas_df
    return client


def _tick_pandas(
    n: int = 10,
    symbol: str = "AAPL",
    base_price: float = 150.0,
    start_ts: str = "2024-01-02 09:30:00",
) -> pd.DataFrame:
    """Create a minimal tick pandas DataFrame for testing."""
    import numpy as np
    rng = pd.date_range(start_ts, periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "ts": rng,
        "symbol": symbol,
        "price": base_price + np.arange(n, dtype=float),
        "size": 100.0,
        "side": 1,
    })


def _bar_pandas(
    n: int = 5,
    symbol: str = "AAPL",
    base_price: float = 150.0,
    start_date: str = "2024-01-02",
) -> pd.DataFrame:
    """Create a minimal bar pandas DataFrame for testing."""
    rng = pd.date_range(start_date, periods=n, freq="B", tz="UTC")
    return pd.DataFrame({
        "ts": rng,
        "symbol": symbol,
        "open": base_price,
        "high": base_price + 2.0,
        "low": base_price - 1.0,
        "close": base_price + 1.0,
        "volume": 1_000_000.0,
        "vwap": base_price + 0.5,
    })


# ---------------------------------------------------------------------------
# ClickHouseLoader — basic construction
# ---------------------------------------------------------------------------

class TestClickHouseLoaderInit:
    def test_init_with_mock_client(self):
        """ClickHouseLoader accepts an injected client without importing clickhouse_connect."""
        mock_client = MagicMock()
        loader = ClickHouseLoader(client=mock_client)
        assert loader._client is mock_client

    def test_init_raises_without_clickhouse_connect(self, monkeypatch):
        """Without a client and without clickhouse_connect installed, raises ImportError."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "clickhouse_connect":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="clickhouse-connect is required"):
            ClickHouseLoader(host="localhost")


# ---------------------------------------------------------------------------
# ClickHouseLoader — get_tick_data
# ---------------------------------------------------------------------------

class TestGetTickData:
    def test_returns_polars_dataframe(self):
        tick_pd = _tick_pandas(n=5)
        loader = ClickHouseLoader(client=_make_mock_client(tick_pd))
        result = loader.get_tick_data(
            symbols=["AAPL"],
            start="2024-01-02",
            end="2024-01-03",
        )
        assert isinstance(result, pl.DataFrame)

    def test_schema_matches_tick_schema(self):
        tick_pd = _tick_pandas(n=5)
        loader = ClickHouseLoader(client=_make_mock_client(tick_pd))
        result = loader.get_tick_data(["AAPL"], "2024-01-02", "2024-01-03")
        for col, dtype in TICK_SCHEMA.items():
            assert col in result.columns, f"Missing column: {col}"
            assert result[col].dtype == dtype, (
                f"Column {col}: expected {dtype}, got {result[col].dtype}"
            )

    def test_empty_result_on_empty_db_response(self):
        loader = ClickHouseLoader(client=_make_mock_client(pd.DataFrame()))
        result = loader.get_tick_data(["AAPL"], "2024-01-02", "2024-01-03")
        assert result.is_empty()
        assert set(result.columns) == set(TICK_SCHEMA.keys())

    def test_pit_cutoff_is_injected_into_query(self):
        """Verify that the SQL query uses the pit_cutoff as the effective end bound."""
        tick_pd = _tick_pandas(n=3)
        mock_client = _make_mock_client(tick_pd)
        loader = ClickHouseLoader(client=mock_client)
        loader.get_tick_data(
            symbols=["AAPL"],
            start="2024-01-02",
            end="2024-12-31",
            pit_cutoff="2024-06-01",
        )
        call_kwargs = mock_client.query_df.call_args
        params = call_kwargs[1]["parameters"] if call_kwargs[1] else call_kwargs[0][1]
        # The effective end should be min(end, pit_cutoff) = 2024-06-01
        assert "2024-06-01" in params["end"]

    def test_pit_cutoff_earlier_than_end_wins(self):
        """pit_cutoff earlier than end must constrain the query."""
        tick_pd = _tick_pandas(n=3)
        mock_client = _make_mock_client(tick_pd)
        loader = ClickHouseLoader(client=mock_client)
        loader.get_tick_data(
            symbols=["AAPL"],
            start="2024-01-02",
            end="2024-12-31",
            pit_cutoff="2024-03-15",
        )
        params = mock_client.query_df.call_args[1]["parameters"]
        assert params["end"].startswith("2024-03-15")

    def test_no_pit_cutoff_uses_end(self):
        """When pit_cutoff is None, end date is used as the cutoff."""
        tick_pd = _tick_pandas(n=3)
        mock_client = _make_mock_client(tick_pd)
        loader = ClickHouseLoader(client=mock_client)
        loader.get_tick_data(
            symbols=["AAPL"],
            start="2024-01-02",
            end="2024-06-30",
        )
        params = mock_client.query_df.call_args[1]["parameters"]
        assert params["end"].startswith("2024-06-30")


# ---------------------------------------------------------------------------
# ClickHouseLoader — get_bar_data
# ---------------------------------------------------------------------------

class TestGetBarData:
    def test_returns_polars_dataframe(self):
        bar_pd = _bar_pandas(n=5)
        loader = ClickHouseLoader(client=_make_mock_client(bar_pd))
        result = loader.get_bar_data(["AAPL"], "2024-01-02", "2024-01-10", freq="1d")
        assert isinstance(result, pl.DataFrame)

    def test_schema_matches_bar_schema(self):
        bar_pd = _bar_pandas(n=5)
        loader = ClickHouseLoader(client=_make_mock_client(bar_pd))
        result = loader.get_bar_data(["AAPL"], "2024-01-02", "2024-01-10", freq="1d")
        for col, dtype in BAR_SCHEMA.items():
            assert col in result.columns, f"Missing column: {col}"
            assert result[col].dtype == dtype, (
                f"Column {col}: expected {dtype}, got {result[col].dtype}"
            )

    def test_from_ticks_delegates_to_resample(self):
        """When from_ticks=True, result comes from resample_ticks_to_bars."""
        tick_pd = _tick_pandas(n=30, start_ts="2024-01-02 09:30:00")
        loader = ClickHouseLoader(client=_make_mock_client(tick_pd))
        result = loader.get_bar_data(
            ["AAPL"], "2024-01-02", "2024-01-03", freq="5min", from_ticks=True
        )
        assert isinstance(result, pl.DataFrame)
        assert set(result.columns) == set(BAR_SCHEMA.keys())

    def test_freq_is_passed_to_query(self):
        bar_pd = _bar_pandas(n=5)
        mock_client = _make_mock_client(bar_pd)
        loader = ClickHouseLoader(client=mock_client)
        loader.get_bar_data(["AAPL"], "2024-01-02", "2024-01-10", freq="1d")
        params = mock_client.query_df.call_args[1]["parameters"]
        assert params["freq"] == "1d"


# ---------------------------------------------------------------------------
# ClickHouseLoader — resample_ticks_to_bars (pure Polars, no mock needed)
# ---------------------------------------------------------------------------

class TestResampleTicksToBars:
    def _make_tick_df(
        self,
        n: int = 60,
        symbol: str = "AAPL",
        base_price: float = 100.0,
        start: str = "2024-01-02 09:30:00",
    ) -> pl.DataFrame:
        import numpy as np
        ts = [
            datetime(2024, 1, 2, 9, 30, 0, tzinfo=timezone.utc)
        ]
        for i in range(1, n):
            prev = ts[-1]
            ts.append(prev.replace(second=prev.second + 1) if prev.second < 59
                       else prev.replace(minute=prev.minute + 1, second=0))
        return pl.DataFrame({
            "ts": pl.Series(ts).cast(pl.Datetime("us", "UTC")),
            "symbol": symbol,
            "price": [base_price + i * 0.01 for i in range(n)],
            "size": [100.0] * n,
        })

    def test_returns_bar_schema(self):
        tick_df = self._make_tick_df(n=60)
        bars = ClickHouseLoader.resample_ticks_to_bars(tick_df, "1m")
        for col in BAR_SCHEMA:
            assert col in bars.columns, f"Missing: {col}"

    def test_empty_ticks_returns_empty_bars(self):
        empty = pl.DataFrame(schema={
            "ts": pl.Datetime("us", "UTC"),
            "symbol": pl.Utf8,
            "price": pl.Float64,
            "size": pl.Float64,
        })
        result = ClickHouseLoader.resample_ticks_to_bars(empty, "1m")
        assert result.is_empty()

    def test_open_is_first_price(self):
        """The 'open' of each bar must equal the first tick price in that period."""
        tick_df = pl.DataFrame({
            "ts": pl.Series([
                datetime(2024, 1, 2, 9, 30, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 2, 9, 30, 30, tzinfo=timezone.utc),
                datetime(2024, 1, 2, 9, 31, 0, tzinfo=timezone.utc),
            ]).cast(pl.Datetime("us", "UTC")),
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "price": [100.0, 101.0, 102.0],
            "size": [100.0, 200.0, 150.0],
        })
        bars = ClickHouseLoader.resample_ticks_to_bars(tick_df, "1m")
        bar_930 = bars.filter(pl.col("symbol") == "AAPL").sort("ts").row(0)
        open_price = bar_930[bars.columns.index("open")]
        assert open_price == 100.0

    def test_vwap_formula(self):
        """VWAP = sum(price * size) / sum(size) within each bar."""
        tick_df = pl.DataFrame({
            "ts": pl.Series([
                datetime(2024, 1, 2, 9, 30, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 2, 9, 30, 30, tzinfo=timezone.utc),
            ]).cast(pl.Datetime("us", "UTC")),
            "symbol": ["AAPL", "AAPL"],
            "price": [100.0, 200.0],
            "size": [100.0, 300.0],
        })
        bars = ClickHouseLoader.resample_ticks_to_bars(tick_df, "1m")
        expected_vwap = (100.0 * 100.0 + 200.0 * 300.0) / (100.0 + 300.0)
        actual_vwap = bars["vwap"][0]
        assert abs(actual_vwap - expected_vwap) < 1e-9

    def test_missing_columns_raises(self):
        bad_df = pl.DataFrame({"ts": [], "symbol": []})
        with pytest.raises(ValueError, match="missing required columns"):
            ClickHouseLoader.resample_ticks_to_bars(bad_df, "1d")

    def test_multiple_symbols_are_grouped_separately(self):
        tick_df = pl.DataFrame({
            "ts": pl.Series([
                datetime(2024, 1, 2, 9, 30, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 2, 9, 30, 0, tzinfo=timezone.utc),
            ]).cast(pl.Datetime("us", "UTC")),
            "symbol": ["AAPL", "MSFT"],
            "price": [150.0, 300.0],
            "size": [100.0, 200.0],
        })
        bars = ClickHouseLoader.resample_ticks_to_bars(tick_df, "1d")
        assert set(bars["symbol"].to_list()) == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# UniverseBuilder — basic construction
# ---------------------------------------------------------------------------

def _make_membership(rows: list[tuple]) -> pl.DataFrame:
    """rows: [(symbol, index, entry_date, exit_date), ...]"""
    return pl.DataFrame({
        "symbol": [r[0] for r in rows],
        "index": [r[1] for r in rows],
        "entry_date": pl.Series([r[2] for r in rows]).str.to_date(),
        "exit_date": pl.Series([r[3] for r in rows]).str.to_date(),
    })


def _make_price(rows: list[tuple]) -> pl.DataFrame:
    """rows: [(date, symbol, close), ...]"""
    return pl.DataFrame({
        "date": pl.Series([r[0] for r in rows]).str.to_date() if rows else pl.Series([], dtype=pl.Date),
        "symbol": [r[1] for r in rows],
        "close": pl.Series([r[2] for r in rows], dtype=pl.Float64),
    })


def _make_adv(rows: list[tuple]) -> pl.DataFrame:
    """rows: [(date, symbol, adv_20d), ...]"""
    return pl.DataFrame({
        "date": pl.Series([r[0] for r in rows]).str.to_date() if rows else pl.Series([], dtype=pl.Date),
        "symbol": [r[1] for r in rows],
        "adv_20d": pl.Series([r[2] for r in rows], dtype=pl.Float64),
    })


def _make_mcap(rows: list[tuple]) -> pl.DataFrame:
    """rows: [(date, symbol, market_cap), ...]"""
    return pl.DataFrame({
        "date": pl.Series([r[0] for r in rows]).str.to_date() if rows else pl.Series([], dtype=pl.Date),
        "symbol": [r[1] for r in rows],
        "market_cap": pl.Series([r[2] for r in rows], dtype=pl.Float64),
    })


class TestUniverseBuilderInit:
    def test_missing_membership_col_raises(self):
        bad = pl.DataFrame({"symbol": [], "index": []})  # missing entry/exit date
        with pytest.raises(ValueError, match="membership_df"):
            UniverseBuilder(
                membership_df=bad,
                price_df=_make_price([]),
                adv_df=_make_adv([]),
                mcap_df=_make_mcap([]),
            )


class TestUniverseBuilderBuild:
    def _default_builder(self) -> UniverseBuilder:
        membership = _make_membership([
            ("AAPL", "SP500", "2020-01-01", "9999-12-31"),
            ("MSFT", "SP500", "2020-01-01", "9999-12-31"),
            ("DELISTED", "SP500", "2020-01-01", "2023-12-31"),  # delisted before as_of
            ("NOTYET", "SP500", "2025-01-01", "9999-12-31"),    # added after as_of
        ])
        price = _make_price([
            ("2024-01-02", "AAPL", 180.0),
            ("2024-01-02", "MSFT", 380.0),
            ("2024-01-02", "DELISTED", 5.0),
            ("2024-01-02", "NOTYET", 50.0),
        ])
        adv = _make_adv([
            ("2024-01-02", "AAPL", 5_000_000.0),
            ("2024-01-02", "MSFT", 8_000_000.0),
            ("2024-01-02", "DELISTED", 200_000.0),
            ("2024-01-02", "NOTYET", 1_500_000.0),
        ])
        mcap = _make_mcap([
            ("2024-01-02", "AAPL", 3e12),
            ("2024-01-02", "MSFT", 2.5e12),
            ("2024-01-02", "DELISTED", 1e8),
            ("2024-01-02", "NOTYET", 5e9),
        ])
        return UniverseBuilder(membership, price, adv, mcap)

    def test_basic_build_returns_dataframe(self):
        builder = self._default_builder()
        result = builder.build("2024-01-02", index="SP500")
        assert isinstance(result, pl.DataFrame)

    def test_output_schema(self):
        builder = self._default_builder()
        result = builder.build("2024-01-02", index="SP500")
        for col in UNIVERSE_SCHEMA:
            assert col in result.columns

    def test_pit_excludes_delisted_symbol(self):
        """Symbol with exit_date <= as_of_date must not appear in the universe."""
        builder = self._default_builder()
        result = builder.build("2024-01-02", index="SP500")
        assert "DELISTED" not in result["symbol"].to_list()

    def test_pit_excludes_future_addition(self):
        """Symbol with entry_date > as_of_date must not appear in the universe."""
        builder = self._default_builder()
        result = builder.build("2024-01-02", index="SP500")
        assert "NOTYET" not in result["symbol"].to_list()

    def test_active_symbols_are_present(self):
        """Symbols that are active on as_of_date and pass all filters appear."""
        builder = self._default_builder()
        result = builder.build("2024-01-02", index="SP500",
                               adv_floor=1_000_000, price_floor=5.0, mcap_floor=1e9)
        symbols = result["symbol"].to_list()
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    def test_adv_floor_excludes_illiquid(self):
        """Symbols below the ADV floor are excluded."""
        membership = _make_membership([
            ("AAPL", "SP500", "2020-01-01", "9999-12-31"),
            ("ILLIQ", "SP500", "2020-01-01", "9999-12-31"),
        ])
        price = _make_price([("2024-01-02", "AAPL", 180.0), ("2024-01-02", "ILLIQ", 50.0)])
        adv = _make_adv([("2024-01-02", "AAPL", 5e6), ("2024-01-02", "ILLIQ", 499_999.0)])
        mcap = _make_mcap([("2024-01-02", "AAPL", 3e12), ("2024-01-02", "ILLIQ", 1e10)])
        builder = UniverseBuilder(membership, price, adv, mcap)
        result = builder.build("2024-01-02", adv_floor=500_000.0)
        assert "AAPL" in result["symbol"].to_list()
        assert "ILLIQ" not in result["symbol"].to_list()

    def test_adv_floor_boundary_exact_value_is_included(self):
        """A symbol exactly at the ADV floor must be included (>=, not >)."""
        membership = _make_membership([("AAPL", "SP500", "2020-01-01", "9999-12-31")])
        price = _make_price([("2024-01-02", "AAPL", 180.0)])
        adv = _make_adv([("2024-01-02", "AAPL", 1_000_000.0)])
        mcap = _make_mcap([("2024-01-02", "AAPL", 3e12)])
        builder = UniverseBuilder(membership, price, adv, mcap)
        result = builder.build("2024-01-02", adv_floor=1_000_000.0)
        assert "AAPL" in result["symbol"].to_list()

    def test_price_floor_boundary_below_excludes(self):
        """A symbol with close just below the price floor must be excluded."""
        membership = _make_membership([("PENNY", "SP500", "2020-01-01", "9999-12-31")])
        price = _make_price([("2024-01-02", "PENNY", 4.99)])
        adv = _make_adv([("2024-01-02", "PENNY", 2e6)])
        mcap = _make_mcap([("2024-01-02", "PENNY", 5e8)])
        builder = UniverseBuilder(membership, price, adv, mcap)
        result = builder.build("2024-01-02", price_floor=5.0)
        assert result.is_empty()

    def test_mcap_floor_excludes_small_cap(self):
        """Symbols below the market-cap floor are excluded."""
        membership = _make_membership([("SMALL", "SP500", "2020-01-01", "9999-12-31")])
        price = _make_price([("2024-01-02", "SMALL", 10.0)])
        adv = _make_adv([("2024-01-02", "SMALL", 2e6)])
        mcap = _make_mcap([("2024-01-02", "SMALL", 2.99e8)])
        builder = UniverseBuilder(membership, price, adv, mcap)
        result = builder.build("2024-01-02", mcap_floor=3e8)
        assert result.is_empty()

    def test_index_filter(self):
        """Index filter restricts result to symbols in that index."""
        membership = _make_membership([
            ("AAPL", "SP500", "2020-01-01", "9999-12-31"),
            ("EXTRA", "R1000", "2020-01-01", "9999-12-31"),
        ])
        price = _make_price([("2024-01-02", "AAPL", 180.0), ("2024-01-02", "EXTRA", 50.0)])
        adv = _make_adv([("2024-01-02", "AAPL", 5e6), ("2024-01-02", "EXTRA", 5e6)])
        mcap = _make_mcap([("2024-01-02", "AAPL", 3e12), ("2024-01-02", "EXTRA", 1e10)])
        builder = UniverseBuilder(membership, price, adv, mcap)
        result = builder.build("2024-01-02", index="SP500")
        assert all(r == "SP500" for r in result["index"].to_list())
        assert "EXTRA" not in result["symbol"].to_list()

    def test_no_data_returns_empty_dataframe(self):
        """build() on a date with no members returns an empty DataFrame."""
        membership = _make_membership([
            ("AAPL", "SP500", "2025-01-01", "9999-12-31"),  # future entry
        ])
        builder = UniverseBuilder(
            membership,
            _make_price([("2024-01-02", "AAPL", 180.0)]),
            _make_adv([("2024-01-02", "AAPL", 5e6)]),
            _make_mcap([("2024-01-02", "AAPL", 3e12)]),
        )
        result = builder.build("2024-01-02")
        assert result.is_empty()
        assert set(result.columns) == set(UNIVERSE_SCHEMA.keys())

    def test_pit_slice_uses_last_available_date(self):
        """_pit_slice returns the most recent observation on or before as_of_date."""
        membership = _make_membership([("AAPL", "SP500", "2020-01-01", "9999-12-31")])
        # Price data: only available on Jan 2, not Jan 3
        price = _make_price([("2024-01-02", "AAPL", 200.0)])
        adv = _make_adv([("2024-01-02", "AAPL", 5e6)])
        mcap = _make_mcap([("2024-01-02", "AAPL", 3e12)])
        builder = UniverseBuilder(membership, price, adv, mcap)
        # as_of Jan 3: last available price is Jan 2 → should still work
        result = builder.build("2024-01-03")
        assert "AAPL" in result["symbol"].to_list()
        assert result.filter(pl.col("symbol") == "AAPL")["close"][0] == 200.0


class TestUniverseBuilderPanel:
    def test_build_panel_returns_long_format(self):
        membership = _make_membership([("AAPL", "SP500", "2020-01-01", "9999-12-31")])
        price = _make_price([
            ("2024-01-02", "AAPL", 180.0),
            ("2024-01-03", "AAPL", 182.0),
            ("2024-01-04", "AAPL", 184.0),
        ])
        adv = _make_adv([("2024-01-02", "AAPL", 5e6)])
        mcap = _make_mcap([("2024-01-02", "AAPL", 3e12)])
        builder = UniverseBuilder(membership, price, adv, mcap)
        panel = builder.build_panel(["2024-01-02", "2024-01-03", "2024-01-04"])
        assert len(panel) == 3
        assert set(panel["date"].to_list()) == {
            date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)
        }


# ---------------------------------------------------------------------------
# LiquidityFilter
# ---------------------------------------------------------------------------

class TestLiquidityFilter:
    def test_default_values(self):
        f = LiquidityFilter()
        assert f.adv_floor == 1_000_000.0
        assert f.price_floor == 5.0
        assert f.mcap_floor == 300_000_000.0

    def test_as_kwargs(self):
        f = LiquidityFilter(adv_floor=2e6, price_floor=10.0, mcap_floor=1e9)
        kwargs = f.as_kwargs()
        assert kwargs == {"adv_floor": 2e6, "price_floor": 10.0, "mcap_floor": 1e9}

    def test_repr(self):
        f = LiquidityFilter()
        assert "LiquidityFilter" in repr(f)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

class TestUtilityHelpers:
    def test_parse_dt_string(self):
        dt = _parse_dt("2024-01-02T09:30:00")
        assert dt.year == 2024 and dt.month == 1 and dt.day == 2

    def test_parse_dt_aware(self):
        dt = _parse_dt("2024-06-01T00:00:00+00:00")
        assert dt.tzinfo is not None

    def test_parse_dt_passthrough(self):
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert _parse_dt(now) is now

    def test_normalise_freq(self):
        assert _normalise_freq("1T") == "1m"
        assert _normalise_freq("5min") == "5m"
        assert _normalise_freq("1D") == "1d"

    def test_to_polars_duration(self):
        assert _to_polars_duration("1d") == "1d"
        assert _to_polars_duration("5T") == "5m"

    def test_parse_date_string(self):
        d = _parse_date("2024-03-15")
        assert d == date(2024, 3, 15)

    def test_parse_date_passthrough(self):
        d = date(2024, 3, 15)
        assert _parse_date(d) is d
