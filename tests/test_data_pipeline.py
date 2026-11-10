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
)
from semn.data.universe import (
    ADV_WINDOW,
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
        "session_date": rng.tz_convert("America/New_York").date,
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

    def test_symbols_are_bound_not_interpolated(self):
        """Symbols travel as an Array(String) parameter; nothing user-supplied is spliced into SQL."""
        mock_client = _make_mock_client(_tick_pandas(n=3))
        loader = ClickHouseLoader(client=mock_client)
        loader.get_tick_data(["AAPL", "O'NEIL"], "2024-01-02", "2024-01-03")
        sql = mock_client.query_df.call_args[0][0]
        params = mock_client.query_df.call_args[1]["parameters"]
        assert params["symbols"] == ["AAPL", "O'NEIL"]
        assert "{symbols:Array(String)}" in sql
        assert "NEIL" not in sql


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

    def test_daily_bars_follow_new_york_session_date(self):
        """17:00 and 21:00 New York on Jan 2 are one daily bar dated Jan 2, though UTC dates differ."""
        tick_df = pl.DataFrame({
            "ts": pl.Series([
                datetime(2024, 1, 2, 22, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 3, 2, 0, tzinfo=timezone.utc),
            ]).cast(pl.Datetime("us", "UTC")),
            "symbol": ["AAPL", "AAPL"],
            "price": [100.0, 101.0],
            "size": [10.0, 10.0],
        })
        bars = ClickHouseLoader.resample_ticks_to_bars(tick_df, "1d")
        assert len(bars) == 1
        assert bars["session_date"][0] == date(2024, 1, 2)
        assert bars["ts"][0] == datetime(2024, 1, 2, 5, 0, tzinfo=timezone.utc)
        assert bars["close"][0] == 101.0


# ---------------------------------------------------------------------------
# UniverseBuilder — fixtures
# ---------------------------------------------------------------------------

AS_OF = date(2024, 1, 2)


def _sessions(n: int = ADV_WINDOW, end: date = AS_OF) -> list[date]:
    """*n* business days ending on *end* (stand-in for exchange sessions)."""
    return [d.date() for d in pd.bdate_range(end=end, periods=n)]


def _history(sid: str, close, volume=1e6, sessions: Optional[list[date]] = None) -> pl.DataFrame:
    """Raw daily rows for one security; *close*/*volume* may be scalars or per-session lists."""
    sessions = sessions or _sessions()
    n = len(sessions)
    return pl.DataFrame({
        "date": sessions,
        "security_id": [sid] * n,
        "close": [float(c) for c in (close if isinstance(close, list) else [close] * n)],
        "volume": [float(v) for v in (volume if isinstance(volume, list) else [volume] * n)],
    })


def _make_membership(rows: list[tuple]) -> pl.DataFrame:
    """rows: [(security_id, index, entry_date, exit_date), ...]"""
    return pl.DataFrame({
        "security_id": [r[0] for r in rows],
        "index": [r[1] for r in rows],
        "entry_date": pl.Series([r[2] for r in rows]).str.to_date(),
        "exit_date": pl.Series([r[3] for r in rows]).str.to_date(),
    })


def _make_mcap(rows: list[tuple]) -> pl.DataFrame:
    """rows: [(date, security_id, market_cap), ...]"""
    return pl.DataFrame({
        "date": pl.Series([r[0] for r in rows], dtype=pl.Utf8).str.to_date(),
        "security_id": pl.Series([r[1] for r in rows], dtype=pl.Utf8),
        "market_cap": pl.Series([r[2] for r in rows], dtype=pl.Float64),
    })


#: Business days standing in for NYSE sessions in these synthetic fixtures.
ALL_SESSIONS = pl.Series("date", _sessions(400, end=date(2024, 6, 28)), dtype=pl.Date)


def _builder(members: list[tuple], prices: list[pl.DataFrame], mcap: list[tuple] = ()) -> UniverseBuilder:
    return UniverseBuilder(_make_membership(members), pl.concat(prices), _make_mcap(list(mcap)), ALL_SESSIONS)


ACTIVE = ("2020-01-01", "9999-12-31")


class TestUniverseBuilderInit:
    def test_missing_membership_col_raises(self):
        bad = pl.DataFrame({"security_id": [], "index": []})  # missing entry/exit date
        with pytest.raises(ValueError, match="membership_df"):
            UniverseBuilder(bad, _history("A", 10.0), _make_mcap([]), ALL_SESSIONS)

    def test_price_without_volume_raises(self):
        """ADV is computed in-house, so raw volume is required."""
        with pytest.raises(ValueError, match="price_df"):
            UniverseBuilder(
                _make_membership([("A", "SP500", *ACTIVE)]),
                _history("A", 10.0).drop("volume"),
                _make_mcap([]),
                ALL_SESSIONS,
            )

    def test_duplicate_price_rows_raise(self):
        """Duplicates would let 10 real days pass the 20-observation ADV history check."""
        with pytest.raises(ValueError, match="duplicate"):
            _builder([("A", "SP500", *ACTIVE)], [_history("A", 10.0), _history("A", 10.0)])


# ---------------------------------------------------------------------------
# UniverseBuilder — single date
# ---------------------------------------------------------------------------

class TestUniverseBuilderBuild:
    def _default_builder(self) -> UniverseBuilder:
        return _builder(
            [
                ("AAPL", "SP500", *ACTIVE),
                ("MSFT", "SP500", *ACTIVE),
                ("DELISTED", "SP500", "2020-01-01", "2023-12-31"),
                ("NOTYET", "SP500", "2025-01-01", "9999-12-31"),
            ],
            [_history(s, 100.0) for s in ("AAPL", "MSFT", "DELISTED", "NOTYET")],
            [("2024-01-02", "AAPL", 3e12), ("2024-01-02", "MSFT", 2.5e12)],
        )

    def test_output_schema(self):
        result = self._default_builder().build(AS_OF, index="SP500")
        assert result.columns == list(UNIVERSE_SCHEMA)

    def test_pit_excludes_delisted_and_future_members(self):
        ids = self._default_builder().build(AS_OF, index="SP500")["security_id"].to_list()
        assert ids == ["AAPL", "MSFT"]

    def test_adv_is_raw_close_times_raw_volume(self):
        result = self._default_builder().build(AS_OF)
        assert result["adv_20d"].to_list() == [100.0 * 1e6, 100.0 * 1e6]

    def test_adv_floor_boundary(self):
        """Exactly at the floor is included (>=); just below is excluded."""
        b = _builder(
            [("AT", "SP500", *ACTIVE), ("BELOW", "SP500", *ACTIVE)],
            [_history("AT", 25.0, 1e6), _history("BELOW", 25.0, 999_999.0)],
        )
        assert b.build(AS_OF, adv_floor=25e6)["security_id"].to_list() == ["AT"]

    def test_price_floor_boundary_below_excludes(self):
        b = _builder([("PENNY", "SP500", *ACTIVE)], [_history("PENNY", 4.99, 1e8)])
        assert b.build(AS_OF, price_floor=5.0).is_empty()

    def test_mcap_floor_excludes_small_cap(self):
        b = _builder(
            [("SMALL", "SP500", *ACTIVE)], [_history("SMALL", 10.0)],
            [("2024-01-02", "SMALL", 2.99e8)],
        )
        assert b.build(AS_OF, mcap_floor=3e8).is_empty()

    def test_missing_mcap_is_fine_without_mcap_floor(self):
        b = _builder([("NOCAP", "SP500", *ACTIVE)], [_history("NOCAP", 10.0)])
        assert b.build(AS_OF)["security_id"].to_list() == ["NOCAP"]
        assert b.build(AS_OF, mcap_floor=1.0).is_empty()

    def test_nan_close_is_missing_not_eligible(self):
        """NaN compares above every number in Polars; it must not pass the price floor."""
        close = [50.0] * (ADV_WINDOW - 1) + [float("nan")]
        b = _builder([("X", "SP500", *ACTIVE)], [_history("X", close, 1e7)])
        assert b.build(AS_OF, **LiquidityFilter().as_kwargs()).is_empty()
        assert b.exclusions([AS_OF])["exclusion"].to_list() == ["no_data"]

    def test_index_none_returns_each_security_once(self):
        b = _builder(
            [("X", "SP500", *ACTIVE), ("X", "R1000", *ACTIVE)],
            [_history("X", 50.0)],
        )
        result = b.build(AS_OF)
        assert result.select("security_id", "index").rows() == [("X", "R1000")]

    def test_stale_market_cap_is_treated_as_missing(self):
        b = _builder(
            [("X", "SP500", *ACTIVE)], [_history("X", 50.0)],
            [("2023-12-22", "X", 5e10)],  # 7 sessions before AS_OF
        )
        assert b.build(AS_OF)["market_cap"].to_list() == [None]
        assert b.build(AS_OF, mcap_floor=1e9).is_empty()
        fresh = _builder(
            [("X", "SP500", *ACTIVE)], [_history("X", 50.0)],
            [("2023-12-26", "X", 5e10)],  # exactly 5 sessions before AS_OF
        )
        assert fresh.build(AS_OF)["market_cap"].to_list() == [5e10]

    def test_rebalance_after_last_session_raises(self):
        b = _builder([("X", "SP500", *ACTIVE)], [_history("X", 50.0)])
        with pytest.raises(ValueError, match="last session"):
            b.build(date(2030, 1, 2))

    def test_index_filter(self):
        b = _builder(
            [("AAPL", "SP500", *ACTIVE), ("EXTRA", "R1000", *ACTIVE)],
            [_history("AAPL", 100.0), _history("EXTRA", 100.0)],
        )
        assert b.build(AS_OF, index="SP500")["security_id"].to_list() == ["AAPL"]

    def test_no_members_returns_empty_dataframe(self):
        b = _builder([("AAPL", "SP500", "2025-01-01", "9999-12-31")], [_history("AAPL", 100.0)])
        result = b.build(AS_OF)
        assert result.is_empty()
        assert result.columns == list(UNIVERSE_SCHEMA)

    def test_uses_last_observation_on_or_before_date(self):
        b = _builder([("AAPL", "SP500", *ACTIVE)], [_history("AAPL", 200.0)])
        result = b.build(date(2024, 1, 3))  # no row on Jan 3; Jan 2 is the last one
        assert result["close"].to_list() == [200.0]

    def test_future_observations_never_leak(self):
        sessions = _sessions(ADV_WINDOW + 1, end=date(2024, 1, 3))
        b = _builder([("AAPL", "SP500", *ACTIVE)], [_history("AAPL", [100.0] * ADV_WINDOW + [999.0], sessions=sessions)])
        assert b.build(AS_OF)["close"].to_list() == [100.0]


# ---------------------------------------------------------------------------
# Raw prices, splits, ADV history, staleness
# ---------------------------------------------------------------------------

class TestRawPricesAndHistory:
    def _split_builder(self) -> tuple[UniverseBuilder, list[date]]:
        """10:1 split after session 25: raw close 40 → 4, raw volume ×10."""
        sessions = _sessions(30)
        close = [40.0] * 25 + [4.0] * 5
        volume = [1e6] * 25 + [1e7] * 5
        return _builder([("SPLIT", "SP500", *ACTIVE)], [_history("SPLIT", close, volume, sessions)]), sessions

    def test_price_floor_uses_raw_price_before_split(self):
        """Split-adjusted, the pre-split close would be $4 and fail the $5 floor; raw is $40."""
        b, sessions = self._split_builder()
        assert b.build(sessions[24], price_floor=5.0)["security_id"].to_list() == ["SPLIT"]
        assert b.build(sessions[25], price_floor=5.0).is_empty()

    def test_adv_does_not_jump_on_split_date(self):
        b, sessions = self._split_builder()
        adv = b.build_panel(sessions[19:])["adv_20d"].to_list()
        assert adv == [40e6] * len(adv)

    def test_insufficient_adv_history_is_excluded(self):
        b = _builder([("NEW", "SP500", *ACTIVE)], [_history("NEW", 50.0, sessions=_sessions(ADV_WINDOW - 1))])
        assert b.build(AS_OF).is_empty()
        assert b.exclusions([AS_OF])["exclusion"].to_list() == ["adv_history"]

    def test_stale_observations_are_excluded_and_reported(self):
        """HALT stops trading; it survives 5 sessions of staleness and is dropped on the 6th."""
        sessions = _sessions(30)
        b = _builder(
            [("LIVE", "SP500", *ACTIVE), ("HALT", "SP500", *ACTIVE)],
            [_history("LIVE", 50.0, sessions=sessions), _history("HALT", 50.0, sessions=sessions[:24])],
        )
        assert "HALT" in b.build(sessions[23 + 5], max_staleness=5)["security_id"].to_list()
        assert "HALT" not in b.build(sessions[23 + 6], max_staleness=5)["security_id"].to_list()
        report = b.exclusions([sessions[29]], max_staleness=5)
        assert report.select("security_id", "exclusion").rows() == [("HALT", "stale")]

    def test_exclusion_reasons(self):
        b = _builder(
            [("CHEAP", "SP500", *ACTIVE), ("THIN", "SP500", *ACTIVE), ("NODATA", "SP500", *ACTIVE)],
            [_history("CHEAP", 1.0, 1e9), _history("THIN", 50.0, 1.0)],
        )
        report = b.exclusions([AS_OF], **LiquidityFilter().as_kwargs())
        assert dict(report.select("security_id", "exclusion").rows()) == {
            "CHEAP": "price_floor", "THIN": "adv_floor", "NODATA": "no_data",
        }


# ---------------------------------------------------------------------------
# UniverseBuilder — panel
# ---------------------------------------------------------------------------

class TestUniverseBuilderPanel:
    def _builder(self) -> tuple[UniverseBuilder, list[date]]:
        sessions = _sessions(60)
        members = [
            ("A", "SP500", "2020-01-01", "9999-12-31"),
            ("B", "SP500", str(sessions[30]), "9999-12-31"),   # joins mid-panel
            ("C", "SP500", "2020-01-01", str(sessions[40])),   # exits mid-panel
            ("D", "SP500", str(sessions[25]), str(sessions[35])),
        ]
        prices = [_history(s, 10.0 + i, sessions=sessions) for i, s in enumerate("ABCD")]
        return _builder(members, prices), sessions

    def test_membership_holds_on_every_rebalance_date(self):
        """No security before its entry date or on/after its exit date, on any date in the panel."""
        b, sessions = self._builder()
        membership = b._membership
        panel = b.build_panel(sessions).join(membership, on="security_id")
        assert not panel.is_empty()
        assert panel.filter(
            (pl.col("date") < pl.col("entry_date")) | (pl.col("date") >= pl.col("exit_date"))
        ).is_empty()
        assert set(panel.filter(pl.col("date") == sessions[40])["security_id"]) == {"A", "B"}

    def test_panel_equals_per_date_builds(self):
        b, sessions = self._builder()
        per_date = pl.concat([b.build(d) for d in sessions])
        plt.assert_frame_equal(b.build_panel(sessions), per_date)

    def test_panel_is_deterministic(self):
        b, sessions = self._builder()
        plt.assert_frame_equal(b.build_panel(sessions[::-1]), b.build_panel(sessions))


# ---------------------------------------------------------------------------
# LiquidityFilter
# ---------------------------------------------------------------------------

class TestLiquidityFilter:
    def test_default_values_match_roadmap(self):
        assert LiquidityFilter().as_kwargs() == {
            "adv_floor": 25e6, "price_floor": 5.0, "mcap_floor": 0.0, "max_staleness": 5,
        }

    def test_as_kwargs_feeds_build(self):
        b = _builder([("A", "SP500", *ACTIVE)], [_history("A", 30.0, 1e6)])
        assert b.build(AS_OF, **LiquidityFilter().as_kwargs())["security_id"].to_list() == ["A"]
        assert b.build(AS_OF, **LiquidityFilter(adv_floor=31e6).as_kwargs()).is_empty()


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
        assert _parse_dt(now) == now

    def test_parse_dt_naive_is_new_york(self):
        """A date-only bound is the start of the New York session date (05:00 UTC in winter)."""
        assert _parse_dt("2024-01-02") == datetime(2024, 1, 2, 5, 0, tzinfo=timezone.utc)

    def test_parse_dt_other_zone_converted_to_utc(self):
        """16:00 Tokyo is 07:00 UTC; sending the wall clock "16:00" would leak 9 hours of future data."""
        from zoneinfo import ZoneInfo
        tokyo = datetime(2024, 1, 2, 16, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
        assert _parse_dt(tokyo) == datetime(2024, 1, 2, 7, 0, tzinfo=timezone.utc)

    def test_pit_cutoff_sent_as_utc(self):
        from zoneinfo import ZoneInfo
        mock_client = _make_mock_client(_tick_pandas(n=3))
        ClickHouseLoader(client=mock_client).get_tick_data(
            ["AAPL"], "2024-01-02", "2024-01-03",
            pit_cutoff=datetime(2024, 1, 2, 16, 0, tzinfo=ZoneInfo("Asia/Tokyo")),
        )
        sql = mock_client.query_df.call_args[0][0]
        assert mock_client.query_df.call_args[1]["parameters"]["end"] == "2024-01-02 07:00:00"
        assert "DateTime64(6, 'UTC')" in sql

    def test_normalise_freq(self):
        assert _normalise_freq("1T") == "1m"
        assert _normalise_freq("5min") == "5m"
        assert _normalise_freq("1D") == "1d"


    def test_parse_date_string(self):
        d = _parse_date("2024-03-15")
        assert d == date(2024, 3, 15)

    def test_parse_date_passthrough(self):
        d = date(2024, 3, 15)
        assert _parse_date(d) is d
