"""
tests/test_data_layer.py
============================
Phase 1 data layer beyond the loader/universe: calendar, security master,
corporate actions, Parquet store, snapshots and the data-quality report.
Each roadmap verification invariant has a test named after it.
"""

from __future__ import annotations

import hashlib
from datetime import date

import polars as pl
import polars.testing as plt
import pytest

from semn.data import (
    LiquidityFilter,
    ParquetStore,
    SecurityMaster,
    UniverseBuilder,
    calendar,
    content_hash,
    load_snapshot,
    quality_report,
    total_returns,
    write_snapshot,
)

FAR = date(9999, 12, 31)


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------

class TestCalendar:
    def test_sessions_skip_weekends_and_holidays(self):
        s = calendar.sessions("2024-01-01", "2024-01-16").to_list()
        assert date(2024, 1, 1) not in s   # New Year's Day
        assert date(2024, 1, 15) not in s  # MLK Day
        assert date(2024, 1, 6) not in s   # Saturday
        assert s[0] == date(2024, 1, 2) and s[-1] == date(2024, 1, 16)

    def test_sessions_reach_far_back(self):
        assert calendar.sessions("1995-01-03", "1995-01-03").to_list() == [date(1995, 1, 3)]

    def test_monthly_rebalance_is_last_actual_session(self):
        r = calendar.rebalance_dates(calendar.sessions("2024-01-01", "2024-04-30"), "1mo").to_list()
        # March 29, 2024 was Good Friday: the last March session is the 28th.
        assert r == [date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 28), date(2024, 4, 30)]


# ---------------------------------------------------------------------------
# Security master
# ---------------------------------------------------------------------------

def _master() -> SecurityMaster:
    tickers = pl.DataFrame({
        "security_id": ["S1", "S1", "S2"],
        "ticker": ["FB", "META", "FB"],          # S2 reuses "FB" after S1 renamed
        "valid_from": [date(2012, 5, 18), date(2022, 6, 9), date(2024, 1, 2)],
        "valid_to": [date(2022, 6, 9), FAR, FAR],
    })
    gics = pl.DataFrame({
        "security_id": ["S1", "S1"],
        "gics_sector": ["Information Technology", "Communication Services"],
        "gics_industry": ["Internet Software", "Interactive Media"],
        "valid_from": [date(2012, 5, 18), date(2018, 9, 28)],
        "valid_to": [date(2018, 9, 28), FAR],
    })
    return SecurityMaster(tickers, gics)


class TestSecurityMaster:
    def test_reused_ticker_never_merges_two_securities(self):
        vendor = pl.DataFrame({
            "date": [date(2020, 1, 2), date(2023, 1, 3), date(2024, 1, 2)],
            "ticker": ["FB", "META", "FB"],
            "close": [200.0, 120.0, 10.0],
        })
        mapped = _master().map_tickers(vendor)
        assert mapped.select("date", "security_id").rows() == [
            (date(2020, 1, 2), "S1"), (date(2023, 1, 3), "S1"), (date(2024, 1, 2), "S2"),
        ]

    def test_unknown_ticker_on_date_is_reported(self):
        vendor = pl.DataFrame({"date": [date(2023, 1, 3)], "ticker": ["FB"], "close": [1.0]})
        assert _master().map_tickers(vendor).is_empty()
        assert _master().unmapped(vendor).rows() == [(date(2023, 1, 3), "FB", 1.0)]

    def test_overlapping_ticker_intervals_raise(self):
        bad = pl.DataFrame({
            "security_id": ["S1", "S2"], "ticker": ["X", "X"],
            "valid_from": [date(2020, 1, 1), date(2021, 1, 1)], "valid_to": [FAR, FAR],
        })
        with pytest.raises(ValueError, match="overlapping"):
            SecurityMaster(bad, _master().gics)

    def test_gics_is_point_in_time(self):
        df = pl.DataFrame({"date": [date(2018, 9, 27), date(2018, 9, 28)], "security_id": ["S1", "S1"]})
        assert _master().attach_gics(df)["gics_sector"].to_list() == [
            "Information Technology", "Communication Services",
        ]


# ---------------------------------------------------------------------------
# Corporate actions & total returns
# ---------------------------------------------------------------------------

def _px(rows):
    return pl.DataFrame(rows, schema=["date", "security_id", "close"], orient="row")


EMPTY_SPLITS = pl.DataFrame(schema={"security_id": pl.Utf8, "ex_date": pl.Date, "ratio": pl.Float64})
EMPTY_DIVS = pl.DataFrame(schema={"security_id": pl.Utf8, "ex_date": pl.Date, "amount": pl.Float64})
EMPTY_DELIST = pl.DataFrame(schema={
    "security_id": pl.Utf8, "delist_date": pl.Date, "delist_return": pl.Float64, "performance_related": pl.Boolean,
})


def _ret(df, d):
    return df.filter(pl.col("date") == d)["ret"].item()


class TestTotalReturns:
    def test_total_return_across_split_is_economic_return(self):
        """$400 → 10:1 split → $41 is a +2.5% day, not −90%."""
        px = _px([(date(2024, 6, 7), "A", 400.0), (date(2024, 6, 10), "A", 41.0)])
        splits = pl.DataFrame({"security_id": ["A"], "ex_date": [date(2024, 6, 10)], "ratio": [10.0]})
        r = total_returns(px, splits, EMPTY_DIVS, EMPTY_DELIST)
        assert _ret(r, date(2024, 6, 10)) == pytest.approx(0.025)
        assert r["has_action"].to_list() == [False, True]

    def test_dividend_is_added_on_ex_date(self):
        px = _px([(date(2024, 1, 2), "A", 100.0), (date(2024, 1, 3), "A", 99.0)])
        divs = pl.DataFrame({"security_id": ["A"], "ex_date": [date(2024, 1, 3)], "amount": [1.0]})
        assert _ret(total_returns(px, EMPTY_SPLITS, divs, EMPTY_DELIST), date(2024, 1, 3)) == pytest.approx(0.0)

    def test_action_on_unpriced_day_moves_to_next_priced_session(self):
        px = _px([(date(2024, 6, 6), "A", 400.0), (date(2024, 6, 10), "A", 40.0)])
        splits = pl.DataFrame({"security_id": ["A"], "ex_date": [date(2024, 6, 7)], "ratio": [10.0]})
        assert _ret(total_returns(px, splits, EMPTY_DIVS, EMPTY_DELIST), date(2024, 6, 10)) == pytest.approx(0.0)

    def test_delisting_return_applied_exactly_once_on_delisting_date(self):
        px = _px([(date(2024, 1, 2), "A", 10.0), (date(2024, 1, 3), "A", 11.0), (date(2024, 1, 4), "A", 99.0)])
        dl = pl.DataFrame({"security_id": ["A"], "delist_date": [date(2024, 1, 3)],
                           "delist_return": [-0.5], "performance_related": [True]})
        r = total_returns(px, EMPTY_SPLITS, EMPTY_DIVS, dl)
        assert r["date"].to_list() == [date(2024, 1, 2), date(2024, 1, 3)]  # rows after delisting dropped
        assert _ret(r, date(2024, 1, 3)) == pytest.approx(1.1 * 0.5 - 1)

    def test_delisting_without_price_row_adds_one(self):
        px = _px([(date(2024, 1, 2), "A", 10.0)])
        dl = pl.DataFrame({"security_id": ["A"], "delist_date": [date(2024, 1, 5)],
                           "delist_return": [None], "performance_related": [True]},
                          schema_overrides={"delist_return": pl.Float64})
        r = total_returns(px, EMPTY_SPLITS, EMPTY_DIVS, dl)
        assert _ret(r, date(2024, 1, 5)) == pytest.approx(-0.30)

    def test_unknown_non_performance_delisting_falls_back_to_zero(self):
        px = _px([(date(2024, 1, 2), "A", 10.0), (date(2024, 1, 3), "A", 10.0)])
        dl = pl.DataFrame({"security_id": ["A"], "delist_date": [date(2024, 1, 3)],
                           "delist_return": [float("nan")], "performance_related": [False]})
        assert _ret(total_returns(px, EMPTY_SPLITS, EMPTY_DIVS, dl), date(2024, 1, 3)) == pytest.approx(0.0)

    def test_duplicate_delistings_raise(self):
        dl = pl.DataFrame({"security_id": ["A", "A"], "delist_date": [date(2024, 1, 3)] * 2,
                           "delist_return": [0.0, 0.0], "performance_related": [False, False]})
        with pytest.raises(ValueError, match="one row per security_id"):
            total_returns(_px([(date(2024, 1, 2), "A", 1.0)]), EMPTY_SPLITS, EMPTY_DIVS, dl)


# ---------------------------------------------------------------------------
# Parquet store
# ---------------------------------------------------------------------------

class TestParquetStore:
    def test_roundtrip_partitioned_by_year(self, tmp_path):
        df = pl.DataFrame({"date": [date(2023, 12, 29), date(2024, 1, 2)], "security_id": ["A", "A"], "close": [1.0, 2.0]})
        store = ParquetStore(tmp_path)
        store.write("prices", df)
        assert [p.parent.name for p in store.files("prices")] == ["year=2023", "year=2024"]
        plt.assert_frame_equal(store.scan("prices").sort("date").collect(), df)

    def test_write_replaces_dataset(self, tmp_path):
        store = ParquetStore(tmp_path)
        store.write("t", pl.DataFrame({"date": [date(2020, 1, 2)], "x": [1]}))
        store.write("t", pl.DataFrame({"date": [date(2024, 1, 2)], "x": [2]}))
        assert store.scan("t").collect()["x"].to_list() == [2]
        assert [p.name for p in tmp_path.iterdir()] == ["t"]  # no .tmp / .old left behind

    def test_table_without_date_and_missing_dataset(self, tmp_path):
        store = ParquetStore(tmp_path)
        store.write("master", pl.DataFrame({"security_id": ["A"]}))
        assert store.scan("master").collect().height == 1
        with pytest.raises(FileNotFoundError):
            store.scan("nope")


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------

class TestSnapshot:
    def _write(self, tmp_path, df, inputs):
        return write_snapshot(df, tmp_path / "snaps", "universe", params=LiquidityFilter().as_kwargs(),
                              inputs=inputs, vendor="fixture", vendor_version="1")

    def test_same_snapshot_twice_has_same_content_hash(self, tmp_path):
        df = pl.DataFrame({"date": [date(2024, 1, 2)], "security_id": ["A"], "close": [1.5]})
        src = tmp_path / "in.parquet"
        df.write_parquet(src)
        first, second = self._write(tmp_path, df, [src]), self._write(tmp_path, df.clone(), [src])
        assert first == second
        loaded, manifest = load_snapshot(first)
        plt.assert_frame_equal(loaded, df)
        assert manifest["content_hash"] == content_hash(df)
        assert manifest["params"]["adv_floor"] == 25e6
        assert manifest["inputs"] == {str(src): hashlib.sha256(src.read_bytes()).hexdigest()}
        assert manifest["code_revision"]

    def test_different_data_gets_a_different_snapshot(self, tmp_path):
        a = pl.DataFrame({"x": [1.0]})
        assert self._write(tmp_path, a, []) != self._write(tmp_path, a.with_columns(pl.col("x") + 1e-12), [])

    def test_tampered_snapshot_is_rejected(self, tmp_path):
        folder = self._write(tmp_path, pl.DataFrame({"x": [1.0]}), [])
        pl.DataFrame({"x": [2.0]}).write_parquet(folder / "data.parquet")
        with pytest.raises(ValueError, match="does not match"):
            load_snapshot(folder)


# ---------------------------------------------------------------------------
# Data-quality report
# ---------------------------------------------------------------------------

class TestQualityReport:
    def test_each_check_fires(self):
        sessions = calendar.sessions("2024-06-03", "2024-06-14")
        d = sessions.to_list()
        prices = pl.DataFrame({
            "date": [d[0], d[1], d[2], d[3], d[3], d[0], d[1], date(2024, 6, 8)],
            "security_id": ["A", "A", "A", "A", "A", "B", "B", "B"],
            "close": [10.0, 30.0, 0.0, 5.0, 5.0, 100.0, 10.0, 10.0],
            "volume": [1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6],   # B's 10:1 split day volume did not move
        })
        membership = pl.DataFrame({
            "security_id": ["A", "B", "B"], "index": ["SP500"] * 3,
            "entry_date": [d[0], d[0], d[6]], "exit_date": [FAR, d[4], FAR],   # B re-enters 2 days later
        })
        splits = pl.DataFrame({"security_id": ["B"], "ex_date": [d[1]], "ratio": [10.0]})
        returns = total_returns(prices.unique(["date", "security_id"]), splits, EMPTY_DIVS, EMPTY_DELIST)
        exclusions = pl.DataFrame({"date": [d[9]], "security_id": ["A"], "exclusion": ["stale"]})

        report = quality_report(prices, membership, sessions, returns, splits, exclusions)
        found = set(report["check"].unique())
        assert found == {"coverage", "missing_bar", "duplicate_row", "off_session", "bad_price",
                         "suspicious_return", "split_volume", "membership_gap", "stale"}
        # A: +200%, -100%, then up from $0; B's 10:1 split day is explained by the action.
        assert report.filter(pl.col("check") == "suspicious_return")["security_id"].to_list() == ["A"] * 3
        assert report.filter((pl.col("check") == "coverage") & (pl.col("date") == d[0]))["value"].item() == 1.0
        assert report.filter(pl.col("check") == "off_session")["date"].to_list() == [date(2024, 6, 8)]


# ---------------------------------------------------------------------------
# End to end: vendor rows → security_id → universe → snapshot
# ---------------------------------------------------------------------------

def test_end_to_end_reproducible_snapshot(tmp_path):
    sessions = calendar.sessions("2023-11-01", "2024-02-29")
    n = sessions.len()
    vendor = pl.DataFrame({
        "date": pl.concat([sessions, sessions]),
        "ticker": ["FB"] * n + ["GOOG"] * n,
        "close": [300.0] * n + [140.0] * n,
        "volume": [2e7] * n + [3e7] * n,
    })
    tickers = pl.DataFrame({"security_id": ["S1", "S3"], "ticker": ["FB", "GOOG"],
                            "valid_from": [date(2012, 1, 3)] * 2, "valid_to": [FAR, FAR]})
    master = SecurityMaster(tickers, _master().gics)
    prices = master.map_tickers(vendor)
    membership = pl.DataFrame({"security_id": ["S1", "S3"], "index": ["SP500"] * 2,
                               "entry_date": [date(2020, 1, 2)] * 2, "exit_date": [FAR, date(2024, 2, 1)]})
    mcap = prices.select("date", "security_id", (pl.col("close") * 1e9).alias("market_cap"))

    store = ParquetStore(tmp_path / "store")
    store.write("prices", prices)
    builder = UniverseBuilder(membership, store.scan("prices").collect(), mcap, sessions)
    monthly = calendar.rebalance_dates(sessions, "1mo")
    rebal = monthly.filter(monthly >= date(2023, 12, 1))
    panel = builder.build_panel(rebal, index="SP500", **LiquidityFilter().as_kwargs())

    assert panel.group_by("date").len().sort("date")["len"].to_list() == [2, 2, 1]  # S3 left Feb 1
    snaps = [
        write_snapshot(builder.build_panel(rebal, index="SP500", **LiquidityFilter().as_kwargs()),
                       tmp_path / "snaps", "sp500_universe", params=LiquidityFilter().as_kwargs(),
                       inputs=store.files("prices"), vendor="fixture", vendor_version="1")
        for _ in range(2)
    ]
    assert snaps[0] == snaps[1]
    plt.assert_frame_equal(load_snapshot(snaps[0])[0], panel)
