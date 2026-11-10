"""
tests/test_sources.py
============================
Free vendor adapter (Tiingo + GitHub S&P 500 membership), fully offline:
HTTP is replaced by a fake, so no key or network is needed.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date

import polars as pl
import pytest

from semn.data import LiquidityFilter, ParquetStore, SecurityMaster, UniverseBuilder, calendar, total_returns
from semn.data.sources import sp500_github, tiingo

FAR = date(9999, 12, 31)

MEMBERSHIP_CSV = """date,tickers
2020-01-02,"AAA,BBB,FB"
2021-03-01,"AAA,FB"
2022-06-09,"AAA,META"
2023-01-03,"AAA,BBB,META"
"""

LISTINGS_CSV = """ticker,exchange,assetType,priceCurrency,startDate,endDate
AAA,NYSE,Stock,USD,2000-01-03,2024-06-28
BBB,NYSE,Stock,USD,2015-01-02,2021-02-26
BBB,NASDAQ,ETF,USD,2022-11-01,2024-06-28
META,NASDAQ,Stock,USD,2012-05-18,2024-06-28
FB,BATS,ETF,USD,2024-01-02,2024-06-28
BRK.B,NYSE,Stock,USD,1996-05-09,2024-06-28
VFIAX,NMFQS,Mutual Fund,USD,2000-11-13,2024-06-28
000001,SHE,Stock,CNY,2007-01-04,2024-06-28
"""

RENAMES = [("META", "FB", None, "2022-06-09")]


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def _reference(tmp_path, renames=RENAMES):
    listings = tiingo.load_listings(_write(tmp_path, "st.csv", LISTINGS_CSV))
    members = sp500_github.load_membership(_write(tmp_path, "m.csv", MEMBERSHIP_CSV))
    return listings, members, tiingo.build_reference(listings, members, renames)


class TestMembership:
    def test_change_rows_become_intervals(self, tmp_path):
        m = sp500_github.load_membership(_write(tmp_path, "m.csv", MEMBERSHIP_CSV))
        assert m.select("ticker", "entry_date", "exit_date").rows() == [
            ("AAA", date(2020, 1, 2), FAR),
            ("BBB", date(2020, 1, 2), date(2021, 3, 1)),     # left, then re-entered
            ("BBB", date(2023, 1, 3), FAR),
            ("FB", date(2020, 1, 2), date(2022, 6, 9)),
            ("META", date(2022, 6, 9), FAR),
        ]

    def test_class_share_tickers_use_vendor_format(self, tmp_path):
        m = sp500_github.load_membership(_write(tmp_path, "m.csv", 'date,tickers\n2020-01-02,"BRK.B, bf.b"\n'))
        assert m["ticker"].to_list() == ["BF-B", "BRK-B"]


class TestListings:
    def test_keeps_any_exchange_and_asset_type_but_not_funds_or_foreign(self, tmp_path):
        listings = tiingo.load_listings(_write(tmp_path, "st.csv", LISTINGS_CSV))
        assert set(listings["ticker"]) == {"AAA", "BBB", "META", "FB", "BRK-B"}
        assert listings.filter(pl.col("ticker") == "BBB")["security_id"].to_list() == ["BBB@2015-01-02", "BBB@2022-11-01"]
        assert listings.filter(~pl.col("active"))["security_id"].to_list() == ["BBB@2015-01-02"]

    def test_zip_is_read(self, tmp_path):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("supported_tickers.csv", LISTINGS_CSV)
        path = tmp_path / "st.zip"
        path.write_bytes(buf.getvalue())
        assert tiingo.load_listings(path).height == 6


class TestReference:
    def test_rename_maps_old_ticker_to_current_listing(self, tmp_path):
        _, _, ref = _reference(tmp_path)
        meta = ref["membership"].filter(pl.col("security_id") == "META@2012-05-18")
        assert meta.select("entry_date", "exit_date").rows() == [(date(2020, 1, 2), date(2022, 6, 9)), (date(2022, 6, 9), FAR)]
        hist = ref["tickers"].filter(pl.col("security_id") == "META@2012-05-18")
        assert hist.select("ticker", "valid_from", "valid_to").rows() == [
            ("FB", date(2012, 5, 18), date(2022, 6, 9)),
            ("META", date(2022, 6, 9), FAR),
        ]

    def test_ticker_history_is_valid_security_master(self, tmp_path):
        _, _, ref = _reference(tmp_path)
        master = SecurityMaster(ref["tickers"], tiingo.NO_GICS)
        vendor = pl.DataFrame({"date": [date(2021, 1, 4), date(2024, 3, 1)], "ticker": ["FB", "FB"]})
        # FB in 2021 is Meta; FB in 2024 is an unrelated ETF listing.
        assert master.map_tickers(vendor)["security_id"].to_list() == ["META@2012-05-18", "FB@2024-01-02"]

    def test_reused_tickers_old_listing_is_reported_unmapped(self, tmp_path):
        """Tiingo only serves the newest BBB, so the 2020 BBB membership cannot be priced."""
        _, _, ref = _reference(tmp_path)
        assert ref["membership"].filter(pl.col("security_id").str.starts_with("BBB"))["entry_date"].to_list() == [date(2023, 1, 3)]
        assert ref["unmapped_membership"].select("ticker", "entry_date").rows() == [("BBB", date(2020, 1, 2))]

    def test_without_rename_old_ticker_is_suggested_not_linked(self, tmp_path):
        _, _, ref = _reference(tmp_path, renames=[])
        assert "FB" in ref["unmapped_membership"]["ticker"].to_list()
        assert ref["rename_suggestions"].select("ticker", "new_ticker").rows() == [("FB", "META")]


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

PRICE_HEADER = "date,close,high,low,open,volume,adjClose,adjHigh,adjLow,adjOpen,adjVolume,divCash,splitFactor\n"


def _price_csv(rows):
    """rows: (date, close, volume, divCash, splitFactor)"""
    return (PRICE_HEADER + "".join(
        f"{d},{c},{c},{c},{c},{v},{c},{c},{c},{c},{v},{dv},{sf}\n" for d, c, v, dv, sf in rows
    )).encode()


class FakeHttp:
    def __init__(self, responses):
        self.responses = responses  # url substring -> (status, body)
        self.calls = []

    def __call__(self, url, headers):
        self.calls.append((url, headers))
        for key, resp in self.responses.items():
            if key in url:
                return resp
        return 404, b""


class TestFetchPrices:
    def test_caches_and_never_puts_key_in_url(self, tmp_path):
        http = FakeHttp({"/AAA/": (200, _price_csv([("2024-01-02", 10.0, 1e6, 0.0, 1.0)]))})
        tiingo.fetch_prices(["AAA", "ZZZ"], "2024-01-01", "2024-01-31", "secret", tmp_path, 10**9, http)
        assert (tmp_path / "AAA.csv").stat().st_size > 0
        assert (tmp_path / "ZZZ.csv").stat().st_size == 0  # 404: remembered, not retried
        assert all("secret" not in url for url, _ in http.calls)
        assert http.calls[0][1]["Authorization"] == "Token secret"
        tiingo.fetch_prices(["AAA", "ZZZ"], "2024-01-01", "2024-01-31", "secret", tmp_path, 10**9, http)
        assert len(http.calls) == 2  # second run is all cache

    def test_rate_limit_keeps_progress(self, tmp_path):
        http = FakeHttp({"/AAA/": (200, _price_csv([("2024-01-02", 10.0, 1e6, 0.0, 1.0)])), "/BBB/": (429, b"")})
        with pytest.raises(tiingo.RateLimited, match="1 of 2"):
            tiingo.fetch_prices(["AAA", "BBB"], "2024-01-01", "2024-01-31", "k", tmp_path, 10**9, http)
        assert (tmp_path / "AAA.csv").exists() and not (tmp_path / "BBB.csv").exists()

    def test_bad_key_is_explained(self, tmp_path):
        with pytest.raises(RuntimeError, match="rejected the API key"):
            tiingo.fetch_prices(["AAA"], "2024-01-01", "2024-01-31", "k", tmp_path, 10**9, FakeHttp({"/AAA/": (403, b"")}))

    def test_other_http_errors_raise(self, tmp_path):
        with pytest.raises(RuntimeError, match="HTTP 500"):
            tiingo.fetch_prices(["AAA"], "2024-01-01", "2024-01-31", "k", tmp_path, 10**9, FakeHttp({"/AAA/": (500, b"x")}))


class TestPriceTables:
    def test_raw_prices_splits_dividends_delistings(self, tmp_path):
        (tmp_path / "META.csv").write_bytes(_price_csv([
            ("2022-06-08", 200.0, 1e6, 0.0, 1.0),
            ("2022-06-09", 20.5, 1e7, 0.0, 10.0),
            ("2022-06-10", 21.0, 1e7, 0.5, 1.0),
        ]))
        listings = pl.DataFrame({
            "security_id": ["META@2012-05-18", "OLD@2000-01-03"], "ticker": ["META", "OLD"],
            "start_date": [date(2012, 5, 18), date(2000, 1, 3)], "end_date": [date(2024, 6, 28), date(2021, 2, 26)],
            "active": [True, False],
        })
        t = tiingo.price_tables(listings, tmp_path)
        assert t["prices"]["close"].to_list() == [200.0, 20.5, 21.0]  # raw, not split-adjusted
        assert t["splits"].rows() == [("META@2012-05-18", date(2022, 6, 9), 10.0)]
        assert t["dividends"].rows() == [("META@2012-05-18", date(2022, 6, 10), 0.5)]
        assert t["delistings"].select("security_id", "delist_date").rows() == [("OLD@2000-01-03", date(2021, 2, 26))]
        r = total_returns(t["prices"], t["splits"], t["dividends"], t["delistings"])
        meta = r.filter(pl.col("security_id") == "META@2012-05-18")
        assert meta["ret"].to_list()[1:] == pytest.approx([0.025, (21.0 + 0.5) / 20.5 - 1])


# ---------------------------------------------------------------------------
# End to end, offline
# ---------------------------------------------------------------------------

def test_ingest_end_to_end(tmp_path):
    sessions = calendar.sessions("2022-01-03", "2023-03-31").to_list()
    def history(close):
        return _price_csv([(d.isoformat(), close, 1e6, 0.0, 1.0) for d in sessions])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("supported_tickers.csv", LISTINGS_CSV)
    http = FakeHttp({
        tiingo.TICKERS_URL: (200, buf.getvalue()),
        sp500_github.COMMIT: (200, MEMBERSHIP_CSV.encode()),
        "/AAA/": (200, history(50.0)),
        "/META/": (200, history(300.0)),
        "/BBB/": (200, history(40.0)),
    })
    store = ParquetStore(tmp_path / "store")
    counts = tiingo.ingest(store, "k", "2022-01-03", "2023-03-31", tmp_path / "cache", 10**9, http,
                           renames=RENAMES)
    assert counts["prices"] > 0 and counts["unmapped_membership"] == 1

    tables = {n: store.scan(n).collect() for n in ("membership", "prices")}
    builder = UniverseBuilder(tables["membership"], tables["prices"],
                              pl.DataFrame(schema={"date": pl.Date, "security_id": pl.Utf8, "market_cap": pl.Float64}),
                              pl.Series(sessions, dtype=pl.Date))
    panel = builder.build_panel([date(2022, 3, 31), date(2023, 3, 31)], index="SP500", **LiquidityFilter().as_kwargs())
    assert panel.select("date", "security_id").rows() == [
        (date(2022, 3, 31), "AAA@2000-01-03"), (date(2022, 3, 31), "META@2012-05-18"),
        (date(2023, 3, 31), "AAA@2000-01-03"), (date(2023, 3, 31), "BBB@2022-11-01"), (date(2023, 3, 31), "META@2012-05-18"),
    ]


def test_cli_requires_key(monkeypatch, capsys):
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    assert tiingo.main(["--start", "2024-01-01"]) == 1
    assert "TIINGO_API_KEY" in capsys.readouterr().err
