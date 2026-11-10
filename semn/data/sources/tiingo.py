"""
semn/data/sources/tiingo.py
==========================
Free vendor adapter (decision D1, interim): Tiingo end-of-day prices + S&P 500
membership from GitHub → raw Parquet tables in a :class:`ParquetStore`.

Why Tiingo: its free tier returns **raw** OHLCV with ``divCash`` and
``splitFactor`` on each ex-date, and its ticker list keeps delisted names
(e.g. CELG, TWTR, XLNX) with start/end dates — enough to avoid survivorship bias
for most S&P 500 history.  Free key: https://www.tiingo.com (env ``TIINGO_API_KEY``).

Identifiers
-----------
Each Tiingo listing (one row of ``supported_tickers``) is one security:
``security_id = "<ticker>@<first price date>"``.  A reused ticker (MRNA 1993-2018
vs Moderna) is therefore two securities.  Tiingo files a renamed company's whole
history under its *current* ticker; :func:`build_reference` links the old
membership ticker (FB) to that listing (META) when the membership file shows
the swap on the same day.

Known free-data limits (see ROADMAP D1)
---------------------------------------
* No delisting returns or reasons: delistings get the 0% fallback.
* No GICS history and no shares outstanding (market cap); Phase 2 needs another source.
* A reused ticker's *older* listing usually has no downloadable prices.
* Free tier rate limits (50 requests/hour at the time of writing): downloads are
  cached per ticker and resumable; rerun after :class:`RateLimited`.
* Tiingo's terms forbid redistribution: data stays in the local store, never in git.

Usage
-----
    TIINGO_API_KEY=... python -m semn.data.sources.tiingo --start 2010-01-01 --root data/store
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Optional

import polars as pl

from semn.data.security_master import SecurityMaster
from semn.data.sources import sp500_github
from semn.data.store import ParquetStore

TICKERS_URL = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
PRICES_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start}&endDate={end}&format=csv"
FAR = sp500_github.FAR
#: A listing whose last price is within this many days of the newest one is still trading.
ACTIVE_LAG_DAYS = 7

HttpGet = Callable[[str, dict], tuple[int, bytes]]

#: Free data has no GICS history; Phase 2 needs another source for sectors.
NO_GICS = pl.DataFrame(schema={"security_id": pl.Utf8, "gics_sector": pl.Utf8, "gics_industry": pl.Utf8,
                               "valid_from": pl.Date, "valid_to": pl.Date})


class RateLimited(RuntimeError):
    """Tiingo refused more requests; everything fetched so far is cached."""


# ---------------------------------------------------------------------------
# Reference data: listings, ticker history, membership
# ---------------------------------------------------------------------------

#: Ticker changes Tiingo files under the company's *current* ticker, checked by hand:
#: ``(current ticker, old ticker, old from, old to)``; ``None`` = since the listing's first price.
#: Dates are the swap dates in the membership file.  Add more from ``rename_suggestions``
#: only after checking them: a wrong link credits one company's returns to another.
RENAMES: list[tuple[str, str, Optional[str], str]] = [
    ("META", "FB", None, "2022-06-09"),
    ("ELV", "ANTM", None, "2022-06-28"),
    ("GEN", "SYMC", None, "2019-11-05"),
    ("GEN", "NLOK", "2019-11-05", "2022-11-08"),
    ("RVTY", "PKI", None, "2023-05-16"),
    ("EG", "RE", None, "2023-07-10"),
    ("DAY", "CDAY", None, "2024-02-01"),
    ("CPAY", "FLT", None, "2024-03-25"),
    ("WTW", "WLTW", None, "2022-01-10"),
    ("BALL", "BLL", None, "2022-05-10"),
    ("BBWI", "LB", None, "2021-08-03"),
    ("CTRA", "COG", None, "2021-10-04"),
    ("LUMN", "CTL", None, "2020-09-18"),
    ("J", "JEC", None, "2019-12-10"),
    ("GL", "TMK", None, "2019-08-08"),
    ("LHX", "HRS", None, "2019-06-01"),
    ("COR", "ABC", None, "2023-08-30"),
    ("DOC", "HCP", None, "2019-11-05"),
    ("DOC", "PEAK", "2019-11-05", "2024-03-04"),
    ("FISV", "FI", "2023-06-07", "2025-11-11"),
    ("RTX", "UTX", None, "2020-04-03"),
    ("BKR", "BHGE", None, "2019-10-18"),
    ("DD", "DWDP", "2017-09-01", "2019-06-03"),
    ("MRSH", "MMC", None, "2026-01-14"),
    ("BNY", "BK", None, "2026-05-21"),
]


def load_listings(path: str | Path) -> pl.DataFrame:
    """
    US-dollar listings from ``supported_tickers`` (``.zip`` or ``.csv``).

    Every exchange is kept (failed companies move to OTC, e.g. SIVBQ) and the
    vendor's asset type is ignored apart from mutual funds: it is unreliable
    (L3 ``LLL`` and IHS Markit ``INFO`` are filed as ETFs).

    Returns ``security_id, ticker, start_date, end_date, active``.
    """
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix == ".zip":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            raw = z.read(next(n for n in z.namelist() if n.endswith(".csv")))
    listings = (
        pl.read_csv(raw, infer_schema_length=0)
        .filter((pl.col("priceCurrency") == "USD") & (pl.col("assetType") != "Mutual Fund"))
        .select(
            sp500_github.normalise_ticker("ticker"),
            pl.col("startDate").str.to_date(strict=False).alias("start_date"),
            pl.col("endDate").str.to_date(strict=False).alias("end_date"),
        )
        .drop_nulls()
        .unique(["ticker", "start_date"], keep="first")
    )
    newest = listings["end_date"].max()
    return listings.select(
        (pl.col("ticker") + "@" + pl.col("start_date").cast(pl.Utf8)).alias("security_id"),
        "ticker", "start_date", "end_date",
        (pl.col("end_date") >= newest - timedelta(days=ACTIVE_LAG_DAYS)).alias("active"),
    ).sort("ticker", "start_date")


def primary_listings(listings: pl.DataFrame) -> pl.DataFrame:
    """
    One listing per ticker: the one Tiingo's price API serves (active, else the latest).

    Other listings of a reused ticker have no downloadable prices, so nothing maps to them.
    """
    return (
        listings.sort("ticker", "active", "start_date")
        .group_by("ticker", maintain_order=True)
        .last()
    )


def build_reference(
    listings: pl.DataFrame,
    members: pl.DataFrame,
    renames: list[tuple[str, str, Optional[str], str]] = RENAMES,
) -> dict[str, pl.DataFrame]:
    """
    Ticker history and membership keyed by ``security_id``.

    Parameters
    ----------
    listings: :func:`load_listings` output.
    members:  :func:`sp500_github.load_membership` output (``ticker, index, entry_date, exit_date``).
    renames:  hand-checked ticker changes (see :data:`RENAMES`).

    Returns
    -------
    dict with
    ``tickers``             ``security_id, ticker, valid_from, valid_to`` (for :class:`SecurityMaster`)
    ``membership``          ``security_id, index, entry_date, exit_date``
    ``unmapped_membership`` membership rows with no listing at all
    ``rename_suggestions``  unmapped rows that look like renames (a lone same-day swap); check, then add to RENAMES
    """
    primary = primary_listings(listings).select(
        "security_id", "ticker",
        pl.col("start_date").alias("valid_from"),
        pl.when(pl.col("active")).then(pl.lit(FAR)).otherwise(pl.col("end_date") + timedelta(days=1)).alias("valid_to"),
    )
    aliases = (
        pl.DataFrame(renames, schema={"current": pl.Utf8, "ticker": pl.Utf8, "from": pl.Utf8, "to": pl.Utf8}, orient="row")
        .join(primary.rename({"ticker": "current"}), on="current")
        .select(
            "security_id", "ticker",
            pl.coalesce(pl.col("from").str.to_date(), pl.col("valid_from")).alias("valid_from"),
            pl.col("to").str.to_date().alias("valid_to"),
        )
    )
    # A renamed listing carries its current ticker only outside its old-ticker periods:
    # pair the sorted period starts with the sorted period ends.
    starts = pl.concat([
        primary.select("security_id", "valid_from"),
        aliases.select("security_id", pl.col("valid_to").alias("valid_from")),
    ])
    ends = pl.concat([
        primary.select("security_id", "valid_to"),
        aliases.select("security_id", pl.col("valid_from").alias("valid_to")),
    ])
    rank = pl.int_range(pl.len()).over("security_id").alias("k")
    own = (
        starts.sort("security_id", "valid_from").with_columns(rank)
        .join(ends.sort("security_id", "valid_to").with_columns(rank), on=["security_id", "k"])
        .join(primary.select("security_id", "ticker"), on="security_id")
        .filter(pl.col("valid_from") < pl.col("valid_to"))
        .select("security_id", "ticker", "valid_from", "valid_to")
    )
    # A hand-checked rename beats a vendor listing that claims the same ticker at the same time
    # (Tiingo has a 2-day "PEAK" listing inside Healthpeak's PEAK years).
    shadowed = (
        own.join(aliases.select("ticker", pl.col("valid_from").alias("a_from"), pl.col("valid_to").alias("a_to")), on="ticker")
        .filter((pl.col("valid_from") < pl.col("a_to")) & (pl.col("valid_to") > pl.col("a_from")))
        .select("security_id", "valid_from")
    )
    tickers = pl.concat([own.join(shadowed, on=["security_id", "valid_from"], how="anti"), aliases]).sort("security_id", "valid_from")

    overlaps = (pl.col("valid_from") < pl.col("exit_date")) & (pl.col("valid_to") > pl.col("entry_date"))
    matched = members.join(tickers, on="ticker").filter(overlaps)
    membership = matched.select(
        "security_id", "index",
        pl.max_horizontal("entry_date", "valid_from").alias("entry_date"),
        pl.min_horizontal("exit_date", "valid_to").alias("exit_date"),
    ).sort("security_id", "entry_date")
    unmapped = members.join(matched.select("ticker", "entry_date").unique(), on=["ticker", "entry_date"], how="anti")

    # Suggestions only: on rebalance days a delisted name and a new member often swap on the
    # same day too (FRC and AXON), so these are never linked automatically.
    lone = (
        members.group_by("exit_date").len().rename({"len": "n_exits"})
        .join(members.group_by(pl.col("entry_date").alias("exit_date")).len().rename({"len": "n_entries"}), on="exit_date")
        .filter((pl.col("n_exits") == 1) & (pl.col("n_entries") == 1))
    )
    suggestions = (
        unmapped.join(lone.select("exit_date"), on="exit_date")
        .join(members.select(pl.col("ticker").alias("new_ticker"), pl.col("entry_date").alias("exit_date")), on="exit_date")
        .join(primary.rename({"ticker": "new_ticker"}), on="new_ticker")
        .filter(pl.col("valid_from") <= pl.col("entry_date"))
        .select("ticker", "new_ticker", "entry_date", "exit_date", "security_id")
        .sort("exit_date")
    )
    return {"tickers": tickers, "membership": membership, "unmapped_membership": unmapped, "rename_suggestions": suggestions}


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def _http_get(url: str, headers: dict) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    api_key: str,
    cache_dir: str | Path,
    requests_per_hour: int = 50,
    http_get: HttpGet = _http_get,
) -> None:
    """
    Download each ticker's raw daily CSV into ``cache_dir/<ticker>.csv`` (skipped if cached).

    A ticker Tiingo does not have gets an empty file so it is not asked for again.
    Raises :class:`RateLimited` on HTTP 429; rerun later to continue.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    todo = [t for t in dict.fromkeys(tickers) if not (cache / f"{t}.csv").exists()]
    pause = 3600 / requests_per_hour
    next_at = 0.0
    for i, ticker in enumerate(todo):
        time.sleep(max(0.0, next_at - time.monotonic()))
        next_at = time.monotonic() + pause
        url = PRICES_URL.format(ticker=urllib.parse.quote(ticker), start=start, end=end)
        # The key goes in a header so it never appears in URLs or logs.
        status, body = http_get(url, {"Authorization": f"Token {api_key}", "Content-Type": "text/csv"})
        if status == 429:
            raise RateLimited(f"rate limited after {i} of {len(todo)} downloads; rerun later to continue")
        if status in (401, 403):
            raise RuntimeError(f"Tiingo rejected the API key (HTTP {status}); check TIINGO_API_KEY")
        if status == 404:
            body = b""
        elif status != 200:
            raise RuntimeError(f"Tiingo returned HTTP {status} for {ticker}: {body[:200]!r}")
        tmp = cache / f"{ticker}.csv.tmp"
        tmp.write_bytes(body)
        tmp.replace(cache / f"{ticker}.csv")  # a crash mid-write never leaves a truncated cache file


def price_tables(listings: pl.DataFrame, cache_dir: str | Path) -> dict[str, pl.DataFrame]:
    """
    Raw ``prices``, ``splits``, ``dividends`` and ``delistings`` for primary *listings* from the CSV cache.

    A ticker's download belongs to its primary listing (one listing per ticker, so no row is
    credited twice); rows outside the listing's date range are dropped.
    """
    frames = []
    for ticker in listings["ticker"].unique().sort():
        path = Path(cache_dir) / f"{ticker}.csv"
        if path.exists() and path.stat().st_size:
            frames.append(pl.read_csv(path, infer_schema_length=0).with_columns(pl.lit(ticker).alias("ticker")))
    num = ["open", "high", "low", "close", "volume", "divCash", "splitFactor"]
    rows = (
        pl.concat(frames, how="diagonal") if frames
        else pl.DataFrame(schema={"date": pl.Utf8, "ticker": pl.Utf8, **{c: pl.Utf8 for c in num}})
    ).select(
        pl.col("date").str.slice(0, 10).str.to_date().alias("date"),
        "ticker",
        *[pl.col(c).cast(pl.Float64) for c in num],
    )
    rows = (
        rows.join(listings.select("security_id", "ticker", "start_date", "end_date"), on="ticker")
        .filter((pl.col("date") >= pl.col("start_date")) & (pl.col("date") <= pl.col("end_date")))
        .unique(["date", "security_id"], keep="first")
        .sort("security_id", "date")
    )
    return {
        "prices": rows.select("date", "security_id", "open", "high", "low", "close", "volume"),
        "splits": rows.filter(pl.col("splitFactor") != 1.0)
                      .select("security_id", pl.col("date").alias("ex_date"), pl.col("splitFactor").alias("ratio")),
        "dividends": rows.filter(pl.col("divCash") > 0)
                         .select("security_id", pl.col("date").alias("ex_date"), pl.col("divCash").alias("amount")),
        # No delisting returns or reasons in free data: null return, unknown reason → 0% fallback.
        "delistings": listings.filter(~pl.col("active")).select(
            "security_id",
            pl.col("end_date").alias("delist_date"),
            pl.lit(None, dtype=pl.Float64).alias("delist_return"),
            pl.lit(None, dtype=pl.Boolean).alias("performance_related"),
        ),
    }


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def download(url: str, path: Path, http_get: HttpGet = _http_get) -> Path:
    """Fetch a public file once; later calls reuse the copy (delete it to refresh)."""
    if not path.exists():
        status, body = http_get(url, {})
        if status != 200:
            raise RuntimeError(f"HTTP {status} downloading {url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".tmp").write_bytes(body)
        path.with_suffix(".tmp").replace(path)
    return path


def ingest(
    store: ParquetStore,
    api_key: str,
    start: str,
    end: str,
    cache_dir: str | Path,
    requests_per_hour: int = 50,
    http_get: HttpGet = _http_get,
    renames: list[tuple[str, str, Optional[str], str]] = RENAMES,
) -> dict[str, int]:
    """
    Build every raw Phase 1 table for S&P 500 members active in ``[start, end]``.

    Writes ``tickers, membership, unmapped_membership, rename_suggestions, prices,
    splits, dividends, delistings`` to *store* and returns their row counts.
    """
    cache = Path(cache_dir)
    listings = load_listings(download(TICKERS_URL, cache / "supported_tickers.zip", http_get))
    members = sp500_github.load_membership(download(sp500_github.URL, cache / "sp500_membership.csv", http_get))
    reference = build_reference(listings, members, renames)
    SecurityMaster(reference["tickers"], NO_GICS)  # refuses overlapping ticker history before anything is written

    in_window = reference["membership"].filter(
        (pl.col("entry_date") <= date.fromisoformat(end)) & (pl.col("exit_date") > date.fromisoformat(start))
    )
    used = listings.join(in_window.select("security_id").unique(), on="security_id")
    fetch_prices(used["ticker"].unique().sort().to_list(), start, end, api_key, cache / "tiingo",
                 requests_per_hour, http_get)

    tables = {**reference, **price_tables(used, cache / "tiingo")}
    for name, df in tables.items():
        store.write(name, df)
    return {name: df.height for name, df in tables.items()}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download free S&P 500 data (Tiingo + GitHub membership).")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--root", default="data/store", help="ParquetStore folder")
    parser.add_argument("--cache", default="data/cache", help="download cache folder")
    parser.add_argument("--requests-per-hour", type=int, default=50)
    args = parser.parse_args(argv)
    key = os.environ.get("TIINGO_API_KEY")
    if not key:
        print("Set TIINGO_API_KEY (free at https://www.tiingo.com).", file=sys.stderr)
        return 1
    try:
        counts = ingest(ParquetStore(args.root), key, args.start, args.end, args.cache, args.requests_per_hour)
    except RateLimited as e:
        print(e, file=sys.stderr)
        return 2
    for name, n in counts.items():
        print(f"{name:22s} {n:>10,d} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
