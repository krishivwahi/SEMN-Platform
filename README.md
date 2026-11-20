# SEMN — Systematic Equity Research Platform

A research platform for **market-neutral US equity strategies**, built phase by phase:
point-in-time data → factor risk model → cross-sectional ranking → portfolio optimizer →
validation (CPCV, Deflated Sharpe Ratio) → tear-sheet. Python and Polars; a C++20
execution core is planned for Phase 6. Full plan and status: [ROADMAP.md](ROADMAP.md).

**Status:** Phase 1 (point-in-time data layer) is complete. Phase 2 (factor risk model) is in progress:
exposures, the cross-sectional regression and the risk estimates are built and tested on simulated data.

## What Phase 1 does

- **Survivorship-bias-free S&P 500 universe.** Index membership is applied point in time
  (`entry ≤ t < exit`), including companies that were later acquired or delisted.
- **Permanent security IDs.** Every row is keyed by `security_id`, never by ticker, so ticker
  changes (FB → META) and reused tickers (MRNA before and after Moderna) cannot mix two companies.
- **Raw prices only.** The $5 price floor and 20-day dollar volume use what actually traded that day;
  total returns are built from raw prices plus splits, dividends and delisting returns.
- **NYSE calendar.** Staleness and windows are counted in real trading sessions.
- **Reproducible snapshots.** Each universe build is saved under a hash of its contents, with a
  manifest of input-file hashes, parameters and code revision; the hash is checked on load.
- **Data-quality report.** Coverage, missing bars, duplicates, bad prices, unexplained
  >50% moves, split-day volume and membership gaps.

On the free data sources, 85% (2010) to 99.6% (2024) of historical S&P 500 members map to a
security with prices; the rest are listed in an `unmapped_membership` report. Known gaps of free
data (no delisting returns, no GICS history) are recorded under D1 in the roadmap.

## Layout

```
semn/data/
├── calendar.py            NYSE sessions, rebalance dates
├── security_master.py     security_id with dated ticker and GICS history
├── corporate_actions.py   total returns, delisting returns
├── universe.py            point-in-time universe and exclusion report
├── quality.py             data-quality report
├── store.py               Parquet store (year partitions)
├── snapshot.py            content-hashed snapshots + manifests
├── clickhouse_loader.py   optional ClickHouse tick/bar backend
└── sources/               free vendor adapter (Tiingo prices, S&P 500 membership)
semn/factors/
├── exposures.py           beta, momentum, reversal, volatility; standardization
├── risk_model.py          cross-sectional WLS: factor returns and residuals
└── covariance.py          XFXᵀ + D, Ledoit–Wolf, diagnostics, bias statistic
tests/                     offline tests; no network or API key needed
```

## Run

```bash
pip install -e ".[dev]"
pytest                      # 109 tests
```

Download the free data (key from tiingo.com; the free tier allows 50 requests/hour, so a
2010+ build takes about 15 hours and resumes where it stopped):

```bash
export TIINGO_API_KEY=...
python -m semn.data.sources.tiingo --start 2010-01-01 --root data/store --cache data/cache
```

Vendor data stays in `data/`, which is git-ignored (Tiingo's terms forbid redistribution).
