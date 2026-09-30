#ingests and cleans historical OHLCV data

"""OHLCV data loading with local parquet caching.
Single normalized schema everywhere downstream:
    index:   DatetimeIndex (tz-naive, daily, sorted, unique), named "timestamp"
    columns: open, high, low, close, volume (float64)
Prices are split/dividend adjusted (auto_adjust=True) so close is directly
comparable across time. That decision is made once, here, and never revisited
downstream.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache"

COLUMNS = ["open", "high", "low", "close", "volume"]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce any raw OHLCV frame into the canonical schema.
    Lowercases columns, enforces column order and dtype, strips timezone,
    drops duplicate index entries, sorts, and drops bars with missing
    open/close (a bar you cannot trade or mark is not a bar).
    """
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    out = out[COLUMNS].astype("float64")
    out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    out = out.dropna(subset=["open", "close"])
    out.index.name = "timestamp"
    return out


def load_ohlcv(symbol: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """Load daily OHLCV for one symbol, hitting the local parquet cache first."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{symbol}_{start}_{end}.parquet"
    if use_cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    import yfinance as yf

    raw = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for {symbol} between {start} and {end}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = normalize(raw)
    df.to_parquet(cache_file)
    return df
