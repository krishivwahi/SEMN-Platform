"""
semn/factors/exposures.py
==========================
Point-in-time style exposures and cross-sectional standardization (Phase 2).

Every exposure on date t uses returns up to and including t only, so it can
explain the return from t to t+1 without look-ahead.  Windows count a
security's own return rows (its trading sessions), with a minimum number of
observations; a missing window gives a null exposure.

Exposures
---------
beta        OLS beta of the security's return on the market return, 252 rows (min 126)
momentum    sum of log returns from t-251 to t-21 (12-1 month momentum; skips the last month)
reversal    sum of log returns over the last 21 rows (1-month reversal)
volatility  standard deviation of daily returns over 60 rows (min 40)
size        log market cap (only when market cap is supplied)
"""

from __future__ import annotations

from typing import Optional

import polars as pl

_KEY = "security_id"

BETA_WINDOW, BETA_MIN = 252, 126
MOMENTUM_WINDOW, MOMENTUM_SKIP = 252, 21
REVERSAL_WINDOW = 21
VOL_WINDOW, VOL_MIN = 60, 40
STYLES = ["beta", "momentum", "reversal", "volatility"]


def style_exposures(
    returns: pl.DataFrame,
    market: pl.DataFrame,
    market_cap: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    Raw (unstandardized) style exposures.

    Parameters
    ----------
    returns:    ``date, security_id, ret`` (``corporate_actions.total_returns`` output)
    market:     ``date, ret`` market return (e.g. the benchmark's total return)
    market_cap: optional ``date, security_id, market_cap``; adds ``size``

    Returns
    -------
    pl.DataFrame
        ``date, security_id`` + ``STYLES`` (+ ``size``), one row per input return row.
    """
    mom_len = MOMENTUM_WINDOW - MOMENTUM_SKIP
    both = pl.col("ret").is_not_null() & pl.col("mkt").is_not_null()
    x = pl.when(both).then(pl.col("mkt"))
    y = pl.when(both).then(pl.col("ret"))

    def mean(e: pl.Expr) -> pl.Expr:
        return e.rolling_mean(BETA_WINDOW, min_samples=BETA_MIN).over(_KEY)

    out = (
        returns.select("date", _KEY, "ret")
        .join(market.select("date", pl.col("ret").alias("mkt")), on="date", how="left")
        .sort(_KEY, "date")
        .with_columns(pl.col("ret").log1p().alias("lr"))
        .with_columns(
            ((mean(x * y) - mean(x) * mean(y)) / (mean(x * x) - mean(x) ** 2)).alias("beta"),
            pl.col("lr").rolling_sum(mom_len, min_samples=int(0.8 * mom_len)).shift(MOMENTUM_SKIP).over(_KEY).alias("momentum"),
            pl.col("lr").rolling_sum(REVERSAL_WINDOW, min_samples=int(0.8 * REVERSAL_WINDOW)).over(_KEY).alias("reversal"),
            pl.col("ret").rolling_std(VOL_WINDOW, min_samples=VOL_MIN).over(_KEY).alias("volatility"),
        )
        .select("date", _KEY, *STYLES)
    )
    if market_cap is not None:
        out = out.join(
            market_cap.select("date", _KEY, pl.col("market_cap").log().alias("size")),
            on=["date", _KEY], how="left",
        )
    return out


def standardize(
    df: pl.DataFrame,
    cols: list[str],
    weight: Optional[str] = None,
    clip: tuple[float, float] = (0.01, 0.99),
) -> pl.DataFrame:
    """
    Per date: winsorize at the *clip* quantiles, then z-score.

    The mean is weighted by *weight* (cap-weighted when ``weight="market_cap"``;
    equal-weighted when ``None``) and the standard deviation is equal-weighted, so
    a cap-weighted market portfolio has zero exposure to every style.  Missing
    exposures become 0 (the weighted mean) after scaling.
    """
    w = pl.lit(1.0) if weight is None else pl.col(weight)

    def z(c: str) -> pl.Expr:
        lo = pl.col(c).quantile(clip[0]).over("date")
        hi = pl.col(c).quantile(clip[1]).over("date")
        v = pl.col(c).clip(lo, hi)
        wv = pl.when(v.is_not_null()).then(w)
        mu = (wv * v).sum().over("date") / wv.sum().over("date")
        return ((v - mu) / v.std().over("date")).fill_null(0.0).fill_nan(0.0).alias(c)

    return df.with_columns(z(c) for c in cols)
