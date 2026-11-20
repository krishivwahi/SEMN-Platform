"""
semn/factors/risk_model.py
==========================
Cross-sectional factor regression (Phase 2).

On each date t:  r_{t+1} = X_t f_t + ε_{t+1}, fitted by weighted least squares.

* ``X_t`` = standardized exposures on t plus a market intercept, or sector dummies
  in place of the intercept when a sector column is given (the dummies sum to the
  intercept, so both together would be collinear).
* Weights are √market cap when supplied (big stocks have less noisy idiosyncratic
  returns), otherwise equal.
* By construction the residuals are weighted-orthogonal to every exposure:
  ``X_tᵀ W ε = 0``.  Alpha models (Phase 3) train on these residuals.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl

_KEY = "security_id"


def next_returns(returns: pl.DataFrame) -> pl.DataFrame:
    """``date, security_id, ret_next``: each row's *next* return, aligned on the exposure date."""
    return (
        returns.sort(_KEY, "date")
        .select("date", _KEY, pl.col("ret").shift(-1).over(_KEY).alias("ret_next"))
    )


def fit_factor_returns(
    panel: pl.DataFrame,
    factors: list[str],
    weight: Optional[str] = None,
    sector: Optional[str] = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Fit the regression on every date.

    Parameters
    ----------
    panel:   ``date, security_id, ret_next`` + standardized *factors* (+ *weight*, *sector*)
    factors: style columns (already standardized, no nulls)
    weight:  market-cap column; regression weights are its square root.  ``None`` = equal.
    sector:  sector label column; adds one dummy per sector instead of the intercept.

    Returns
    -------
    factor_returns: ``date`` + one column per factor (``market`` or ``sector:<name>``, then styles)
    residuals:      ``date, security_id, resid``
    """
    cols = [c for c in ["date", _KEY, "ret_next", *factors, weight, sector] if c]
    data = panel.select(cols).drop_nulls().sort("date", _KEY)
    sectors = sorted(data[sector].unique().to_list()) if sector else []
    names = ([f"sector:{s}" for s in sectors] if sector else ["market"]) + factors

    f_rows, resid_parts = [], []
    for (day,), g in data.group_by("date", maintain_order=True):
        style = g.select(factors).to_numpy()
        if sector:
            lead = (g[sector].to_numpy()[:, None] == np.array(sectors)[None, :]).astype(float)
        else:
            lead = np.ones((g.height, 1))
        X = np.hstack([lead, style])
        y = g["ret_next"].to_numpy()
        # WLS with w = √cap: scale each row by √w = cap^¼, then plain least squares minimises Σ w·ε².
        sw = g[weight].to_numpy() ** 0.25 if weight else np.ones(g.height)
        # lstsq gives the minimum-norm answer when a sector is empty that day (all-zero column).
        f, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        f_rows.append((day, *f))
        resid_parts.append(g.select("date", _KEY).with_columns(pl.Series("resid", y - X @ f)))

    factor_returns = pl.DataFrame(f_rows, schema=["date", *names], orient="row")
    residuals = pl.concat(resid_parts) if resid_parts else pl.DataFrame(schema={"date": pl.Date, _KEY: pl.Utf8, "resid": pl.Float64})
    return factor_returns, residuals
