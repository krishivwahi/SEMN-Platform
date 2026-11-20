"""
semn/factors/covariance.py
==========================
Risk estimates and their checks (Phase 2).

Σ = X F Xᵀ + D: factor covariance F (EWMA of factor returns) plus specific
variance D (EWMA of squared residuals).  The sample covariance shrunk with
Ledoit–Wolf is the baseline it is compared against.  Diagnostics: symmetry,
smallest eigenvalue (PSD) and condition number; the bias statistic checks
out-of-sample calibration (≈ 1 when forecast volatility is right).
"""

from __future__ import annotations

import numpy as np
import polars as pl

_KEY = "security_id"
HALF_LIFE = 90
MIN_SPECIFIC_OBS = 20


def ewma_weights(n: int, half_life: float) -> np.ndarray:
    """Weights for *n* rows ordered oldest → newest, newest weighted most, summing to 1."""
    w = 0.5 ** (np.arange(n)[::-1] / half_life)
    return w / w.sum()


def ewma_cov(R: np.ndarray, half_life: float = HALF_LIFE) -> np.ndarray:
    """EWMA covariance of the rows of *R* (T × K, oldest first), demeaned with the same weights."""
    w = ewma_weights(len(R), half_life)
    X = R - w @ R
    return (X * w[:, None]).T @ X


def factor_covariance(factor_returns: pl.DataFrame, half_life: float = HALF_LIFE) -> tuple[list[str], np.ndarray]:
    """F from ``fit_factor_returns`` output; returns (factor names, K × K matrix)."""
    names = [c for c in factor_returns.columns if c != "date"]
    R = factor_returns.sort("date").select(names).fill_null(0.0).to_numpy()
    return names, ewma_cov(R, half_life)


def specific_variance(residuals: pl.DataFrame, half_life: float = HALF_LIFE) -> pl.DataFrame:
    """
    ``security_id, specific_var``: EWMA of squared residuals (mean assumed 0).

    Age is measured in dates of the whole panel, so a security that stopped
    trading is not treated as current.  Needs ``MIN_SPECIFIC_OBS`` residuals.
    """
    ages = residuals.select(pl.col("date").unique().sort(descending=True)).with_row_index("age")
    w = 0.5 ** (pl.col("age") / half_life)
    return (
        residuals.join(ages, on="date")
        .group_by(_KEY)
        .agg(((w * pl.col("resid") ** 2).sum() / w.sum()).alias("specific_var"), pl.len().alias("n"))
        .filter(pl.col("n") >= MIN_SPECIFIC_OBS)
        .select(_KEY, "specific_var")
        .sort(_KEY)
    )


def asset_covariance(X: np.ndarray, F: np.ndarray, specific: np.ndarray) -> np.ndarray:
    """Σ = X F Xᵀ + diag(specific) for N assets with exposures X (N × K)."""
    return X @ F @ X.T + np.diag(specific)


def ledoit_wolf(R: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Ledoit–Wolf (2004) shrinkage of the sample covariance of *R* (T × N) towards
    a scaled identity.  Returns (shrunk covariance, shrinkage in [0, 1]).
    Same estimator as ``sklearn.covariance.ledoit_wolf``.
    """
    T, N = R.shape
    X = R - R.mean(axis=0)
    S = X.T @ X / T
    mu = np.trace(S) / N
    X2 = X ** 2
    # Variance of the sample covariance entries (beta) vs distance to the target (delta).
    beta = ((X2.T @ X2).sum() / T - (S ** 2).sum()) / (N * T)
    delta = ((S - mu * np.eye(N)) ** 2).sum() / N
    shrink = 0.0 if delta == 0 else min(beta, delta) / delta
    return (1 - shrink) * S + shrink * mu * np.eye(N), shrink


def diagnostics(S: np.ndarray) -> dict[str, float | bool]:
    """Symmetry, smallest eigenvalue (PSD check: ≥ −1e-10) and condition number."""
    eig = np.linalg.eigvalsh((S + S.T) / 2)
    return {
        "symmetric": bool(np.allclose(S, S.T)),
        "min_eigenvalue": float(eig[0]),
        "psd": bool(eig[0] >= -1e-10),
        "condition_number": float(eig[-1] / eig[0]) if eig[0] > 0 else float("inf"),
    }


def bias_statistic(realized: np.ndarray, forecast_vol: np.ndarray) -> float:
    """Std of realized / forecast volatility.  ≈ 1 when calibrated; > 1 means risk is underestimated."""
    z = np.asarray(realized) / np.asarray(forecast_vol)
    return float(np.std(z[np.isfinite(z)], ddof=1))
