#sharpe, max drawdown and cagr math


from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series) -> float:
    years = len(equity) / TRADING_DAYS
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def sharpe(equity: pd.Series, risk_free_annual: float = 0.0) -> float:
    """
    Mean daily excess return over its standard deviation. Rewards return
    per unit of volatility rather than raw return, which is why a strategy
    with half the return of buy-and-hold can still be the better strategy.
    """
    rets = equity.pct_change().dropna()
    excess = rets - risk_free_annual / TRADING_DAYS
    sd = float(excess.std())
    if sd == 0.0 or np.isnan(sd):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def summarize(result) -> dict:
    eq = result.equity_curve
    closed = [t for t in result.trades if t.exit_price is not None]
    wins = sum(1 for t in closed if t.pnl > 0)
    return {
        "total_return": total_return(eq),
        "cagr": cagr(eq),
        "sharpe": sharpe(eq),
        "max_drawdown": max_drawdown(eq),
        "num_trades": len(closed),
        "win_rate": wins / len(closed) if closed else 0.0,
    }
