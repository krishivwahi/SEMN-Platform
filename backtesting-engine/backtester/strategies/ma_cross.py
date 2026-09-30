#MA crossover implementation


from __future__ import annotations

import numpy as np

from .base import Strategy


class MACrossover(Strategy):
    def __init__(self, fast: int = 20, slow: int = 50):
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.fast = fast
        self.slow = slow
        self.name = f"MA Crossover ({fast}/{slow})"

    def prepare(self, df) -> None:
        close = df["close"]
        self._fast_ma = close.rolling(self.fast).mean().to_numpy()
        self._slow_ma = close.rolling(self.slow).mean().to_numpy()

    def on_bar(self, i: int, df) -> float:
        if np.isnan(self._slow_ma[i]):
            return 0.0
        return 1.0 if self._fast_ma[i] > self._slow_ma[i] else 0.0
