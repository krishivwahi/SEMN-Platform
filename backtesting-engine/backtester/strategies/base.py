#abstract strategy interface

from __future__ import annotations


class Strategy:
    """Base class for signal generators.

    Contract:
      - prepare(df) runs once before the loop. Precompute indicators here,
        but only with causal transforms (rolling, expanding, shift).
      - on_bar(i, df) is called at bar i's CLOSE and returns the target
        portfolio weight in [0, 1]. It must only read values at or before
        index i. The engine fills the resulting order at bar i+1's open,
        so even a leaky indicator cannot be executed at a past price -
        but keep indicators causal anyway; the tests keep you honest.
    """

    name = "strategy"

    def prepare(self, df) -> None:
        pass

    def on_bar(self, i: int, df) -> float:
        raise NotImplementedError


class BuyAndHold(Strategy):

    name = "Buy & Hold"

    def on_bar(self, i: int, df) -> float:
        return 1.0
