#Cash and position tracking for a single instrument, with trade bookkeeping.

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Trade:


    entry_date: object
    entry_price: float
    shares: float
    fees: float = 0.0
    exit_date: object = None
    exit_price: float = None

    @property
    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.shares - self.fees


@dataclass
class Portfolio:
    cash: float
    shares: float = 0.0
    trades: list = field(default_factory=list)
    _open_trade: Trade = None

    def equity(self, price: float) -> float:
        return self.cash + self.shares * price

    def execute(self, date, target_shares: float, fill_price: float, fee: float) -> None:

        delta = target_shares - self.shares
        if abs(delta) < 1e-12:
            return
        self.cash -= delta * fill_price
        self.cash -= fee

        if self.shares == 0.0 and target_shares > 0.0:
            self._open_trade = Trade(
                entry_date=date, entry_price=fill_price, shares=target_shares, fees=fee
            )
        elif target_shares == 0.0 and self._open_trade is not None:
            t = self._open_trade
            t.exit_date = date
            t.exit_price = fill_price
            t.fees += fee
            self.trades.append(t)
            self._open_trade = None
        elif self._open_trade is not None:
            self._open_trade.shares = target_shares
            self._open_trade.fees += fee

        self.shares = target_shares
