"""Trading strategies.

Each strategy implements `generate_signals(df)`, returning a pandas Series
(indexed like `df`) of target positions in {-1, 0, 1} (short, flat, long).

Signals must be causal: the value at index i may only use data from rows
<= i. The backtest engine additionally shifts signals forward one bar before
execution, so a decision made on bar i's close is filled on bar i+1 -- the
strategy never needs to (and must not) see future data to generate a signal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Strategy(ABC):
    name: str = "strategy"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ...


class BuyAndHold(Strategy):
    name = "buy_and_hold"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=df.index)


class SmaCross(Strategy):
    """Long when the fast SMA is above the slow SMA, flat otherwise."""

    def __init__(self, fast: int = 20, slow: int = 50, allow_short: bool = False):
        if fast >= slow:
            raise ValueError("fast period must be less than slow period")
        self.fast = fast
        self.slow = slow
        self.allow_short = allow_short
        self.name = f"sma_cross_{fast}_{slow}"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast_ma = df["close"].rolling(self.fast).mean()
        slow_ma = df["close"].rolling(self.slow).mean()
        long_signal = fast_ma > slow_ma
        if self.allow_short:
            signal = np.where(long_signal, 1, -1)
        else:
            signal = np.where(long_signal, 1, 0)
        signal = pd.Series(signal, index=df.index)
        signal[slow_ma.isna()] = 0
        return signal


class RsiMeanReversion(Strategy):
    """Long when RSI dips below `oversold`, exit when it rises above `exit_level`."""

    def __init__(self, period: int = 14, oversold: float = 30, exit_level: float = 50):
        self.period = period
        self.oversold = oversold
        self.exit_level = exit_level
        self.name = f"rsi_mean_reversion_{period}"

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rsi = self._rsi(df["close"], self.period)
        signal = pd.Series(0, index=df.index)
        position = 0
        values = rsi.to_numpy()
        out = np.zeros(len(values), dtype=int)
        for i, rsi_val in enumerate(values):
            if position == 0 and rsi_val < self.oversold:
                position = 1
            elif position == 1 and rsi_val > self.exit_level:
                position = 0
            out[i] = position
        signal = pd.Series(out, index=df.index)
        return signal


class BollingerBreakout(Strategy):
    """Long when price closes above the upper band, flat when it falls back inside/below."""

    def __init__(self, period: int = 20, num_std: float = 2.0):
        self.period = period
        self.num_std = num_std
        self.name = f"bollinger_breakout_{period}_{num_std}"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        mid = close.rolling(self.period).mean()
        std = close.rolling(self.period).std()
        upper = mid + self.num_std * std
        lower = mid - self.num_std * std

        signal = pd.Series(0, index=df.index)
        position = 0
        out = []
        for i in range(len(df)):
            if mid.iloc[i] is np.nan or pd.isna(mid.iloc[i]):
                out.append(0)
                continue
            if position == 0 and close.iloc[i] > upper.iloc[i]:
                position = 1
            elif position == 1 and close.iloc[i] < mid.iloc[i]:
                position = 0
            out.append(position)
        return pd.Series(out, index=df.index)


STRATEGY_REGISTRY = {
    "buy_and_hold": lambda **kw: BuyAndHold(),
    "sma_cross": lambda **kw: SmaCross(**kw),
    "rsi_mean_reversion": lambda **kw: RsiMeanReversion(**kw),
    "bollinger_breakout": lambda **kw: BollingerBreakout(**kw),
}
