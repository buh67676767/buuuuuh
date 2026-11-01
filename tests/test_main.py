"""Tests for the backtesting engine and strategies."""

import numpy as np
import pandas as pd
import pytest

from src.engine import run_backtest
from src.strategies import BuyAndHold, SmaCross


def make_price_series(prices):
    idx = pd.date_range("2020-01-01", periods=len(prices), freq="D")
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [1000] * len(prices),
        },
        index=idx,
    )
    return df


def test_buy_and_hold_matches_price_return():
    prices = [100, 110, 90, 120, 150]
    df = make_price_series(prices)
    result = run_backtest(df, BuyAndHold(), initial_cash=10_000, commission_pct=0, slippage_pct=0)
    # Entry is delayed one bar (signal shift), so equity tracks price from bar 1's open onward.
    expected_final = 10_000 * (prices[-1] / prices[1])
    assert result.equity_curve.iloc[-1] == pytest.approx(expected_final, rel=1e-6)


def test_signal_is_not_lookahead():
    """A signal flagging the LAST bar's local max must not affect that bar's own return."""
    prices = [100, 100, 100, 200, 100]  # spike on bar 3
    df = make_price_series(prices)

    class PerfectForesight:
        name = "perfect_foresight"

        def generate_signals(self, df):
            # "Knows" the spike is coming and would love to buy right before it.
            return pd.Series([0, 0, 1, 0, 0], index=df.index)

    result = run_backtest(df, PerfectForesight(), initial_cash=10_000, commission_pct=0, slippage_pct=0)
    # Because signals are shifted one bar before execution, the entry fills at bar 3's
    # open (price 200) -- it must NOT capture the 100->200 jump.
    assert result.equity_curve.iloc[-1] == pytest.approx(10_000 * (100 / 200), rel=1e-6)


def test_sma_cross_flat_before_warmup():
    prices = list(np.linspace(100, 120, 10))
    df = make_price_series(prices)
    strat = SmaCross(fast=3, slow=5)
    signals = strat.generate_signals(df)
    assert (signals.iloc[:4] == 0).all()


def test_commission_reduces_equity():
    prices = [100, 110, 120]
    df = make_price_series(prices)
    no_fee = run_backtest(df, BuyAndHold(), initial_cash=10_000, commission_pct=0, slippage_pct=0)
    with_fee = run_backtest(df, BuyAndHold(), initial_cash=10_000, commission_pct=0.01, slippage_pct=0)
    assert with_fee.equity_curve.iloc[-1] < no_fee.equity_curve.iloc[-1]
