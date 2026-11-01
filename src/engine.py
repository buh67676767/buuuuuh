"""No-lookahead backtesting engine.

Core guarantee: a strategy's signal for bar i may only be computed from data
up to and including bar i. The engine then shifts that signal forward one
bar and fills at bar i+1's open -- so even a strategy that accidentally
computed a signal "perfectly" still trades as if it didn't know the future,
matching how a real-time system would have to act.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .strategies import Strategy


@dataclass
class Trade:
    date: pd.Timestamp
    side: str          # "buy" or "sell"
    shares: float
    price: float
    commission: float


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    position_curve: pd.Series
    trades: list[Trade]
    realized_pnls: list[float]
    initial_cash: float
    strategy_name: str


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    initial_cash: float = 10_000,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
) -> BacktestResult:
    """Run `strategy` over `df` (columns: open, high, low, close, volume)."""
    raw_signal = strategy.generate_signals(df)
    # Shift by one bar: a signal computed using bar i's close can only be
    # acted on starting at bar i+1's open.
    executed_target = raw_signal.shift(1).fillna(0)

    cash = float(initial_cash)
    shares = 0.0
    avg_cost = 0.0

    equity_curve = []
    position_curve = []
    trades: list[Trade] = []
    realized_pnls: list[float] = []

    opens = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    targets = executed_target.to_numpy()
    dates = df.index
    prev_target = 0.0

    for i in range(len(df)):
        open_price = opens[i]
        close_price = closes[i]
        target = targets[i]

        if np.isnan(open_price) or open_price <= 0:
            equity_curve.append(cash + shares * close_price)
            position_curve.append(shares)
            continue

        # Only re-size the position when the target actually changes -- holding a
        # fully-invested position shouldn't generate rebalancing trades just
        # because price moved and shares*price no longer equals equity exactly.
        if target == prev_target:
            equity_curve.append(cash + shares * close_price)
            position_curve.append(shares)
            continue
        prev_target = target

        equity_before = cash + shares * open_price
        target_shares = (target * equity_before / open_price) if equity_before > 0 else 0.0
        delta = target_shares - shares

        if abs(delta) > 1e-9:
            direction = 1 if delta > 0 else -1
            fill_price = open_price * (1 + slippage_pct * direction)
            notional = abs(delta) * fill_price
            commission = notional * commission_pct
            cash -= delta * fill_price + commission
            trades.append(
                Trade(
                    date=dates[i],
                    side="buy" if delta > 0 else "sell",
                    shares=delta,
                    price=fill_price,
                    commission=commission,
                )
            )

            same_direction = shares == 0 or np.sign(shares) == np.sign(delta)
            if same_direction:
                total_cost = avg_cost * abs(shares) + fill_price * abs(delta)
                new_shares = shares + delta
                avg_cost = total_cost / abs(new_shares) if new_shares != 0 else 0.0
            else:
                closing_amount = min(abs(delta), abs(shares))
                pnl = closing_amount * (fill_price - avg_cost) * np.sign(shares) - commission
                realized_pnls.append(pnl)
                remaining = abs(delta) - closing_amount
                new_shares = shares + delta
                if remaining > 1e-9:
                    avg_cost = fill_price  # flipped to the opposite side

            shares = new_shares

        equity_curve.append(cash + shares * close_price)
        position_curve.append(shares)

    return BacktestResult(
        equity_curve=pd.Series(equity_curve, index=dates, name="equity"),
        position_curve=pd.Series(position_curve, index=dates, name="position"),
        trades=trades,
        realized_pnls=realized_pnls,
        initial_cash=initial_cash,
        strategy_name=strategy.name,
    )
