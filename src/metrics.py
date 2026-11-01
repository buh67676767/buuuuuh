"""Performance metrics computed from a BacktestResult."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import BacktestResult

TRADING_DAYS_PER_YEAR = 252


def compute_metrics(result: BacktestResult, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    equity = result.equity_curve
    returns = equity.pct_change().dropna()

    total_return = equity.iloc[-1] / result.initial_cash - 1
    n_periods = len(equity)
    years = n_periods / periods_per_year if n_periods > 0 else 0
    cagr = (equity.iloc[-1] / result.initial_cash) ** (1 / years) - 1 if years > 0 else np.nan

    vol = returns.std() * np.sqrt(periods_per_year) if len(returns) > 1 else np.nan
    sharpe = (returns.mean() * periods_per_year) / vol if vol and vol > 0 else np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = drawdown.min()

    wins = [p for p in result.realized_pnls if p > 0]
    win_rate = len(wins) / len(result.realized_pnls) if result.realized_pnls else np.nan

    gross_profit = sum(p for p in result.realized_pnls if p > 0)
    gross_loss = -sum(p for p in result.realized_pnls if p < 0)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    return {
        "strategy": result.strategy_name,
        "final_equity": round(equity.iloc[-1], 2),
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2) if not np.isnan(cagr) else np.nan,
        "annualized_vol_pct": round(vol * 100, 2) if vol and not np.isnan(vol) else np.nan,
        "sharpe": round(sharpe, 2) if sharpe and not np.isnan(sharpe) else np.nan,
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "num_round_trips": len(result.realized_pnls),
        "win_rate_pct": round(win_rate * 100, 2) if not np.isnan(win_rate) else np.nan,
        "profit_factor": round(profit_factor, 2) if not np.isnan(profit_factor) else np.nan,
        "num_transactions": len(result.trades),
    }


def metrics_table(results: list[BacktestResult], periods_per_year: int = TRADING_DAYS_PER_YEAR) -> pd.DataFrame:
    rows = [compute_metrics(r, periods_per_year) for r in results]
    return pd.DataFrame(rows).set_index("strategy")


def rolling_window_returns(result: BacktestResult, window_days: int = 126) -> dict:
    """How much the strategy's return varies depending on *when* you started.

    Computes the return over every rolling `window_days`-bar window (default:
    ~6 trading months) and summarizes the spread -- this is "risk" in the
    sense of result variance across start dates, distinct from day-to-day
    volatility (which `compute_metrics` already reports).
    """
    equity = result.equity_curve
    if len(equity) < window_days + 1:
        return {
            "strategy": result.strategy_name,
            "window_days": window_days,
            "n_windows": 0,
            "mean_pct": np.nan,
            "std_pct": np.nan,
            "best_pct": np.nan,
            "worst_pct": np.nan,
        }

    rolling_returns = (equity / equity.shift(window_days) - 1).dropna()
    return {
        "strategy": result.strategy_name,
        "window_days": window_days,
        "n_windows": len(rolling_returns),
        "mean_pct": round(rolling_returns.mean() * 100, 2),
        "std_pct": round(rolling_returns.std() * 100, 2),
        "best_pct": round(rolling_returns.max() * 100, 2),
        "worst_pct": round(rolling_returns.min() * 100, 2),
    }


def rolling_window_table(results: list[BacktestResult], window_days: int = 126) -> pd.DataFrame:
    rows = [rolling_window_returns(r, window_days) for r in results]
    return pd.DataFrame(rows).set_index("strategy")
