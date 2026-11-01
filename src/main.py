"""CLI entry point: run one or more strategies on historical data and compare results.

Example:
    python -m src.main --ticker AAPL --start 2018-01-01 --end 2024-01-01 \\
        --strategies buy_and_hold sma_cross:fast=10,slow=30 rsi_mean_reversion:period=14
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from tabulate import tabulate

from .data import load_ohlcv
from .engine import run_backtest
from .metrics import metrics_table
from .strategies import STRATEGY_REGISTRY

load_dotenv()

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def parse_strategy_spec(spec: str):
    """Parse 'name:key=val,key=val' into (name, kwargs)."""
    if ":" in spec:
        name, params_str = spec.split(":", 1)
    else:
        name, params_str = spec, ""

    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY)}")

    kwargs = {}
    for pair in filter(None, params_str.split(",")):
        key, _, val = pair.partition("=")
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass
        kwargs[key] = val

    return STRATEGY_REGISTRY[name](**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest trading strategies on historical data.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--interval", default="1d", help="Bar interval (default: 1d)")
    parser.add_argument("--cash", type=float, default=10_000, help="Initial cash")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission pct per trade")
    parser.add_argument("--slippage", type=float, default=0.0005, help="Slippage pct per trade")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["buy_and_hold"],
        help="Strategy specs, e.g. sma_cross:fast=10,slow=30",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip saving the equity curve plot")
    args = parser.parse_args()

    df = load_ohlcv(args.ticker, args.start, args.end, args.interval)
    print(f"Loaded {len(df)} bars for {args.ticker} ({args.start} to {args.end})\n")

    specs = list(args.strategies)
    if "buy_and_hold" not in specs:
        specs = ["buy_and_hold"] + specs  # always show the baseline

    results = []
    for spec in specs:
        strategy = parse_strategy_spec(spec)
        result = run_backtest(
            df,
            strategy,
            initial_cash=args.cash,
            commission_pct=args.commission,
            slippage_pct=args.slippage,
        )
        results.append(result)

    table = metrics_table(results)
    print(tabulate(table, headers="keys", tablefmt="github"))

    if not args.no_plot:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 5))
        for result in results:
            plt.plot(result.equity_curve.index, result.equity_curve.values, label=result.strategy_name)
        plt.title(f"{args.ticker} — equity curves")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.legend()
        plt.tight_layout()
        out_path = PROCESSED_DIR / f"{args.ticker.upper()}_equity.png"
        plt.savefig(out_path, dpi=150)
        print(f"\nSaved equity curve plot to {out_path}")


if __name__ == "__main__":
    main()
