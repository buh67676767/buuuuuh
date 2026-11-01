# Trading Lab

> Backtest trading strategies against real historical market data, with a no-lookahead engine, and get results fed back as a comparison table + equity curve plot.

## Overview

Describe a strategy idea, it gets implemented as a `Strategy` subclass, and the
engine runs it bar-by-bar over historical OHLCV data — a strategy's signal for
bar *i* may only use data up to bar *i*, and the engine fills at bar *i+1*'s
open, so results can't be inflated by accidentally peeking at the future.

Built-in strategies: `buy_and_hold`, `sma_cross`, `rsi_mean_reversion`, `bollinger_breakout`.

## Setup

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Usage

```bash
python -m src.main --ticker AAPL --start 2019-01-01 --end 2024-12-31 \
    --strategies sma_cross:fast=20,slow=50 rsi_mean_reversion:period=14 bollinger_breakout:period=20,num_std=2
```

This prints a comparison table (total return, CAGR, Sharpe, max drawdown, win
rate, profit factor) across all listed strategies plus a `buy_and_hold`
baseline, and saves an equity-curve plot to `data/processed/<TICKER>_equity.png`.

Add `--cash`, `--commission`, `--slippage` to tune starting capital and trading
costs. Historical data is cached in `data/raw/` after the first download.

## Project Structure

```
├── src/           # data loading, strategies, engine, metrics, CLI
├── tests/         # engine correctness tests (incl. a no-lookahead guard test)
├── docs/          # project documentation
├── config/        # default backtest settings
├── scripts/       # utility and automation scripts
├── data/          # cached raw OHLCV + processed outputs (plots)
├── models/        # (unused for now)
├── notebooks/     # Jupyter notebooks for exploration
└── CLAUDE.md      # AI assistant instructions for this project
```

## Status

- [x] Initial setup
- [x] Core implementation (data, engine, strategies, metrics, CLI)
- [x] Tests passing
- [ ] More strategies / asset classes (crypto, forex)
