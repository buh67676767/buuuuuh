# Project Instructions

## About
A backtesting lab for trying out trading strategies against real historical market
data without lookahead bias. Strategies produce a signal per bar using only data up
to that bar; the engine shifts execution to the next bar's open, so no strategy can
"see the future" even by accident. Used to compare strategies (returns, Sharpe, max
drawdown, win rate) before ever risking real money.

## Tech Stack
- Python 3.11+
- pandas / numpy for data handling
- yfinance for historical OHLCV data (cached locally in `data/raw/`)
- matplotlib for equity curve plots
- pytest for tests

## Conventions
- Use type hints on all function signatures
- Write docstrings for public functions
- Tests go in `tests/` mirroring `src/` structure
- Config values go in `config/settings.yaml`, not hardcoded
- Any new strategy: subclass `Strategy` in `src/strategies.py`, implement
  `generate_signals(df) -> pd.Series` returning values in {-1, 0, 1}, and register it
  in `STRATEGY_REGISTRY`. Signals must only use data up to the current row (no
  forward-looking `.shift(-n)`, no computing stats over the full series and
  indexing backward).
- The engine (`src/engine.py`) already shifts signals one bar forward before
  filling — don't duplicate that lag inside a strategy.

## Key Files
- `src/data.py` - historical OHLCV loading + local CSV cache
- `src/strategies.py` - strategy definitions + registry
- `src/engine.py` - no-lookahead backtest engine (fills, commission, slippage, P&L)
- `src/metrics.py` - performance stats (CAGR, Sharpe, max drawdown, win rate, etc.)
- `src/main.py` - CLI to run and compare strategies
- `config/settings.yaml` - default cash/commission/slippage

## Commands
- Run a backtest: `python -m src.main --ticker AAPL --start 2019-01-01 --end 2024-12-31 --strategies sma_cross:fast=20,slow=50 rsi_mean_reversion:period=14`
- Test: `pytest tests/`
- Lint: `ruff check src/`
