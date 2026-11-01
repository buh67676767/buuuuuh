"""Historical market data loading, with local caching."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _cache_path(ticker: str, start: str, end: str, interval: str) -> Path:
    name = f"{ticker.upper()}_{interval}_{start}_{end}.csv"
    return RAW_DIR / name


def load_ohlcv(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load historical OHLCV data for a ticker between start and end (YYYY-MM-DD).

    Caches results to data/raw/ so repeated backtests don't re-hit the network.
    Returns a DataFrame indexed by date with columns: open, high, low, close, volume.
    """
    path = _cache_path(ticker, start, end, interval)
    if use_cache and path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df

    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} between {start} and {end}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]]
    df.index.name = "date"

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df
