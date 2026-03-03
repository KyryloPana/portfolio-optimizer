from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path("portfolio/cache")


@dataclass
class DataConfig:
    """
    Configuration for market data retrieval
    Attributes:
        start: Start date (YYYY-MM-DD) or None for maximum available history
        end: End date (YYYY-MM-DD) or None
        interval: Sampling frequency supported by yfinance (eg "1d")
        cache_data: If True cache downloaded prices to disk and reuse if fresh
        cache_days: Max age (in days) for cached files before re-download
    """

    start: str | None = None
    end: str | None = None
    interval: str = "1d"
    cache_data: bool = True
    cache_days: int = 3


def _is_cache_fresh(path: Path, cache_days: int) -> bool:
    """Return True if cache file exists and is newer than 'cache_days'"""
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime) < timedelta(days=cache_days)


def get_prices(tickers: list[str], cfg: DataConfig) -> pd.DataFrame:
    """
    Download (or load from cache) adjusted close prices for 'tickers'

    Returns:
        DataFrame indexed by date, with one column per ticker

    Notes:
        Uses yfinance with auto_adjust=True (splits/dividends adjusted)
        Normalizes yfinance output so caller always receives a DataFrame
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    key = f"{'_'.join(tickers)}_{cfg.start}_{cfg.end}_{cfg.interval}".replace(":", "-")
    cache_path = CACHE_DIR / f"{key}.csv"

    if cfg.cache_data and _is_cache_fresh(cache_path, cfg.cache_days):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    data = yf.download(
        tickers=tickers,
        start=cfg.start,
        end=cfg.end,
        interval=cfg.interval,
        auto_adjust=True,
        progress=False,
    )

    # yfinance returns MultiIndex columns when requesting multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = tickers

    prices = prices.dropna(how="all").sort_index()

    if cfg.cache_data:
        prices.to_csv(cache_path)

    return prices


def portfolio_return(
    prices: pd.DataFrame, weights: pd.Series | dict, V0: float = 1.0, start_date=None
) -> pd.Series:
    if isinstance(weights, dict):
        weights = pd.Series(weights)
    P0 = prices.iloc[0]
    shares = weights * V0 / P0
    portfolio_value = prices.mul(shares, axis=1).sum(axis=1)
    portfolio_return = portfolio_value.pct_change().drop(index=portfolio_value.index[0])
    portfolio_return.name = "Portfolio"
    return portfolio_return


def load_series_from_csv(
    path: Path | str, fmt: str, series_name: str = "SERIES"
) -> pd.Series:
    """
    Load a portfolio series from a CSV

    Expected CSV:
        - Either a single column of values with a date index column
        - Or multiple columns (e.g., tickers) where it will be attempted to locate a 'Portfolio' column

    Returns:
        pd.Series of returns indexed by datetime
    """
    p = Path(path).expanduser().resolve()
    print(f"DEBUG csv resolved path = {p}")

    if not p.exists():
        raise FileNotFoundError(f"CSV file not found at path: {p}")

    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"CSV file is empty: {p}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV from path: {p}") from e

    # Set datetime index if date column exists
    dt_col = None
    for c in ("date", "Date", "datetime", "Datetime", "time", "Time"):
        if c in df.columns:
            dt_col = c
            break
    if dt_col is not None:
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.dropna(subset=[dt_col]).set_index(dt_col)

    # Pick one value column
    if series_name in df.columns:
        s = df[series_name]
    else:
        value_cols = list(df.columns)
        if len(value_cols) != 1:
            raise ValueError(
                f"CSV must contain '{series_name}' or exactly 1 value column. Found: {value_cols}"
            )
        s = df[value_cols[0]]

    s = pd.to_numeric(s, errors="coerce").dropna().sort_index()

    if fmt == "prices":
        s = s.pct_change().dropna()
    elif fmt != "returns":
        raise ValueError("fmt must be 'returns' or 'prices'")

    s.name = series_name
    return s


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert price series to simple arithmetic returns (pct_change)
    """
    return prices.pct_change().dropna()
