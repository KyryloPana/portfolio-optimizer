# portfolio/data.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf

CACHE_DIR = Path("portfolio/cache")


@dataclass(frozen=True)
class DataConfig:
    """
    Configuration for market data retrieval.

    Attributes:
        start: Start date (YYYY-MM-DD) or None for maximum available history
        end: End date (YYYY-MM-DD) or None
        interval: Sampling frequency supported by yfinance (e.g. "1d")
        auto_adjust: If True, yfinance returns split/dividend-adjusted prices
        price_field: Which OHLC field to extract ("Close" recommended with auto_adjust=True)
        cache_data: If True, cache downloaded prices to disk and reuse if fresh
        cache_days: Max age (in days) for cached files before re-download
        ffill: Forward-fill missing prices before returning
    """

    start: str | None = None
    end: str | None = None
    interval: str = "1d"
    auto_adjust: bool = True
    price_field: Literal["Close", "Adj Close"] = "Close"
    cache_data: bool = True
    cache_days: int = 3
    ffill: bool = True


def _is_cache_fresh(path: Path, cache_days: int) -> bool:
    """Return True if cache file exists and is newer than 'cache_days'."""
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime) < timedelta(days=cache_days)


def _cache_key(tickers: list[str], cfg: DataConfig) -> str:
    # Stable ordering prevents accidental cache misses from ticker order changes.
    tickers_key = "_".join(sorted([t.strip().upper() for t in tickers]))
    key = f"{tickers_key}_{cfg.start}_{cfg.end}_{cfg.interval}_{cfg.auto_adjust}_{cfg.price_field}"
    return key.replace(":", "-").replace("/", "-")


def _extract_price_field(data: pd.DataFrame, tickers: list[str], price_field: str) -> pd.DataFrame:
    """
    Normalize yfinance output so caller always receives a DataFrame indexed by date,
    with one column per ticker containing the requested price field.
    """
    if data.empty:
        raise ValueError("No data returned from yfinance. Check tickers and date range.")

    # Multiple tickers -> MultiIndex columns: (field, ticker)
    if isinstance(data.columns, pd.MultiIndex):
        # Prefer requested field if present
        if price_field in data.columns.levels[0]:
            prices = data[price_field].copy()
        else:
            # Fall back to "Close" if present
            if "Close" in data.columns.levels[0]:
                prices = data["Close"].copy()
            else:
                # Last resort: take first field
                first_field = data.columns.levels[0][0]
                prices = data[first_field].copy()

        # Ensure all tickers present
        missing = set([t.upper() for t in tickers]) - set([c.upper() for c in prices.columns])
        if missing:
            raise ValueError(f"Missing tickers in downloaded data: {sorted(missing)}")

        # Preserve original ticker case/order if possible
        prices = prices[[t for t in tickers if t in prices.columns]]
        return prices

    # Single ticker -> columns are fields (Open/High/Low/Close/Volume)
    # Extract the requested field; fallback to Close.
    field = price_field if price_field in data.columns else "Close"
    if field not in data.columns:
        raise ValueError(f"Could not find price field '{price_field}' (or 'Close') in yfinance output.")
    prices = data[[field]].copy()
    prices.columns = [tickers[0]]
    return prices


def get_prices(tickers: list[str], cfg: DataConfig) -> pd.DataFrame:
    """
    Download (or load from cache) price series for 'tickers'.

    Returns:
        DataFrame indexed by date, with one column per ticker.

    Notes:
        - Uses yfinance download.
        - Normalizes output to a simple DataFrame regardless of 1 vs many tickers.
        - With auto_adjust=True, "Close" is adjusted for splits/dividends.
    """
    if not tickers:
        raise ValueError("tickers must be a non-empty list.")
    tickers = [t.strip().upper() for t in tickers]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{_cache_key(tickers, cfg)}.csv"

    if cfg.cache_data and _is_cache_fresh(cache_path, cfg.cache_days):
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        # ensure proper sort + dtype
        prices = prices.sort_index()
        return prices

    data = yf.download(
        tickers=tickers,
        start=cfg.start,
        end=cfg.end,
        interval=cfg.interval,
        auto_adjust=cfg.auto_adjust,
        progress=False,
        actions=False,
        threads=True,
    )

    prices = _extract_price_field(data, tickers, cfg.price_field)
    prices = prices.dropna(how="all").sort_index()

    if cfg.ffill:
        prices = prices.ffill()

    # Drop columns that are entirely missing after ffill
    prices = prices.dropna(axis=1, how="all")

    if cfg.cache_data:
        prices.to_csv(cache_path)

    return prices


# ----------------------------
# Returns helpers (for optimizer)
# ----------------------------

def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple arithmetic returns (pct_change)."""
    return prices.pct_change().dropna(how="any")


def prices_to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns: ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1)).dropna(how="any")


def compute_returns(prices: pd.DataFrame, method: Literal["simple", "log"] = "log") -> pd.DataFrame:
    """
    Convenience wrapper for return construction.
    Default is log returns (preferred for many optimization pipelines).
    """
    if method == "log":
        return prices_to_log_returns(prices)
    if method == "simple":
        return prices_to_returns(prices)
    raise ValueError("method must be 'simple' or 'log'")


def _weights_to_array(
    tickers: list[str],
    weights: pd.Series | dict | np.ndarray,
) -> np.ndarray:
    """
    Convert weights into an array aligned to tickers.
    Enforces that all tickers exist and weights are numeric.
    """
    if isinstance(weights, np.ndarray):
        w = weights.astype(float).copy()
        if w.shape[0] != len(tickers):
            raise ValueError("weights ndarray length must match tickers length.")
        return w

    if isinstance(weights, dict):
        weights = pd.Series(weights, dtype=float)

    if isinstance(weights, pd.Series):
        # Align by ticker name; require full coverage
        weights = weights.astype(float)
        missing = set(tickers) - set(weights.index.astype(str).str.upper())
        if missing:
            raise ValueError(f"weights missing tickers: {sorted(missing)}")

        # Normalize index casing
        w_map = {str(k).upper(): float(v) for k, v in weights.items()}
        w = np.array([w_map[t] for t in tickers], dtype=float)
        return w

    raise TypeError("weights must be pd.Series, dict, or np.ndarray.")


def portfolio_returns_from_returns(
    asset_returns: pd.DataFrame,
    weights: pd.Series | dict | np.ndarray,
    rebalance: bool = True,
) -> pd.Series:
    """
    Portfolio return series from asset return matrix.

    If rebalance=True (default):
        r_p[t] = sum_i w_i * r_i[t]  (constant weights, daily rebalanced)
        This is typically what you want for optimizer backtests and reporting.

    If rebalance=False:
        Approximates buy-and-hold by converting returns to a price index and using fixed shares.
        (Kept mainly for compatibility; prefer explicit buy-and-hold on prices.)
    """
    tickers = [str(c).upper() for c in asset_returns.columns]
    R = asset_returns.copy()
    R.columns = tickers
    R = R.dropna(how="any")

    w = _weights_to_array(tickers, weights)
    s = float(np.sum(w))
    if s == 0:
        raise ValueError("weights sum to 0.")
    w = w / s

    if rebalance:
        rp = R.to_numpy(dtype=float) @ w
        out = pd.Series(rp, index=R.index, name="Portfolio")
        return out

    # buy-and-hold approximation: build synthetic price index from returns
    # (works best with simple returns; if you used log returns, convert first)
    if (R.min().min() < -1.0) or (R.max().max() > 5.0):
        raise ValueError("rebalance=False expects simple returns in reasonable range.")

    price_index = (1.0 + R).cumprod()
    return portfolio_return_buy_and_hold(price_index, dict(zip(tickers, w)))


def portfolio_return_buy_and_hold(
    prices: pd.DataFrame,
    weights: pd.Series | dict,
    V0: float = 1.0,
) -> pd.Series:
    """
    Buy-and-hold portfolio returns computed from prices by fixing initial shares.

    Note:
        - This is distinct from constant-weight (daily rebalanced) returns.
        - Useful to compare vs rebalanced portfolios; not used for the optimizer objective itself.
    """
    tickers = [str(c).upper() for c in prices.columns]
    P = prices.copy()
    P.columns = tickers
    P = P.dropna(how="any")

    if isinstance(weights, dict):
        weights = pd.Series(weights, dtype=float)
    weights.index = weights.index.astype(str).str.upper()

    missing = set(tickers) - set(weights.index)
    if missing:
        raise ValueError(f"weights missing tickers: {sorted(missing)}")

    w = weights.reindex(tickers).astype(float)
    w = w / float(w.sum())

    P0 = P.iloc[0]
    shares = (w * V0 / P0).astype(float)
    V = P.mul(shares, axis=1).sum(axis=1)
    r = V.pct_change().drop(index=V.index[0])
    r.name = "Portfolio"
    return r


def load_series_from_csv(path: Path | str, fmt: str, series_name: str = "SERIES") -> pd.Series:
    """
    Load a time series from CSV.

    Expected CSV:
        - Either a single column of values with a date index column
        - Or multiple columns where it will locate `series_name`

    Args:
        fmt: "prices" -> convert to returns; "returns" -> keep as returns
        series_name: which column to extract if multiple columns exist

    Returns:
        pd.Series indexed by datetime
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found at path: {p}")

    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"CSV file is empty: {p}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV from path: {p}") from e

    # Set datetime index if a likely date column exists
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