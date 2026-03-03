from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from portfolio.metrics import cumulative_returns, drawdown_series

TRADING_DAYS = 252


def plot_equity_curve(returns: pd.Series, title: str = "Equity Curve"):
    """Plot compounded growth of $1 from periodic returns."""
    equity = cumulative_returns(returns)

    fig, ax = plt.subplots(figsize=(10, 5))
    equity.plot(ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.grid(True)

    return fig


def plot_drawdowns(returns: pd.Series, title: str = "Drawdowns"):
    """Plot drawdown series (peak-to-trough declines) from periodic returns."""
    dd = drawdown_series(returns)

    fig, ax = plt.subplots(figsize=(10, 5))
    dd.plot(ax=ax, color="red")
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.grid(True)

    return fig


def plot_rolling_volatility(
    returns: pd.Series,
    window: int = 63,
    title: str = "Rolling Volatility",
):
    """
    Plot annualized rolling volatility

    Args:
        window: Rolling window in trading days (63 ~ 3 months)
    """
    rolling_vol = returns.rolling(window).std() * np.sqrt(TRADING_DAYS)

    fig, ax = plt.subplots(figsize=(10, 5))
    rolling_vol.plot(ax=ax, color="orange")
    ax.set_title(title)
    ax.set_ylabel("Annualized Volatility")
    ax.grid(True)

    return fig


def plot_rolling_sharpe(
    returns: pd.Series,
    window: int = 63,
    risk_free_rate: float = 0.0,
    title: str = "Rolling Sharpe Ratio",
):
    """
    Plot annualized rolling Sharpe ratio

    Assumes risk_free_rate is annualized (e.g., 0.05 for 5%)
    """
    excess = returns - (risk_free_rate / TRADING_DAYS)
    rolling_sharpe = (
        excess.rolling(window).mean() / excess.rolling(window).std()
    ) * np.sqrt(TRADING_DAYS)

    fig, ax = plt.subplots(figsize=(10, 5))
    rolling_sharpe.plot(ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Sharpe")
    ax.grid(True)

    return fig
