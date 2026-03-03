from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _to_dataframe(x: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Normalize Series/DataFrame input to a DataFrame."""
    if isinstance(x, pd.Series):
        return x.to_frame(name=x.name or "strategy")
    return x


def align_returns(
    strategy: pd.Series | pd.DataFrame,
    benchmark: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Align strategy returns (one or many) with benchmark returns on common dates.

    Returns:
        (strategy_df, benchmark_series) with no missing values on the shared index.
    """
    s = _to_dataframe(strategy).copy()
    b = benchmark.copy()

    if not isinstance(b, pd.Series):
        raise TypeError(f"benchmark must be a pd.Series, got {type(b)}")
    b.name = "Benchmark"  # set Series name (column name after join)

    joined = s.join(b, how="inner").dropna()
    return joined.drop(columns=["Benchmark"]), joined["Benchmark"]


def cumulative_returns(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compound returns into an equity curve (growth of $1)."""
    return (1 + returns).cumprod()


def annualized_return(
    returns: pd.Series | pd.DataFrame, periods: int = TRADING_DAYS
) -> pd.Series:
    """Compute CAGR (annualized geometric return)."""
    r = _to_dataframe(returns)
    total = (1 + r).prod()
    years = len(r) / periods
    return (total ** (1 / years) - 1).rename("cagr")


def annualized_volatility(
    returns: pd.Series | pd.DataFrame, periods: int = TRADING_DAYS
) -> pd.Series:
    """Compute annualized volatility from periodic returns."""
    r = _to_dataframe(returns)
    return (r.std() * np.sqrt(periods)).rename("vol")


def sharpe_ratio(
    returns: pd.Series | pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods: int = TRADING_DAYS,
) -> pd.Series:
    """
    Compute annualized Sharpe ratio

    Assumes:
        - returns are arithmetic periodic returns
        - risk_free_rate is annualized (e.g., 0.05 for 5%)
    """
    r = _to_dataframe(returns)
    excess = r - (risk_free_rate / periods)

    std = excess.std()
    sr = (excess.mean() / std) * np.sqrt(periods)
    sr = sr.replace([np.inf, -np.inf], np.nan)
    return sr.rename("sharpe")


def drawdown_series(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute drawdown series from an equity curve"""
    eq = cumulative_returns(returns)
    peak = eq.cummax()
    return (eq - peak) / peak


def max_drawdown(returns: pd.Series | pd.DataFrame) -> pd.Series:
    """Compute maximum drawdown (most negative drawdown)"""
    dd = drawdown_series(returns)
    ddf = _to_dataframe(dd)
    return ddf.min().rename("max_dd")


def beta(strategy: pd.Series | pd.DataFrame, benchmark: pd.Series) -> pd.Series:
    """Compute CAPM beta vs benchmark using covariance/variance."""
    s, b = align_returns(strategy, benchmark)
    b_var = b.var()
    if not np.isfinite(b_var) or b_var == 0:
        return pd.Series(index=s.columns, data=np.nan, name="beta")

    betas = {col: s[col].cov(b) / b_var for col in s.columns}
    return pd.Series(betas, name="beta")


def alpha_annualized(
    strategy: pd.Series | pd.DataFrame,
    benchmark: pd.Series,
    periods: int = TRADING_DAYS,
) -> pd.Series:
    """
    Compute realized CAPM-style alpha, annualized:

        alpha_daily = mean(strategy - beta * benchmark)
        alpha_annual = alpha_daily * periods
    """
    s, b = align_returns(strategy, benchmark)
    betas = beta(s, b)

    alphas = {col: (s[col] - betas[col] * b).mean() * periods for col in s.columns}
    return pd.Series(alphas, name="alpha_annual")


def tracking_error_annualized(
    strategy: pd.Series | pd.DataFrame,
    benchmark: pd.Series,
    periods: int = TRADING_DAYS,
) -> pd.Series:
    """Annualized tracking error: std(strategy - benchmark) * sqrt(periods)."""
    s, b = align_returns(strategy, benchmark)
    te = {col: (s[col] - b).std() * np.sqrt(periods) for col in s.columns}
    return pd.Series(te, name="tracking_error")


def information_ratio(
    strategy: pd.Series | pd.DataFrame,
    benchmark: pd.Series,
    periods: int = TRADING_DAYS,
) -> pd.Series:
    """
    Information ratio:

        IR = (mean(strategy - benchmark) * periods) / tracking_error
    """
    s, b = align_returns(strategy, benchmark)

    ir = {}
    for col in s.columns:
        active = s[col] - b
        te = active.std() * np.sqrt(periods)
        ir[col] = (active.mean() * periods / te) if te and np.isfinite(te) else np.nan

    return pd.Series(ir, name="information_ratio")


def correlation(strategy: pd.Series | pd.DataFrame, benchmark: pd.Series) -> pd.Series:
    """Pearson correlation vs benchmark."""
    s, b = align_returns(strategy, benchmark)
    return pd.Series({col: s[col].corr(b) for col in s.columns}, name="correlation")


def benchmark_summary(
    strategy: pd.Series | pd.DataFrame,
    benchmark: pd.Series,
) -> pd.DataFrame:
    """Collect benchmark-relative statistics into a single table"""
    s, b = align_returns(strategy, benchmark)

    out = pd.DataFrame(index=s.columns)
    out["beta"] = beta(s, b)
    out["alpha_annual"] = alpha_annualized(s, b)
    out["tracking_error"] = tracking_error_annualized(s, b)
    out["information_ratio"] = information_ratio(s, b)
    out["correlation"] = correlation(s, b)
    return out
