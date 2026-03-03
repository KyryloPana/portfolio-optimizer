# portfolio/estimation.py
"""
Estimation layer for Modern Portfolio Theory optimization.

Provides:
- Expected returns (mu) estimators:
    * historical mean
    * EWMA mean
- Covariance (Sigma) estimators:
    * sample covariance
    * EWMA covariance
    * shrinkage covariance (Ledoit-Wolf if sklearn available; else simple shrink-to-diagonal)
- PSD enforcement (nearest PSD via eigenvalue clipping)

Design goals:
- Deterministic, explicit assumptions
- Numerically stable (PSD covariance, ridge epsilon)
- Compatible with min-variance and max-Sharpe optimizers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS_DEFAULT = 252

MeanMethod = Literal["historical", "ewma"]
CovMethod = Literal["sample", "ewma", "shrink"]


@dataclass(frozen=True)
class EstimationConfig:
    annualization: int = TRADING_DAYS_DEFAULT

    # EWMA settings
    ewma_lambda: float = 0.94  # RiskMetrics-style default; 0.97-0.99 for slower decay

    # Numerical stability / PSD
    ridge_eps: float = 1e-10
    psd_eig_floor: float = 1e-12

    # Shrinkage fallback (when sklearn not available or user wants no dependency)
    # alpha in [0,1]: 0 => pure sample; 1 => pure target
    shrink_alpha_fallback: Optional[float] = None  # if None: auto heuristic
    shrink_target: Literal["diagonal", "identity"] = "diagonal"

    winsorize: bool = False
    winsor_q: float = 0.01  # 1% tails


def _as_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("Expected a 2D array.")
    return x


def _validate_returns(returns: pd.DataFrame, cfg: EstimationConfig) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame with columns = assets.")
    if returns.shape[1] < 2:
        raise ValueError("returns must contain at least 2 assets (2 columns).")

    r = returns.dropna(how="any")
    if len(r) < 20:
        raise ValueError("Too few return observations after dropping NaNs.")
    if not np.isfinite(r.to_numpy()).all():
        raise ValueError("returns contains non-finite values.")

    if cfg.winsorize:
        r = winsorize_returns(r, cfg.winsor_q)

    return r

def winsorize_returns(r: pd.DataFrame, q: float) -> pd.DataFrame:
    """
    Winsorize per-asset returns by clipping at [q, 1-q] quantiles.
    Example: q=0.01 clips 1% tails.
    """
    if not (0.0 <= q < 0.5):
        raise ValueError("winsor_q must be in [0, 0.5).")

    lower = r.quantile(q, axis=0, numeric_only=True)
    upper = r.quantile(1.0 - q, axis=0, numeric_only=True)

    # Per-column clipping
    r_clip = r.clip(lower=lower, upper=upper, axis=1)
    return r_clip

def winsorization_stats(r: pd.DataFrame, q: float) -> pd.DataFrame:
    """
    Returns per-asset stats: how many points were clipped at the lower/upper bounds.
    """
    if not (0.0 <= q < 0.5):
        raise ValueError("winsor_q must be in [0, 0.5).")

    lower = r.quantile(q, axis=0)
    upper = r.quantile(1.0 - q, axis=0)

    below = (r.lt(lower, axis=1)).sum(axis=0)
    above = (r.gt(upper, axis=1)).sum(axis=0)
    total = pd.Series(len(r), index=r.columns)

    out = pd.DataFrame({
        "n_obs": total,
        "n_clipped_low": below,
        "n_clipped_high": above,
        "pct_clipped": (below + above) / total.replace(0, pd.NA),
        "q_low": lower,
        "q_high": upper,
    })
    return out

def annualize_mean(mu_daily: np.ndarray, annualization: int) -> np.ndarray:
    return mu_daily * float(annualization)


def annualize_cov(cov_daily: np.ndarray, annualization: int) -> np.ndarray:
    return cov_daily * float(annualization)


def ewma_weights(n: int, lam: float) -> np.ndarray:
    """
    EWMA weights that sum to 1, with higher weight on most recent observation.
    w_t ∝ (1-lam) * lam^(n-1-t)
    """
    if not (0.0 < lam < 1.0):
        raise ValueError("ewma_lambda must be in (0,1).")
    # t=0 oldest ... t=n-1 newest
    exponents = np.arange(n - 1, -1, -1, dtype=float)
    w = (1.0 - lam) * (lam ** exponents)
    w /= w.sum()
    return w


def ensure_psd(mat: np.ndarray, eig_floor: float = 1e-12, ridge_eps: float = 1e-10) -> np.ndarray:
    """
    Force matrix to be symmetric positive semidefinite (PSD) via eigenvalue clipping.
    Adds a tiny ridge to help conditioning.
    """
    A = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, eig_floor)
    psd = vecs @ np.diag(vals) @ vecs.T
    psd = 0.5 * (psd + psd.T)
    psd = psd + np.eye(psd.shape[0]) * ridge_eps
    return psd


# ------------------------
# Mean estimation
# ------------------------

def estimate_mean(
    returns: pd.DataFrame,
    method: MeanMethod = "historical",
    cfg: EstimationConfig = EstimationConfig(),
) -> pd.Series:
    """
    Returns annualized expected returns (mu) as a Series indexed by asset tickers.
    """
    r = _validate_returns(returns, cfg)

    if method == "historical":
        mu_d = r.mean(axis=0).to_numpy(dtype=float)
        mu_a = annualize_mean(mu_d, cfg.annualization)
        return pd.Series(mu_a, index=r.columns, name="mu")

    if method == "ewma":
        w = ewma_weights(len(r), cfg.ewma_lambda)  # length T
        # weighted mean per asset: sum_t w_t * r_ti
        mu_d = (r.to_numpy(dtype=float) * w[:, None]).sum(axis=0)
        mu_a = annualize_mean(mu_d, cfg.annualization)
        return pd.Series(mu_a, index=r.columns, name="mu")

    raise ValueError("method must be one of: 'historical', 'ewma'")


# ------------------------
# Covariance estimation
# ------------------------

def _sample_cov(returns: pd.DataFrame) -> np.ndarray:
    X = returns.to_numpy(dtype=float)
    cov = np.cov(X, rowvar=False, ddof=1)
    return _as_2d(cov)


def _ewma_cov(returns: pd.DataFrame, lam: float) -> np.ndarray:
    """
    EWMA covariance with weights normalized to sum to 1.
    """
    X = returns.to_numpy(dtype=float)
    w = ewma_weights(len(returns), lam)
    # Weighted mean
    mu = (X * w[:, None]).sum(axis=0, keepdims=True)
    Xc = X - mu
    cov = (Xc * w[:, None]).T @ Xc
    return _as_2d(cov)


def _shrink_to_target(sample_cov: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0,1].")
    return (1.0 - alpha) * sample_cov + alpha * target


def _shrink_target_matrix(sample_cov: np.ndarray, target_kind: str) -> np.ndarray:
    if target_kind == "diagonal":
        return np.diag(np.diag(sample_cov))
    if target_kind == "identity":
        avg_var = float(np.mean(np.diag(sample_cov)))
        return np.eye(sample_cov.shape[0]) * avg_var
    raise ValueError("shrink_target must be 'diagonal' or 'identity'.")


def _auto_alpha_heuristic(returns: pd.DataFrame) -> float:
    """
    Simple heuristic: more shrinkage when sample is noisy (small T vs N).
    Not Ledoit-Wolf; just a pragmatic fallback.
    """
    T, N = returns.shape
    # If T is close to N, sample cov is unstable => alpha near 0.5-0.8
    ratio = N / max(T, 1)
    alpha = 0.1 + 0.9 * min(1.0, ratio * 3.0)  # ramps quickly as N/T increases
    return float(np.clip(alpha, 0.1, 0.9))


def estimate_cov(
    returns: pd.DataFrame,
    method: CovMethod = "shrink",
    cfg: EstimationConfig = EstimationConfig(),
) -> pd.DataFrame:
    """
    Returns annualized covariance matrix (Sigma) as a DataFrame (assets x assets).
    Always PSD-enforced.
    """
    r = _validate_returns(returns, cfg)

    if method == "sample":
        cov_d = _sample_cov(r)
        cov_a = annualize_cov(cov_d, cfg.annualization)
        cov_psd = ensure_psd(cov_a, cfg.psd_eig_floor, cfg.ridge_eps)
        return pd.DataFrame(cov_psd, index=r.columns, columns=r.columns)

    if method == "ewma":
        cov_d = _ewma_cov(r, cfg.ewma_lambda)
        cov_a = annualize_cov(cov_d, cfg.annualization)
        cov_psd = ensure_psd(cov_a, cfg.psd_eig_floor, cfg.ridge_eps)
        return pd.DataFrame(cov_psd, index=r.columns, columns=r.columns)

    if method == "shrink":
        # Preferred: Ledoit-Wolf shrinkage if sklearn is installed.
        X = r.to_numpy(dtype=float)
        cov_d: Optional[np.ndarray] = None

        try:
            from sklearn.covariance import LedoitWolf  # type: ignore
            lw = LedoitWolf().fit(X)
            cov_d = lw.covariance_
        except Exception:
            cov_d = None

        if cov_d is None:
            sample = _sample_cov(r)
            target = _shrink_target_matrix(sample, cfg.shrink_target)
            alpha = cfg.shrink_alpha_fallback
            if alpha is None:
                alpha = _auto_alpha_heuristic(r)
            cov_d = _shrink_to_target(sample, target, float(alpha))

        cov_a = annualize_cov(cov_d, cfg.annualization)
        cov_psd = ensure_psd(cov_a, cfg.psd_eig_floor, cfg.ridge_eps)
        return pd.DataFrame(cov_psd, index=r.columns, columns=r.columns)

    raise ValueError("method must be one of: 'sample', 'ewma', 'shrink'")


# ------------------------
# Convenience: estimate both
# ------------------------

def estimate_mu_sigma(
    returns: pd.DataFrame,
    mean_method: MeanMethod = "historical",
    cov_method: CovMethod = "shrink",
    cfg: EstimationConfig = EstimationConfig(),
) -> Tuple[pd.Series, pd.DataFrame]:
    mu = estimate_mean(returns, method=mean_method, cfg=cfg)
    sigma = estimate_cov(returns, method=cov_method, cfg=cfg)
    return mu, sigma