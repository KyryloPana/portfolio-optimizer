from __future__ import annotations

import numpy as np


def port_return(w: np.ndarray, mu: np.ndarray) -> float:
    return float(w @ mu)


def port_var(w: np.ndarray, sigma: np.ndarray) -> float:
    return float(w.T @ sigma @ w)


def port_vol(w: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.sqrt(max(port_var(w, sigma), 1e-18)))


def sharpe(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, rf: float) -> float:
    ex = port_return(w, mu) - rf
    vol = port_vol(w, sigma)
    return float(ex / max(vol, 1e-18))


def neg_sharpe_l2(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, rf: float, l2_lambda: float) -> float:
    """
    Minimize: -Sharpe(w) + l2_lambda * sum(w^2)
    """
    if l2_lambda < 0:
        raise ValueError("l2_lambda must be >= 0")
    base = -sharpe(w, mu, sigma, rf)
    penalty = float(l2_lambda * np.sum(np.square(w)))
    return base + penalty


def min_variance(w: np.ndarray, sigma: np.ndarray) -> float:
    """
    Minimize: portfolio variance w' Σ w
    """
    return port_var(w, sigma)