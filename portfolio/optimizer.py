# portfolio/optimizer.py
"""
Optimization-as-a-generator (not a single point solution).

Produces:
- best Sharpe portfolio (max return per unit risk)
- minimum variance portfolio
- candidate set from multi-start optimization
- ranked candidates table (Sharpe prioritized)
- Pareto-efficient subset (risk/return frontier approximation)

Dependencies:
  numpy, pandas, scipy

Designed to plug into:
- portfolio/estimation.py  -> provides mu (annual) and sigma (annual, PSD)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from portfolio.constraints import sum_to_one_constraint, hhi_max_constraint

Objective = Literal["max_sharpe", "min_var"]


@dataclass(frozen=True)
class OptimizerConfig:
    # constraints
    long_only: bool = True
    min_weight: float = 0.0
    max_weight: float = 1.0

    # search
    n_starts: int = 300
    random_seed: int = 42
    round_decimals: int = 4

    # solver
    method: str = "SLSQP"
    maxiter: int = 2000

    # selection
    top_n: int = 25
    # anti-corner constraint
    hhi_max: float | None = None

    l2_lambda: float = 0.0


# -----------------------
# Core portfolio metrics
# -----------------------

def port_return(w: np.ndarray, mu: np.ndarray) -> float:
    return float(w @ mu)


def port_var(w: np.ndarray, sigma: np.ndarray) -> float:
    return float(w.T @ sigma @ w)


def port_vol(w: np.ndarray, sigma: np.ndarray) -> float:
    v = port_var(w, sigma)
    return float(np.sqrt(max(v, 1e-18)))


def sharpe(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, rf: float) -> float:
    ex = port_return(w, mu) - rf
    vol = port_vol(w, sigma)
    return float(ex / max(vol, 1e-18))


def hhi(w: np.ndarray) -> float:
    # concentration proxy: sum of squared weights
    return float(np.sum(np.square(w)))


# -----------------------
# Constraints + bounds
# -----------------------

def make_bounds(cfg: OptimizerConfig, n: int) -> Tuple[Tuple[float, float], ...]:
    if cfg.long_only:
        lo, hi = cfg.min_weight, cfg.max_weight
    else:
        # if you allow shorting, you should decide symmetric bounds explicitly.
        # This default keeps behavior simple; adjust as needed.
        lo, hi = -cfg.max_weight, cfg.max_weight
    return tuple((lo, hi) for _ in range(n))


def sum_to_one_constraint() -> Dict:
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def _clean_weights(w: np.ndarray, bounds: Tuple[Tuple[float, float], ...]) -> np.ndarray:
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    w = np.clip(w, lo, hi)
    s = float(np.sum(w))
    if s != 0:
        w = w / s
    return w


# -----------------------
# Objectives for optimizer
# -----------------------

def obj_neg_sharpe(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, rf: float) -> float:
    return -sharpe(w, mu, sigma, rf)


def obj_min_var(w: np.ndarray, sigma: np.ndarray) -> float:
    return port_var(w, sigma)


# -----------------------
# Multi-start generation
# -----------------------

def dirichlet_starts(rng: np.random.Generator, n_assets: int, n_starts: int) -> np.ndarray:
    """
    Generates long-only starting points on the simplex (sum=1, w>=0).
    If you use min_weight > 0, the solver will adjust anyway; starts remain feasible for long_only.
    """
    return rng.dirichlet(alpha=np.ones(n_assets), size=n_starts)


def unique_weights(weights: List[np.ndarray], decimals: int = 4) -> List[np.ndarray]:
    seen = set()
    out: List[np.ndarray] = []
    for w in weights:
        key = tuple(np.round(w, decimals))
        if key not in seen:
            seen.add(key)
            out.append(w)
    return out


# -----------------------
# Solving
# -----------------------

def solve(
    x0: np.ndarray,
    objective: Objective,
    mu: np.ndarray,
    sigma: np.ndarray,
    rf: float,
    bounds: Tuple[Tuple[float, float], ...],
    constraints: List[Dict],
    cfg: OptimizerConfig,
) -> Tuple[np.ndarray, bool]:
    from portfolio.objectives import neg_sharpe_l2, min_variance  # or however your min-var is defined

    if objective == "max_sharpe":
        fun = lambda w: neg_sharpe_l2(w, mu, sigma, rf, cfg.l2_lambda)
    elif objective == "min_var":
        fun = lambda w: min_variance(w, sigma)
    else:
        raise ValueError("objective must be 'max_sharpe' or 'min_var'")

    res = minimize(
        fun,
        x0=x0,
        method=cfg.method,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": cfg.maxiter, "disp": False},
    )
    if not res.success or res.x is None:
        return _clean_weights(np.asarray(x0, dtype=float), bounds), False

    w = _clean_weights(np.asarray(res.x, dtype=float), bounds)
    return w, True


# -----------------------
# Ranking + Pareto frontier
# -----------------------

def build_candidates_table(
    weights: List[np.ndarray],
    tickers: List[str],
    mu: np.ndarray,
    sigma: np.ndarray,
    rf: float,
) -> pd.DataFrame:
    rows = []
    for w in weights:
        r = port_return(w, mu)
        v = port_vol(w, sigma)
        s = (r - rf) / max(v, 1e-18)
        rows.append(
            {
                "exp_return": r,
                "vol": v,
                "sharpe": s,
                "hhi": hhi(w),
                "max_weight": float(np.max(w)),
                "min_weight": float(np.min(w)),
                **{f"w_{t}": float(w[i]) for i, t in enumerate(tickers)},
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)
    return df


def pareto_efficient_set(df: pd.DataFrame, ret_col: str = "exp_return", risk_col: str = "vol") -> pd.DataFrame:
    """
    Keep points that are not dominated (higher return AND lower/equal risk).
    Simple O(n^2) filter: OK for a few thousand candidates.
    """
    pts = df[[ret_col, risk_col]].to_numpy()
    n = len(df)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated_by = (pts[:, 0] >= pts[i, 0]) & (pts[:, 1] <= pts[i, 1]) & (
            (pts[:, 0] > pts[i, 0]) | (pts[:, 1] < pts[i, 1])
        )
        if np.any(dominated_by):
            keep[i] = False
    return df.loc[keep].sort_values(risk_col).reset_index(drop=True)


# -----------------------
# Public API: generate solutions
# -----------------------

def generate_portfolios(
    tickers: List[str],
    mu: pd.Series | np.ndarray,
    sigma: pd.DataFrame | np.ndarray,
    rf: float,
    cfg: OptimizerConfig = OptimizerConfig(),
    baseline_weights: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Main entry point.

    Inputs:
      - tickers: list of assets (order must match mu/sigma)
      - mu: annual expected returns (Series or ndarray)
      - sigma: annual covariance matrix (DataFrame or ndarray)
      - rf: annual risk-free rate (e.g., 0.04)
      - cfg: optimization settings
      - baseline_weights: optional starting weights to include

    Outputs dict:
      {
        "best_sharpe": np.ndarray,
        "min_var": np.ndarray,
        "candidates": pd.DataFrame,       # ranked by Sharpe
        "efficient_set": pd.DataFrame,    # pareto-efficient subset
      }
    """
    if isinstance(mu, pd.Series):
        mu_vec = mu.loc[tickers].to_numpy(dtype=float)
    else:
        mu_vec = np.asarray(mu, dtype=float)

    if isinstance(sigma, pd.DataFrame):
        sig = sigma.loc[tickers, tickers].to_numpy(dtype=float)
    else:
        sig = np.asarray(sigma, dtype=float)

    n = len(tickers)
    if mu_vec.shape[0] != n or sig.shape != (n, n):
        raise ValueError("Dimensions mismatch: ensure tickers align with mu and sigma.")

    from portfolio.constraints import sum_to_one_constraint, hhi_max_constraint

    bounds = make_bounds(cfg, n)
    constraints = [sum_to_one_constraint()]

    if cfg.hhi_max is not None:
        constraints.append(hhi_max_constraint(cfg.hhi_max))

    rng = np.random.default_rng(cfg.random_seed)
    starts = dirichlet_starts(rng, n, cfg.n_starts)

    candidates: List[np.ndarray] = []

    # Include baseline if provided; else equal weight baseline
    if baseline_weights is None:
        w0 = np.ones(n) / n
    else:
        w0 = np.asarray(baseline_weights, dtype=float)
        if w0.shape[0] != n:
            raise ValueError("baseline_weights length must equal number of tickers.")
        w0 = w0 / np.sum(w0)
    candidates.append(_clean_weights(w0, bounds))

    # Solve both objectives from each start
    for x0 in starts:
        w_s, ok_s = solve(x0, "max_sharpe", mu_vec, sig, rf, bounds, constraints, cfg)
        if ok_s:
            candidates.append(w_s)

        w_m, ok_m = solve(x0, "min_var", mu_vec, sig, rf, bounds, constraints, cfg)
        if ok_m:
            candidates.append(w_m)

    candidates = unique_weights(candidates, decimals=cfg.round_decimals)

    table = build_candidates_table(candidates, tickers, mu_vec, sig, rf)
    eff = pareto_efficient_set(table)

    # Extract top solutions
    best_row = table.iloc[0]
    best_sharpe = np.array([best_row[f"w_{t}"] for t in tickers], dtype=float)

    idx_min_var = int(table["vol"].idxmin())
    minvar_row = table.loc[idx_min_var]
    min_var = np.array([minvar_row[f"w_{t}"] for t in tickers], dtype=float)

    # Keep only top_n in ranked output if desired (but return full table too)
    table_top = table.head(cfg.top_n).copy()

    return {
        "best_sharpe": best_sharpe,
        "min_var": min_var,
        "candidates": table,        # full ranked set
        "top_candidates": table_top,
        "efficient_set": eff,
    }