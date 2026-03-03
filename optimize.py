# optimize.py
from __future__ import annotations

import argparse
import math
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from portfolio.data import (
    DataConfig,
    get_prices,
    compute_returns,
    portfolio_returns_from_returns,
)
from portfolio.estimation import (
    EstimationConfig,
    estimate_mu_sigma,
)
from portfolio.optimizer import (
    OptimizerConfig,
    generate_portfolios,
)
from portfolio.report import (
    ensure_output_dirs,
    timestamp_tag,
    save_figure,
    save_table,
    write_text_report,
)


# ----------------------------
# CLI parsing + validation
# ----------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Portfolio Optimizer (MPT): min-variance and/or max-Sharpe, candidate set, exports, plots."
    )

    p.add_argument(
        "--tickers",
        type=str,
        required=True,
        help="Comma-separated tickers to optimize (e.g., 'AAPL,MSFT,NVDA,SPY').",
    )

    p.add_argument("--start", type=str, default="2022-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (optional)")
    p.add_argument("--interval", type=str, default="1d", help="yfinance interval (default: 1d)")
    p.add_argument("--cache-days", type=int, default=3, help="Re-download if cache older than N days")

    # Return construction
    p.add_argument(
        "--returns",
        choices=["log", "simple"],
        default="log",
        help="Return type used for estimation + reporting (default: log).",
    )
    p.add_argument(
        "--annualization",
        type=int,
        default=252,
        help="Annualization factor for daily data (default: 252).",
    )

    # Estimation layer
    p.add_argument(
        "--mean",
        choices=["historical", "ewma"],
        default="historical",
        help="Expected return estimator (default: historical).",
    )
    p.add_argument(
        "--cov",
        choices=["sample", "ewma", "shrink"],
        default="shrink",
        help="Covariance estimator (default: shrink).",
    )
    p.add_argument(
        "--lam",
        type=float,
        default=0.94,
        help="EWMA lambda for mean/cov if ewma is used (default: 0.94).",
    )

    # Optimization objectives
    p.add_argument(
        "--objective",
        choices=["max_sharpe", "min_var", "both"],
        default="both",
        help="Which optimizer outputs to prioritize/export (default: both).",
    )
    p.add_argument(
        "--rf",
        type=float,
        default=0.04,
        help="Annual risk-free rate used in Sharpe (e.g., 0.045). Default: 0.04",
    )

    # Constraints
    p.add_argument("--minw", type=float, default=0.0, help="Minimum weight per asset (default: 0.0)")
    p.add_argument("--maxw", type=float, default=1.0, help="Maximum weight per asset (default: 1.0)")
    p.add_argument(
        "--long-only",
        action="store_true",
        help="Enforce long-only weights (default: off unless you pass this flag).",
    )

    # Search
    p.add_argument("--starts", type=int, default=300, help="Number of random multi-starts (default: 300)")
    p.add_argument("--top", type=int, default=20, help="Top N portfolios to export (default: 20)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Baseline portfolio for comparison
    p.add_argument(
        "--baseline-weights",
        type=str,
        default=None,
        help="Optional baseline weights for comparison. Formats:\n"
             "  keyed: 'AAPL=0.5,MSFT=0.3,SPY=0.2'\n"
             "  positional: '0.5,0.3,0.2'\n"
             "If omitted: equal-weight baseline.",
    )

    # Output
    p.add_argument("--outdir", type=str, default="output", help="Output directory")
    p.add_argument("--show-plots", action="store_true", help="Display plots interactively (default: off)")
    p.add_argument("--tag", type=str, default=None, help="Custom run tag for output filenames (default: timestamp)")

    args = p.parse_args(argv)

    # Basic validation
    if args.minw < 0 or args.maxw <= 0:
        p.error("--minw must be >= 0 and --maxw must be > 0")
    if args.minw > args.maxw:
        p.error("--minw must be <= --maxw")
    if not (0.0 <= args.rf < 1.0):
        p.error("--rf must be in [0,1)")
    if args.starts <= 0:
        p.error("--starts must be > 0")
    if args.top <= 0:
        p.error("--top must be > 0")
    if not (0.0 < args.lam < 1.0):
        p.error("--lam must be in (0,1)")

    return args


def parse_tickers(tickers_str: str) -> List[str]:
    tickers: List[str] = []
    for t in tickers_str.split(","):
        t = t.strip().upper()
        if t:
            tickers.append(t)

    if len(tickers) < 2:
        raise ValueError("Provide at least 2 tickers.")

    if len(set(tickers)) != len(tickers):
        raise ValueError(f"Duplicate tickers detected: {tickers}")

    return tickers


def parse_baseline_weights(tickers: List[str], weights_str: Optional[str]) -> np.ndarray:
    """
    Supports:
      --baseline-weights "AAPL=0.5,MSFT=0.3,SPY=0.2"
      --baseline-weights "0.5,0.3,0.2"
      omitted -> equal weights
    """
    n = len(tickers)
    if weights_str is None:
        return np.ones(n) / n

    items = [x.strip() for x in weights_str.split(",") if x.strip()]
    if not items:
        raise ValueError("Baseline weights provided but empty.")

    w_by_t: Dict[str, float] = {}

    if any("=" in it for it in items):
        if any("=" not in it for it in items):
            raise ValueError("Mixed formats not allowed: use either all keyed or all positional weights.")

        for it in items:
            k, v = it.split("=", 1)
            k = k.strip().upper()
            if not k:
                raise ValueError(f"Invalid weight key in {it!r}")
            if k in w_by_t:
                raise ValueError(f"Duplicate baseline weight specified for ticker: {k}")

            try:
                w = float(v.strip())
            except ValueError as e:
                raise ValueError(f"Invalid baseline weight value for {k}: {v!r}") from e

            if not math.isfinite(w):
                raise ValueError(f"Weight for {k} must be finite.")
            if w < 0:
                raise ValueError(f"Weight for {k} must be non-negative.")

            w_by_t[k] = w

        unknown = set(w_by_t) - set(tickers)
        if unknown:
            raise ValueError(f"Baseline weights given for unknown tickers: {sorted(unknown)}")

        missing = set(tickers) - set(w_by_t)
        if missing:
            raise ValueError(f"Missing baseline weights for tickers: {sorted(missing)}")

        w = np.array([w_by_t[t] for t in tickers], dtype=float)

    else:
        if len(items) != n:
            raise ValueError(f"Positional baseline weights count ({len(items)}) must match tickers ({n}).")
        w = np.array([float(x) for x in items], dtype=float)
        if not np.isfinite(w).all():
            raise ValueError("Baseline weights must be finite.")
        if (w < 0).any():
            raise ValueError("Baseline weights must be non-negative.")

    s = float(w.sum())
    if s <= 0:
        raise ValueError("Baseline weights sum must be > 0.")
    w = w / s
    return w


# ----------------------------
# Local plotting (optimizer-specific)
# ----------------------------

def equity_curve_from_returns(r: pd.Series, method: str) -> pd.Series:
    if method == "log":
        return np.exp(r.cumsum())
    return (1.0 + r).cumprod()


def max_drawdown_from_equity(eq: pd.Series) -> float:
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())


def plot_equity(r: pd.Series, title: str, returns_method: str) -> plt.Figure:
    eq = equity_curve_from_returns(r.dropna(), returns_method)
    fig, ax = plt.subplots()
    ax.plot(eq.index, eq.values)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    fig.tight_layout()
    return fig


def plot_drawdown(r: pd.Series, title: str, returns_method: str) -> plt.Figure:
    eq = equity_curve_from_returns(r.dropna(), returns_method)
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    fig, ax = plt.subplots()
    ax.plot(dd.index, dd.values)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    fig.tight_layout()
    return fig


def plot_frontier(candidates: pd.DataFrame, title: str = "Candidate Set (Risk-Return)") -> plt.Figure:
    fig, ax = plt.subplots()
    ax.scatter(candidates["vol"], candidates["exp_return"], s=12, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Volatility (annualized)")
    ax.set_ylabel("Expected Return (annualized)")
    fig.tight_layout()
    return fig


def plot_weights_bar(weights: np.ndarray, tickers: List[str], title: str) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.bar(tickers, weights)
    ax.set_title(title)
    ax.set_xlabel("Assets")
    ax.set_ylabel("Weight")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


# ----------------------------
# Metrics for optimizer outputs
# ----------------------------

def annualized_return(r: pd.Series, returns_method: str, annualization: int) -> float:
    r = r.dropna()
    if len(r) < 2:
        return float("nan")
    if returns_method == "log":
        return float(np.exp(r.mean() * annualization) - 1.0)
    # geometric CAGR from cumulative return
    eq = (1.0 + r).cumprod()
    years = len(r) / float(annualization)
    if years <= 0:
        return float("nan")
    return float(eq.iloc[-1] ** (1.0 / years) - 1.0)


def annualized_vol(r: pd.Series, annualization: int) -> float:
    r = r.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * math.sqrt(annualization))


def sharpe_ratio(r: pd.Series, rf: float, returns_method: str, annualization: int) -> float:
    ar = annualized_return(r, returns_method, annualization)
    av = annualized_vol(r, annualization)
    if not np.isfinite(av) or av <= 0:
        return float("nan")
    return float((ar - rf) / av)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    args = parse_args()
    tickers = parse_tickers(args.tickers)
    baseline_w = parse_baseline_weights(tickers, args.baseline_weights)

    data_cfg = DataConfig(
        start=args.start,
        end=args.end,
        interval=args.interval,
        cache_days=args.cache_days,
    )

    prices = get_prices(tickers, data_cfg)
    asset_returns = compute_returns(prices, method=args.returns)

    est_cfg = EstimationConfig(
        annualization=args.annualization,
        ewma_lambda=args.lam,
    )
    mu, sigma = estimate_mu_sigma(
        asset_returns,
        mean_method=args.mean,
        cov_method=args.cov,
        cfg=est_cfg,
    )

    opt_cfg = OptimizerConfig(
        long_only=bool(args.long_only),
        min_weight=float(args.minw),
        max_weight=float(args.maxw),
        n_starts=int(args.starts),
        top_n=int(args.top),
        random_seed=int(args.seed),
    )

    res = generate_portfolios(
        tickers=tickers,
        mu=mu,
        sigma=sigma,
        rf=float(args.rf),
        cfg=opt_cfg,
        baseline_weights=baseline_w,
    )

    candidates: pd.DataFrame = res["candidates"]       # full ranked by Sharpe (from generator)
    efficient_set: pd.DataFrame = res["efficient_set"]
    w_best_sharpe: np.ndarray = res["best_sharpe"]
    w_min_var: np.ndarray = res["min_var"]

    # Build portfolio return series (constant-weight daily rebalanced)
    pr_baseline = portfolio_returns_from_returns(asset_returns, baseline_w, rebalance=True)
    pr_best_sharpe = portfolio_returns_from_returns(asset_returns, w_best_sharpe, rebalance=True)
    pr_min_var = portfolio_returns_from_returns(asset_returns, w_min_var, rebalance=True)

    rets_df = pd.concat(
        {
            "baseline": pr_baseline,
            "best_sharpe": pr_best_sharpe,
            "min_var": pr_min_var,
        },
        axis=1,
    ).dropna(how="any")

    # Rank top candidates depending on chosen objective
    if args.objective == "max_sharpe":
        ranked = candidates.sort_values("sharpe", ascending=False).reset_index(drop=True)
        top = ranked.head(args.top).copy()
        top_name = f"top_{args.top}_by_sharpe"
    elif args.objective == "min_var":
        ranked = candidates.sort_values("vol", ascending=True).reset_index(drop=True)
        top = ranked.head(args.top).copy()
        top_name = f"top_{args.top}_by_min_var"
    else:
        # both: export both orderings
        ranked_sharpe = candidates.sort_values("sharpe", ascending=False).reset_index(drop=True)
        ranked_vol = candidates.sort_values("vol", ascending=True).reset_index(drop=True)
        top_sharpe = ranked_sharpe.head(args.top).copy()
        top_vol = ranked_vol.head(args.top).copy()
        top = None  # handled below
        top_name = "both"

    # Output dirs + tag
    dirs = ensure_output_dirs(args.outdir)
    tag = args.tag.strip() if args.tag else timestamp_tag()

    # Save tables
    save_table(mu.to_frame("mu").round(8), dirs["tables"] / f"mu_{tag}.csv")
    save_table(sigma.round(10), dirs["tables"] / f"sigma_{tag}.csv")

    save_table(candidates.round(10), dirs["tables"] / f"candidates_ranked_by_sharpe_{tag}.csv")
    save_table(efficient_set.round(10), dirs["tables"] / f"efficient_set_{tag}.csv")

    if args.objective == "both":
        save_table(top_sharpe.round(10), dirs["tables"] / f"top_{args.top}_by_sharpe_{tag}.csv")
        save_table(top_vol.round(10), dirs["tables"] / f"top_{args.top}_by_min_var_{tag}.csv")
    else:
        save_table(top.round(10), dirs["tables"] / f"{top_name}_{tag}.csv")

    # Save portfolio returns (for further analysis/pyfolio/etc.)
    save_table(rets_df.round(10), dirs["tables"] / f"portfolio_returns_{tag}.csv")

    # Summary metrics table
    summary_rows = []

    # Map portfolio name -> weights vector (aligned with `tickers`)
    weights_map = {
        "baseline": baseline_w,
        "best_sharpe": w_best_sharpe,
        "min_var": w_min_var,
    }

    def normalize_weights(w):
        w = np.asarray(w, dtype=float)
        s = np.nansum(w)
        return w / s if s and np.isfinite(s) else w

    for name in rets_df.columns:
        r = rets_df[name]
        eq = equity_curve_from_returns(r, args.returns)

        w = weights_map.get(name)
        hhi = float(np.sum(np.square(normalize_weights(w)))) if w is not None else np.nan

        summary_rows.append(
            {
                "name": name,
                "ann_return": annualized_return(r, args.returns, args.annualization),
                "ann_vol": annualized_vol(r, args.annualization),
                "sharpe": sharpe_ratio(r, args.rf, args.returns, args.annualization),
                "max_dd": max_drawdown_from_equity(eq),
                "hhi": hhi,
            }
        )

    summary = pd.DataFrame(summary_rows).set_index("name")
    save_table(summary.round(8), dirs["tables"] / f"summary_{tag}.csv")

    # Save key weights tables
    wtbl = pd.DataFrame(
        {
            "baseline": baseline_w,
            "best_sharpe": w_best_sharpe,
            "min_var": w_min_var,
        },
        index=tickers,
    )
    save_table(wtbl.round(8), dirs["tables"] / f"weights_{tag}.csv")

    # Plots
    fig = plot_frontier(candidates, title="Candidate Set (Risk-Return)")
    save_figure(fig, dirs["figures"] / f"frontier_{tag}.png")
    plt.close(fig)

    for name in rets_df.columns:
        fig = plot_equity(rets_df[name], title=f"{name} Equity Curve", returns_method=args.returns)
        save_figure(fig, dirs["figures"] / f"{name}_equity_{tag}.png")
        plt.close(fig)

        fig = plot_drawdown(rets_df[name], title=f"{name} Drawdowns", returns_method=args.returns)
        save_figure(fig, dirs["figures"] / f"{name}_drawdowns_{tag}.png")
        plt.close(fig)

    fig = plot_weights_bar(baseline_w, tickers, title="Weights: Baseline")
    save_figure(fig, dirs["figures"] / f"weights_baseline_{tag}.png")
    plt.close(fig)

    fig = plot_weights_bar(w_best_sharpe, tickers, title="Weights: Best Sharpe")
    save_figure(fig, dirs["figures"] / f"weights_best_sharpe_{tag}.png")
    plt.close(fig)

    fig = plot_weights_bar(w_min_var, tickers, title="Weights: Min Variance")
    save_figure(fig, dirs["figures"] / f"weights_min_var_{tag}.png")
    plt.close(fig)

    # Text report
    start_dt = rets_df.index.min().date()
    end_dt = rets_df.index.max().date()

    lines = [
        f"PORTFOLIO OPTIMIZATION REPORT  |  run={tag}",
        f"Tickers: {', '.join(tickers)}",
        f"Date range (returns): {start_dt} -> {end_dt}",
        f"Returns: {args.returns} | annualization={args.annualization}",
        f"Estimation: mean={args.mean} | cov={args.cov} | ewma_lambda={args.lam}",
        f"Optimization: objective={args.objective} | long_only={bool(args.long_only)} | minw={args.minw} | maxw={args.maxw}",
        f"Search: starts={args.starts} | seed={args.seed}",
        "",
        "Summary metrics (baseline vs optimized):",
        summary.round(6).to_string(),
        "",
        "Top weights (best_sharpe):",
        pd.Series(w_best_sharpe, index=tickers).sort_values(ascending=False).round(6).to_string(),
        "",
        "Top weights (min_var):",
        pd.Series(w_min_var, index=tickers).sort_values(ascending=False).round(6).to_string(),
        "",
        "Notes:",
        "- Candidate ranking uses in-sample mu/sigma estimates (not out-of-sample).",
        "- If you want institutional-grade robustness, next step is walk-forward OOS + turnover/cost penalties.",
    ]
    write_text_report(lines, dirs["base"] / f"opt_report_{tag}.txt")

    # Console output
    print("Saved outputs to:", dirs["base"])
    print("\nSummary:\n", summary.round(6))
    print("\nBest Sharpe weights:\n", pd.Series(w_best_sharpe, index=tickers).sort_values(ascending=False).round(6))
    print("\nMin Variance weights:\n", pd.Series(w_min_var, index=tickers).sort_values(ascending=False).round(6))

    if args.show_plots:
        plt.show()


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    s = float(w.sum())
    return w / s if s != 0 else w


def get_weights_for_name(
    name: str,
    baseline_w: np.ndarray,
    w_best_sharpe: np.ndarray,
    w_min_var: np.ndarray,
) -> np.ndarray:
    if name == "baseline":
        return baseline_w
    if name == "best_sharpe":
        return w_best_sharpe
    if name == "min_var":
        return w_min_var
    raise ValueError(f"Unknown portfolio name: {name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(2)
    except Exception:
        raise