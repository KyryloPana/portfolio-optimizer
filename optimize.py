from __future__ import annotations

import argparse
import math
import sys

import matplotlib.pyplot as plt
import pandas as pd

from portfolio.data import (
    DataConfig,
    get_prices,
    prices_to_returns,
    portfolio_return,
    load_series_from_csv,
)
from portfolio.metrics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
    benchmark_summary,
)
from portfolio.plots import (
    plot_equity_curve,
    plot_drawdowns,
    plot_rolling_volatility,
    plot_rolling_sharpe,
)
from portfolio.report import (
    ensure_output_dirs,
    timestamp_tag,
    save_figure,
    save_table,
    write_text_report,
)

from portfolio.pyfolio_report import pyfolio_generate


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Portfolio Analytics Tool: metrics, plots, benchmark comparison, and exports."
    )
    src_portfolio = p.add_mutually_exclusive_group(required=True)
    src_benchmark = p.add_mutually_exclusive_group(required=True)

    src_portfolio.add_argument(
        "--portfolio-tickers",
        type=str,
        help="Comma-separated tickers used to construct PORTFOLIO (e.g., 'AAPL,MSFT,SPY').",
    )

    src_portfolio.add_argument(
        "--portfolio-csv",
        type=str,
        help="Path to CSV file containing a precomputed portfolio series (returns or prices).",
    )

    # only meaningful in CSV mode
    p.add_argument(
        "--portfolio-csv-format",
        type=str,
        choices=["returns", "prices"],
        default="returns",
        help="CSV content type: 'returns' (default) or 'prices'. Used only with --portfolio-csv.",
    )

    # only meaningful in tickers mode
    p.add_argument(
        "--portfolio-weights",
        type=str,
        default=None,
        help="Comma-separated ticker=weight pairs (e.g., 'AAPL=0.5,MSFT=0.3,SPY=0.2'). "
        "If omitted, equal weights are used.",
    )

    p.add_argument(
        "--v0",
        type=float,
        default=1.0,
        help="Initial portfolio value (scale factor). Default: 1.0.",
    )

    p.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated tickers to analyze (e.g., 'AAPL,MSFT,SPY')",
    )

    src_benchmark.add_argument(
        "--benchmark-ticker",
        type=str,
        default=None,
        help="Benchmark ticker symbol (e.g., 'SPY')",
    )

    src_benchmark.add_argument(
        "--benchmark-csv",
        type=str,
        default=None,
        help="Path to CSV file containing a precomputed benchmark series (returns or prices).",
    )

    # only meaningful in CSV mode
    p.add_argument(
        "--benchmark-csv-format",
        type=str,
        choices=["returns", "prices"],
        default="returns",
        help="CSV content type: 'returns' (default) or 'prices'. Used only with --benchmark-csv.",
    )

    p.add_argument(
        "--start", type=str, default="2022-01-01", help="Start date YYYY-MM-DD"
    )
    p.add_argument(
        "--end", type=str, default=None, help="End date YYYY-MM-DD (optional)"
    )
    p.add_argument(
        "--cache-days",
        type=int,
        default=3,
        help="Re-download data if cached file is older than N days",
    )
    p.add_argument("--outdir", type=str, default="output", help="Output directory")
    p.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively (default: off)",
    )

    p.add_argument(
        "--pyfolio", action="store_true", help="Generate PyFolio report (optional)"
    )
    p.add_argument(
        "--pyfolio-target",
        type=str,
        default=None,
        help="Ticker/column to run PyFolio on (default: first non-benchmark ticker)",
    )

    p.add_argument(
        "--pyfolio-out",
        type=str,
        default=None,
        help="Path to save PyFolio HTML report (optional)",
    )
    p.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Custom run tag used in output filenames (default: timestamp)",
    )

    args = p.parse_args(argv)
    # Cross-argument validation
    if args.portfolio_csv and args.portfolio_weights:
        p.error(
            "--portfolio-weights can only be used with --portfolio-tickers (not with --portfolio-csv)"
        )

    if args.portfolio_csv is None and args.portfolio_csv_format != "returns":
        # if passing --portfolio-csv-format without --portfolio-csv
        p.error("--portfolio-csv-format is only valid when using --portfolio-csv")

    if args.benchmark_csv and args.benchmark_ticker:
        p.error("you can either use --benchmark-ticker or --benchmark-csv")

    if args.benchmark_csv is None and args.benchmark_csv_format != "returns":
        # if passing --portfolio-csv-format without --portfolio-csv
        p.error("--benchmark-csv-format is only valid when using --benchmark-csv")

    return args


def parse_portfolio(portfolio_tickers: str, portfolio_weights: str | None):
    """
    Supports:
        --portfolio-tickers "SPY,TLT"
        --portfolio-weights "SPY=0.6,TLT=0.4"   # keyed
        --portfolio-weights "0.6,0.4"           # positional
        --portfolio-weights omitted             # equal weights
    """

    portfolio_tickers_list = []
    for pt in portfolio_tickers.split(","):
        pt = pt.strip().upper()
        if not pt:
            continue
        portfolio_tickers_list.append(pt)

    # check for duplicates
    if len(set(portfolio_tickers_list)) != len(portfolio_tickers_list):
        raise ValueError(
            f"Duplicate tickers in --portfolio-tickers: {portfolio_tickers_list}"
        )

    # make a dictionary with ticker and its weight
    weights_by_ticker: dict[str, float] = {}
    if portfolio_weights:
        items = []
        for item in portfolio_weights.split(","):
            item = item.strip()
            if not item:
                continue
            items.append(item)
        if not items:
            raise ValueError(
                "Portfolio weights were provided but contained no usable items."
            )

        if any("=" in it for it in items):
            if any("=" not in it for it in items):
                raise ValueError(
                    "Mixed weight formats are not allowed. Use either "
                    "'TICKER=weight' for all items or plain numbers only."
                )
            for item in items:
                k, v = item.split("=", 1)
                k = k.strip().upper()
                if not k:
                    raise ValueError(f"Invalid weight key in item: {item!r}")

                if k in weights_by_ticker:
                    raise ValueError(f"Duplicate weight specified for ticker: {k}")

                try:
                    w = float(v.strip())
                except ValueError as e:
                    raise ValueError(f"Invalid weight value for {k}: {v!r}") from e

                if not math.isfinite(w):
                    raise ValueError(f"Weight for {k} must be finite (got {w}).")
                if w < 0:
                    raise ValueError(f"Weight for {k} must be non-negative (got {w})")

                weights_by_ticker[k] = w

            # validate tickers set and tickers for which weights are given
            unknown = set(weights_by_ticker) - set(portfolio_tickers_list)
            if unknown:
                raise ValueError(
                    f"Weights given for unknown tickers: {sorted(unknown)}"
                )

            missing = set(portfolio_tickers_list) - set(weights_by_ticker)
            if missing:
                raise ValueError(f"Missing weights for tickers: {sorted(missing)}")

        else:
            if len(items) != len(portfolio_tickers_list):
                raise ValueError(
                    f"Number of weights ({len(items)}) must match number of tickers "
                    f"({len(portfolio_tickers_list)}) when using plain numeric weights."
                )

            for t, w_str in zip(portfolio_tickers_list, items):
                try:
                    w = float(w_str)
                except ValueError as e:
                    raise ValueError(f"Invalid weight value for {t}: {w_str!r}") from e

                if not math.isfinite(w):
                    raise ValueError(f"Weight for {t} must be finite (got {w}).")
                if w < 0:
                    raise ValueError(f"Weight for {t} must be non-negative (got {w}).")

                weights_by_ticker[t] = float(w)

        total = sum(weights_by_ticker[t] for t in portfolio_tickers_list)
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"Weights must sum to 1.0 (got {total})")
    else:
        # equal weights
        w = 1.0 / len(portfolio_tickers_list)
        weights_by_ticker = {t: w for t in portfolio_tickers_list}

    return portfolio_tickers_list, weights_by_ticker


def main() -> None:
    args = parse_args()
    cfg = DataConfig(start=args.start, end=args.end, cache_days=args.cache_days)

    if args.portfolio_csv is not None:
        portfolio_returns = load_series_from_csv(
            args.portfolio_csv.strip(),
            fmt=args.portfolio_csv_format.strip().lower(),
            series_name="Portfolio",
        ).dropna()

    elif args.portfolio_tickers is not None:
        portfolio_tickers, weights_by_ticker = parse_portfolio(
            args.portfolio_tickers, args.portfolio_weights
        )
        portfolio_prices = get_prices(portfolio_tickers, cfg)
        portfolio_returns = portfolio_return(
            portfolio_prices, weights_by_ticker, args.v0
        ).dropna()
        portfolio_returns.name = "Portfolio"

    if args.benchmark_csv is not None:
        benchmark_rets = load_series_from_csv(
            args.benchmark_csv.strip(),
            fmt=args.benchmark_csv_format.strip().lower(),
            series_name="Benchmark",
        ).dropna()
        benchmark_symbol = "Benchmark"

    elif args.benchmark_ticker is not None:
        b = (args.benchmark_ticker or "SPY").strip().upper()
        benchmark_rets = prices_to_returns(get_prices([b], cfg))[b].dropna()
        benchmark_rets.name = "Benchmark"
        benchmark_symbol = b

    if args.tickers is not None:
        tickers = []
        for t in args.tickers.split(","):
            tickers.append(t.strip().upper())
        prices = get_prices(tickers, cfg)
        tickers_rets = prices_to_returns(prices)
        rets_df = pd.concat(
            [portfolio_returns, tickers_rets, benchmark_rets], axis=1, join="outer"
        ).sort_index()
    else:
        rets_df = pd.concat(
            [portfolio_returns, benchmark_rets], axis=1, join="outer"
        ).sort_index()

    core = pd.DataFrame(index=rets_df.columns)
    core["cagr"] = annualized_return(rets_df)
    core["vol"] = annualized_volatility(rets_df)
    core["sharpe"] = sharpe_ratio(rets_df)
    core["max_dd"] = max_drawdown(rets_df)

    strategy_df = rets_df.drop(columns=[benchmark_rets.name], errors="ignore")
    rel = benchmark_summary(strategy_df, benchmark_rets)

    print("Core metrics:\n", core.round(4))
    print("\nBenchmark-relative stats:\n", rel.round(4))

    dirs = ensure_output_dirs(args.outdir)
    tag = timestamp_tag()

    save_table(core.round(6), dirs["tables"] / f"core_metrics_{tag}.csv")
    save_table(rel.round(6), dirs["tables"] / f"benchmark_summary_{tag}.csv")

    for col in rets_df.columns:
        fig = plot_equity_curve(rets_df[col], title=f"{col} Equity Curve")
        save_figure(fig, dirs["figures"] / f"{col}_equity_{tag}.png")
        plt.close(fig)

        fig = plot_drawdowns(rets_df[col], title=f"{col} Drawdowns")
        save_figure(fig, dirs["figures"] / f"{col}_drawdowns_{tag}.png")
        plt.close(fig)

        fig = plot_rolling_volatility(rets_df[col], title=f"{col} Rolling Volatility")
        save_figure(fig, dirs["figures"] / f"{col}_rolling_vol_{tag}.png")
        plt.close(fig)

        fig = plot_rolling_sharpe(rets_df[col], title=f"{col} Rolling Sharpe")
        save_figure(fig, dirs["figures"] / f"{col}_rolling_sharpe_{tag}.png")
        plt.close(fig)

    start = rets_df.index.min().date()
    end = rets_df.index.max().date()

    lines = [
        f"PORTFOLIO ANALYTICS REPORT  |  run={tag}",
        f"Date range: {start} -> {end}",
        f"Assets: {', '.join(rets_df.columns)}",
        "",
        "Core metrics:",
        core.round(4).to_string(),
        "",
        f"Benchmark-relative metrics (benchmark={benchmark_symbol}):",
        rel.round(4).to_string(),
    ]
    write_text_report(lines, dirs["base"] / f"report_{tag}.txt")

    if args.show_plots:
        plt.show()

    if args.pyfolio:
        candidates = []
        """
            the tool supports multi-ticker analysis (Data Frame of returns); 
            PyFolio can generate a report only for one return series at a time (a pandas Series)
            So here it is chosen a column from rets_df to be “the strategy” that PyFolio will analyze
        """
        for c in rets_df.columns:
            if c != benchmark_symbol:
                candidates.append(c)

        if len(candidates) > 0:
            if args.pyfolio_target is not None:
                target = args.pyfolio_target.strip()
                if not "Portfolio":
                    target.upper()

            elif args.portfolio_tickers is not None:
                target = "Portfolio"
            else:
                # chooses first non-benchmark ticker from the order user typed
                target = tickers[0]
        else:
            target = benchmark_symbol

        pyfolio_html = args.pyfolio_out
        if pyfolio_html is None:
            pyfolio_html = str(dirs["base"] / f"pyfolio_{target}_{tag}.html")

        out_path = pyfolio_generate(
            returns=rets_df[target],
            benchmark_rets=benchmark_rets,
            output_html=pyfolio_html,
            title=f"PyFolio Tear Sheet — {target}",
            tag=tag,
            target=target,
        )
        print(f"PyFolio report saved: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(2)
    except Exception:
        raise  # real bug, show traceback
