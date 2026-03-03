# Changelog

## v0.2.0
- Added **flexible portfolio/benchmark input sources**:
  - Portfolio from `--portfolio-tickers` + optional `--portfolio-weights`
  - Portfolio from `--portfolio-csv` with `--portfolio-csv-format {returns,prices}`
  - Benchmark from `--benchmark-ticker`
  - Benchmark from `--benchmark-csv` with `--benchmark-csv-format {returns,prices}`
- Introduced **generic CSV series loader** path for both portfolio and benchmark flows
- Added support for **price-to-return conversion** when CSV input is provided as prices
- Switched core return panel construction to **outer-join master frame** to preserve full history for plotting/export
- Standardized validation for portfolio weights:
  - keyed (`TICKER=weight`) and positional (`w1,w2,...`) formats
  - duplicate/unknown/missing ticker checks
  - finite, non-negative, sum-to-1 constraints
- Improved CLI guardrails and cross-argument checks:
  - source-mode consistency checks
  - clearer input validation errors
- Improved runtime UX for expected user mistakes:
  - clean error output without traceback for validation failures
- Refactored main pipeline for cleaner separation of concerns:
  - independent portfolio resolution
  - independent benchmark resolution
  - unified return-frame assembly for analytics/plots/reports
- Internal cleanup and robustness improvements to support upcoming multi-strategy comparison work

## v0.1.1
- Added **optional PyFolio integration** via `pyfolio-reloaded`
  - Generates a PyFolio tear sheet for a selected return series
  - Exports a standalone HTML report with embedded performance tables
  - Captures PyFolio Matplotlib figures and saves them as PNG files alongside the HTML
- Introduced CLI flags for PyFolio reporting:
  - `--pyfolio` to enable tear sheet generation
  - `--pyfolio-out` to control output file or directory
- Ensured PyFolio remains a **non-core, optional add-on**
  - Core analytics pipeline is unaffected if PyFolio is not installed
- Improved output handling and robustness for report generation paths
- Internal cleanup and refactoring to support future extensibility

## v0.1.0
- Deterministic market data ingestion with caching (yfinance)
- Portfolio return construction and core risk/return metrics (CAGR, vol, Sharpe, max drawdown)
- Benchmark-relative analytics (beta, alpha, tracking error, information ratio, correlation)
- Rolling diagnostics (rolling volatility and rolling Sharpe)
- Export of figures, tables, and text reports
- CLI interface for tickers, benchmark, date range, cache-days, and output directory
