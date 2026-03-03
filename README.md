# Portfolio Analytics Tool — v0.2.0

A Python-based command-line tool for analyzing portfolio performance using historical market data and user-provided series.

It computes core risk/return metrics, benchmark-relative statistics, rolling diagnostics, and exports reproducible reports (tables, figures, and a text summary).

Designed as a foundational analytics engine for quantitative research, risk analysis, and systematic evaluation—built to be extended toward multi-strategy comparison in future releases.

---

## 1. Key Features (v0.2.0)

- **Deterministic market data ingestion**
  - Downloads historical adjusted prices via Yahoo Finance (`yfinance`)
  - Uses local caching to avoid unnecessary re-downloads (`--cache-days`)

- **Flexible portfolio + benchmark inputs (ticker-built or CSV-loaded)**
  - Portfolio from tickers: `--portfolio-tickers` with optional `--portfolio-weights`
  - Portfolio from CSV: `--portfolio-csv` with `--portfolio-csv-format {returns,prices}`
  - Benchmark from ticker: `--benchmark-ticker`
  - Benchmark from CSV: `--benchmark-csv` with `--benchmark-csv-format {returns,prices}`

- **Return construction and portfolio building**
  - Converts prices to returns when needed (Yahoo Finance pulls or CSV in `prices` mode)
  - Builds a buy-and-hold portfolio return series from weights and initial capital (`--v0`)

- **Core performance metrics**
  - CAGR (compound annual growth rate)
  - Annualized volatility
  - Sharpe ratio
  - Maximum drawdown

- **Benchmark-relative analytics**
  - Beta
  - Annualized alpha
  - Tracking error
  - Information ratio
  - Correlation  
  *(computed on the overlapping date range between strategy and benchmark)*

- **Rolling diagnostics**
  - Rolling volatility
  - Rolling Sharpe ratio

- **Automated exports**
  - Figures (PNG)
  - Tables (CSV)
  - Text-based summary report

- **Optional PyFolio tear sheet export**
  - Generates a PyFolio tear sheet for a selected return series (`--pyfolio`)
  - Optional HTML export path (`--pyfolio-out`)
  - Supports selecting the target series (`--pyfolio-target`)

- **CLI interface**
  - Dates (`--start`, `--end`), caching (`--cache-days`), output directory (`--outdir`)
  - Optional interactive plotting (`--show-plots`)
  - Run tagging for reproducible outputs (`--tag`)

---

## 2. Project Structure

```bash
portfolio-analytics-tool/
├─ portfolio/
│  ├─ __init__.py
│  ├─ data.py            # Market data ingestion & caching (yfinance) + CSV series loading
│  ├─ metrics.py         # Performance & risk analytics + benchmark-relative metrics
│  ├─ plots.py           # Visualization utilities
│  ├─ report.py          # Export + reporting utilities (tables/figures/text)
│  └─ pyfolio_report.py  # OPTIONAL: PyFolio tear sheet export
├─ data/
│  ├─ sample_portfolio.csv
│  └─ sample_benchmark.csv
├─ notebooks/            # Optional notebooks / scratch work
├─ output/               # Default output directory (generated)
├─ optimize.py            # CLI entry point / orchestration layer
├─ README.md
├─ CHANGELOG.md
├─ requirements.txt
└─ .gitignore

```

**Design principles**

- Clear separation of concerns (ingestion → analytics → plots → exports)
- Single-responsibility modules (each file has one job)
- No hidden side effects (explicit inputs/outputs, predictable artifacts)
- Deterministic outputs given identical inputs (incl. cached market data)

---

## 3. Installation

### Prerequisites
- Python 3.10 or newer
- `pip` package manager
- Internet connection (for initial market data download)

### Steps

1. Clone the repository:
```bash
git clone https://github.com/KyryloPana/portfolio-analytics-tool.git
cd portfolio-analytics-tool
```
2. (Optional but recommended) Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. (Optional) Install PyFolio add-on (v0.1.1):
```bash
pip install pyfolio-reloaded
```
---

## 4. Quick Start

### Run using the included sample CSV files (prices)
```bash
python optimize.py  --portfolio-csv './data/sample_portfolio.csv'  --portfolio-csv-format 'prices'  --benchmark-csv './data/sample_benchmark.csv' --benchmark-csv-format 'prices'
```

### Run a portfolio built from tickers (equal weights) vs a benchmark ticker
```bash
python optimize.py  --portfolio-tickers "AAPL,MSFT,SPY"  --benchmark-ticker 'SPY'
```

### Run a portfolio built from tickers with explicit weights
```bash
python optimize.py  --portfolio-tickers "SPY,TLT,GLD"  --portfolio-weights "SPY=0.6,TLT=0.3,GLD=0.1"  --benchmark-ticker 'SPY'
```

Default configuration
- Start date: 2022-01-01
- End date: latest available
- Cache validity: 3 days
- Output directory: output/

---

## 5. CLI Usage

### Examples

**1) Run using the included sample CSV files (both are prices)**
```bash
python optimize.py  --tickers 'AAPL,MSFT,SPY'  --benchmark 'SPY'  --start '2018-01-01'  --cache-days 5  --outdir 'output'  --show-plots
```

**2) Build a portfolio from tickers (equal weights) vs a benchmark ticker**
```bash
python optimize.py  --portfolio-tickers "AAPL,MSFT,SPY"  --benchmark-ticker 'SPY'  --start '2018-01-01'  --cache-days 5  --outdir 'output'  --show-plots
```

**3) Build a portfolio from tickers with explicit weights**
```bash
python optimize.py   --portfolio-tickers "SPY,TLT,GLD"   --portfolio-weights "SPY=0.6,TLT=0.3,GLD=0.1"   --benchmark-ticker 'SPY'   --start '2018-01-01' 
```

**4) Optional: PyFolio tear sheet**
```bash
python optimize.py   --portfolio-tickers "MSTR,NVDA,AAPL"   --benchmark-ticker SPY   --start 2018-01-01  --pyfolio --pyfolio-target NVDA --pyfolio-out "output/pyfolio.html"
```



**Available Arguments**
| Argument | Description |
|---|---|
| `--portfolio-tickers` | Comma-separated tickers used to construct PORTFOLIO (e.g., `"AAPL,MSFT,SPY"`) |
| `--portfolio-weights` | Optional weights for `--portfolio-tickers`: `TICKER=w` pairs (e.g., `"AAPL=0.5,MSFT=0.3,SPY=0.2"`) or positional list (e.g., `"0.5,0.3,0.2"`) |
| `--portfolio-csv` | Path to CSV containing a portfolio series (returns or prices) |
| `--portfolio-csv-format` | CSV content type: `returns` (default) or `prices` (used only with `--portfolio-csv`) |
| `--benchmark-ticker` | Benchmark ticker symbol (e.g., `"SPY"`) |
| `--benchmark-csv` | Path to CSV containing a benchmark series (returns or prices) |
| `--benchmark-csv-format` | CSV content type: `returns` (default) or `prices` (used only with `--benchmark-csv`) |
| `--V0` | Initial portfolio value (scale factor). Default: `1.0` |
| `--tickers` | Optional extra tickers to include in the analysis panel (add-on columns) |
| `--start` | Start date (`YYYY-MM-DD`). Default: `2022-01-01` |
| `--end` | End date (`YYYY-MM-DD`, optional). Default: latest available |
| `--cache-days` | Re-download market data if cache is older than N days. Default: `3` |
| `--outdir` | Output directory. Default: `output/` |
| `--show-plots` | Display plots interactively (default: off) |
| `--pyfolio` | Generate PyFolio tear sheet (optional add-on) |
| `--pyfolio-target` | Column/ticker to analyze with PyFolio (default: first non-benchmark series) |
| `--pyfolio-out` | Output path for PyFolio HTML (optional) |
| `--tag` | Custom run tag used in output filenames (default: timestamp) |


***To see the full, authoritative CLI help:***
```bash
python optimize.py --help
```
---

## 6. Outputs
Each run generates a tagged (timestamped by default) set of outputs in the output directory.

### Figures (`output/figures/`)
- Equity curve
- Drawdowns
- Rolling volatility
- Rolling Sharpe ratio

### Tables (`output/tables/`)
- Core performance metrics
- Benchmark-relative statistics

### Text Report (`output/`)
A concise, human-readable summary including:
- Date range
- Assets analyzed
- Core metrics
- Benchmark-relative metrics

### PyFolio Report (optional)
If `--pyfolio` is enabled:
- An HTML tear sheet (path controlled by `--pyfolio-out`)
- A companion folder of PNGs next to the HTML (captured PyFolio figures)

All outputs are deterministic and reproducible given the same inputs (including cached market data).


## 7. Methodology Notes

- Prices are sourced from Yahoo Finance (`yfinance`) and use adjusted pricing (`auto_adjust=True`) to account for splits/dividends.
- Returns are computed as simple arithmetic returns (`pct_change`).
- Annualization assumes 252 trading days per year.
- Alpha and beta use a realized CAPM-style formulation.
- Information ratio is undefined when tracking error is zero (e.g., comparing a series to itself over the same dates).
- Rolling metrics use a default 63-day (~3-month) window.

### PyFolio notes (optional)
- PyFolio generates a tear sheet for a single return series (`pd.Series`).
- When multiple return columns are available, the tool selects one target series (or uses `--pyfolio-target` if provided).
- PyFolio is treated as an optional reporting utility; the core analytics pipeline remains independent.

This tool prioritizes clarity, correctness, and reproducibility over predictive claims.


---

## 8. Roadmap

### v0.2.1 — Multi-Strategy Comparison + PyFolio Targeting
- Multi-strategy comparison (side-by-side metrics/plots vs the same benchmark)
- Strategy-level benchmarking (pairwise alignment per strategy/benchmark)
- Cleaner PyFolio targeting and export behavior for multi-series inputs (`--pyfolio-target`, `--pyfolio-out`)
- Additional input/validation polish around CSV series (prices vs returns) and naming consistency

### v0.3.0 — Risk Scaling & Regime Awareness
- Regime-conditioned analytics
- Regime segmentation
- Volatility targeting and risk scaling
- Expanded internal metrics (progressive removal of external dependencies such as PyFolio)
- Expanded benchmark universe

### v0.4.0 — Factor & Attribution Analysis
- Factor regression framework
- Performance attribution
- Rolling factor exposure analysis
- Attribution tables and charts

> **Note:** PyFolio support is transitional and may be deprecated in future releases in favor of native, transparent analytics.

---

## Disclaimer

This project is for research and educational purposes only.  
It does not constitute financial advice.

---

## Why This Project Exists

Most beginner finance projects stop at plotting prices.

This project focuses on how performance is evaluated in professional settings:  
risk-adjusted returns, benchmark context, drawdowns, and regime sensitivity.

It is intentionally built as a clean, extensible analytics core suitable as a base  
for more advanced quantitative research.
