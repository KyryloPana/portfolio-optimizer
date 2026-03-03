# Portfolio Optimizer (MPT) — v0.2

A reproducible, modular portfolio optimization tool built on Modern Portfolio Theory (MPT).

Core goals:
- Optimize **minimum variance** and/or **maximum Sharpe ratio** (return per unit risk)
- Generate a **ranked candidate set** of feasible portfolios (not a single fragile allocation)
- Enforce practical constraints (long-only, min/max weights, concentration control)
- Improve estimator robustness (EWMA expected returns, covariance shrinkage, winsorization)
- Export transparent artifacts (CSV tables, plots, text report) for auditability

> Important: Outputs are **in-sample** unless walk-forward is enabled (planned/next). In-sample ranking is useful for debugging and iteration, not for decision-grade deployment.

---

## 1. Key Features (v0.2)

### Optimization
- **Max Sharpe** (maximize excess return per unit volatility)
- **Min Variance** (minimum variance portfolio)
- Multi-start optimization (random simplex starts) → **many candidate solutions**
- Candidate ranking (Sharpe-first by default)
- Pareto-efficient subset approximation (risk/return “efficient set”)

### Constraints & Regularization
- Long-only weights (optional)
- Per-asset `min_weight` / `max_weight`
- Concentration control:
  - Optional **HHI cap**: `sum(w^2) <= hhi_max`
- Max Sharpe stability:
  - Optional **L2 regularization**: `-Sharpe + λ * sum(w^2)`

### Estimation Layer
- Expected return (μ):
  - historical mean
  - EWMA mean
- Covariance (Σ):
  - sample covariance
  - EWMA covariance
  - shrinkage covariance (Ledoit–Wolf if available; fallback shrink-to-target)
- Optional **winsorization** (clip 1% tails) before estimating μ and Σ
- PSD enforcement for numerical stability

### Outputs (Reproducible Artifacts)
Saved in `output/` (or custom `--outdir`) with a run tag:
- Tables (CSV): μ, Σ, candidate set, top candidates, efficient set, portfolio returns, weights, summary
- Figures (PNG): frontier scatter, equity curves, drawdowns, weight bars
- Text report: run configuration + summary tables

---

## 2. Project Structure

```bash
portfolio-optimizer/
├─ optimize.py # CLI orchestrator
├─ portfolio/
│ ├─ init.py
│ ├─ data.py # yfinance ingestion + caching + returns utilities
│ ├─ estimation.py # μ/Σ estimation (EWMA, shrinkage, winsorization)
│ ├─ objectives.py # max Sharpe + L2 regularization, min variance
│ ├─ constraints.py # sum-to-1, HHI cap constraint
│ ├─ optimizer.py # multi-start optimization + ranking + efficient set
│ └─ report.py # output dirs + CSV/PNG/text writing
└─ output/ # generated artifacts (gitignored)
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
git clone https://github.com/KyryloPana/portfolio-optimizer.git
cd portfolio-optimizer
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

---

## 4. Quick Start

### Run max Sharpe + min variance, export top candidates
```bash
python optimize.py  --tickers 'AAPL,MSFT,NVDA,AMZN,SPY'  --start '2021-01-01'  --returns 'log'  --mean 'ewma'  --lam 0.97  --cov 'shrink'  --winsorize  --winsor-p 0.01  --objective 'both'  --rf 0.045  --long-only  --minw 0.0  --maxw 0.6  --hhi-max 0.30  --l2 0.10  --starts 400  --top 20  --outdir 'output'
```

---

## 5. CLI Usage

### Universe & time range

| Argument | Description |
|---|---|
| `--tickers` | **Required.** Comma-separated tickers. |
| `--start` | Start date (YYYY-MM-DD). |
| `--end` | End date (YYYY-MM-DD). Optional. |
| `--interval` | Data interval. Default: `1d`. |
| `--cache-days` | Cache TTL in days. |

### Return construction

| Argument | Description |
|---|---|
| `--returns {log,simple}` | Return type. Default: `log`. |
| `--annualization` | Annualization factor. Default: `252`. |

### Estimation

| Argument | Description |
|---|---|
| `--mean {historical,ewma}` | Mean estimator. |
| `--cov {sample,ewma,shrink}` | Covariance estimator. |
| `--lam` | EWMA lambda. Default: `0.94`. |
| `--winsorize` | Enable winsorization (flag). |
| `--winsor-p` | Winsorization percentile. Default: `0.01`. |

### Optimization

| Argument | Description |
|---|---|
| `--objective {max_sharpe,min_var,both}` | Optimization objective(s). Default: `both`. |
| `--rf` | Annual risk-free rate. |
| `--starts` | Multi-start count. |
| `--top` | Export top N candidates. |
| `--seed` | Random seed. |

### Constraints & stability

| Argument | Description |
|---|---|
| `--long-only` | Enforce long-only constraint (flag). |
| `--minw` | Minimum per-asset weight. |
| `--maxw` | Maximum per-asset weight. |
| `--hhi-max` | Optional concentration cap (HHI). |
| `--l2` | L2 regularization strength (for max Sharpe). |

### Output

| Argument | Description |
|---|---|
| `--outdir` | Output directory. |
| `--tag` | Optional run tag override. |
| `--show-plots` | Show interactive plots (flag). |

---

## Example
```bash
python optimize.py  --tickers 'AAPL,MSFT,NVDA,AMZN,SPY'  --start '2021-01-01'  --returns 'log'  --mean 'ewma'  --lam 0.97  --cov 'shrink'  --winsorize  --winsor-p 0.01  --objective 'both'  --rf 0.045  --long-only  --minw 0.0  --maxw 0.6  --hhi-max 0.30  --l2 0.10  --starts 400  --top 20  --outdir 'output'
```

***To see the full, authoritative CLI help:***
```bash
python optimize.py --help
```
---

## 6. Notes on Interpretation

### In-sample vs decision-grade

This tool currently ranks candidate portfolios using the same data used for estimation. This is useful for:

- debugging and sanity checks
- understanding constraint impacts
- iterating on estimators, constraints, and regularization

For decision-grade validation, the next step is walk-forward out-of-sample testing:

- rolling 252-day estimation window
- optimize
- apply weights for the next 21 days
- repeat
- compute OOS Sharpe, max drawdown, and turnover


## 7. Techniques and Mathematical Tools Used

This project implements a practical Modern Portfolio Theory (MPT) workflow: estimate expected returns and risk, then solve a constrained optimization problem to obtain feasible portfolios.

### 1) Data & Return Construction

**Price source:** yfinance (adjusted prices by default).

**Daily return series** can be constructed as:

- **Log returns:**  
  r_t = ln(P_t / P_{t-1})  
  Used by default due to additivity over time.

- **Simple returns:**  
  r_t = (P_t / P_{t-1}) - 1

**Portfolio returns (constant-weight / daily rebalanced):**  
r_p,t = Σ_i (w_i * r_i,t)  
This matches typical optimizer assumptions and avoids path-dependent “share” effects.

---

### 2) Estimation Layer (Inputs to Optimization)

#### Expected returns (mu)

The optimizer requires a vector of annualized expected returns:  
mu = E[r]

Supported estimators:

- **Historical mean (annualized):**  
  mu_daily = (1/T) * Σ_t r_t  
  mu_annual = mu_daily * A  
  where A is the annualization factor (default 252).

- **EWMA mean (exponentially weighted):**  
  mu_daily = Σ_t (omega_t * r_t)  
  omega_t ∝ (1 - lambda) * lambda^(T - t)  
  This reduces sensitivity to stale history by emphasizing recent observations.

#### Covariance matrix (Sigma)

Risk is modeled via the annualized covariance matrix:  
Sigma = Cov(r)

Supported estimators:

- **Sample covariance (annualized):**  
  Sigma_daily = (1/(T-1)) * Σ_t (r_t - mu)(r_t - mu)^T  
  Sigma_annual = Sigma_daily * A

- **EWMA covariance (RiskMetrics-style):**  
  Sigma_daily = Σ_t omega_t * (r_t - mu)(r_t - mu)^T  
  Sigma_annual = Sigma_daily * A

- **Shrinkage covariance (preferred in practice):**  
  Sigma_shrink = (1 - alpha) * Sigma_sample + alpha * Target  
  Target is typically a structured matrix (diagonal or scaled identity).  
  If available, Ledoit–Wolf shrinkage (scikit-learn) is used.

#### Winsorization (robust preprocessing)

To reduce the impact of extreme outliers (common in daily returns), returns can be winsorized per asset by clipping tails:

r_i,t = clip(r_i,t, q_i(p), q_i(1-p))

where q_i(p) is the p-quantile for asset i.  
Typical default: p = 0.01 (clip below 1% and above 99%).

#### PSD enforcement (numerical stability)

Optimizers require Sigma to behave like a valid covariance matrix (positive semidefinite).  
If numerical issues occur, Sigma is symmetrized and eigenvalues are clipped to a small floor, then a small ridge term is added for conditioning.

---

### 3) Optimization Problems Solved

#### A) Minimum Variance Portfolio (Global Minimum Variance)

Objective: minimize portfolio variance

min_w ( w^T * Sigma * w )

subject to constraints (see below).

#### B) Maximum Sharpe Ratio Portfolio (Return per Unit Risk)

Sharpe ratio:

S(w) = (w^T * mu - r_f) / sqrt(w^T * Sigma * w)

The solver maximizes Sharpe by minimizing the negative Sharpe:

min_w ( -S(w) )

Optionally with L2 regularization for stability:

min_w ( -S(w) + lambda * Σ_i w_i^2 )

This discourages fragile, concentrated solutions caused by noisy mu.

---

### 4) Constraints (Feasibility / Mandate Controls)

- **Full investment:** Σ_i w_i = 1
- **Bounds:** w_i ∈ [w_min, w_max] (and w_i >= 0 in long-only mode)
- **Optional concentration cap (HHI):**  
  HHI(w) = Σ_i w_i^2 <= HHI_max  
  Lower HHI implies more diversified allocations.

---

### 5) Numerical Optimization Method

Optimization uses **SciPy SLSQP** (Sequential Least Squares Quadratic Programming), which supports:
- equality constraints (e.g., sum-to-one),
- inequality constraints (e.g., HHI cap),
- bound constraints (min/max weights).

To reduce dependence on starting conditions, the tool uses **multi-start optimization**:
- generate many feasible starting weights (Dirichlet simplex samples),
- solve from each start,
- deduplicate and rank solutions.

---

### 6) Candidate Ranking & Efficient Set Approximation

Each candidate portfolio is evaluated by:
- expected return: w^T * mu
- volatility: sqrt(w^T * Sigma * w)
- Sharpe ratio

A Pareto-efficient subset (“efficient set”) is extracted by removing dominated portfolios (higher risk and lower return than another candidate).

---

### 7) Reporting Metrics

The reporting layer exports:
- annualized return
- annualized volatility
- Sharpe ratio (using user-supplied r_f)
- max drawdown (from cumulative equity curve)
- concentration proxy (HHI)
- full weights and portfolio return series for auditability

---

## 8. Roadmap

- Walk-forward OOS validation mode (`--walk-forward`)
- Turnover + transaction cost modeling
- Benchmark-relative constraints (tracking error)
- Factor risk model covariance (stability upgrade)
- Black–Litterman / return shrinkage models


## Disclaimer

This project is for research and educational purposes only.  
It does not constitute financial advice.
 
