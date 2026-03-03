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
**Price source:** `yfinance` (adjusted prices by default).  
**Return series:** daily returns are constructed as either:
- **Log returns:**  
  \[
  r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
  \]
  Used by default due to additivity over time.
- **Simple returns:**  
  \[
  r_t = \frac{P_t}{P_{t-1}} - 1
  \]

**Portfolio returns (constant-weight / daily rebalanced):**
\[
r_{p,t} = \sum_{i=1}^N w_i \, r_{i,t}
\]
This matches typical optimizer assumptions and avoids path-dependent “share” effects.

---

### 2) Estimation Layer (Inputs to Optimization)

#### Expected returns \(\mu\)
The optimizer requires a vector of annualized expected returns:
\[
\mu = \mathbb{E}[r]
\]
This tool supports:

- **Historical mean:**
  \[
  \hat{\mu}_{\text{daily}} = \frac{1}{T}\sum_{t=1}^T r_t, 
  \quad \hat{\mu}_{\text{annual}} = \hat{\mu}_{\text{daily}} \cdot A
  \]
  where \(A\) is the annualization factor (default \(252\)).

- **EWMA mean (exponentially weighted):**
  \[
  \hat{\mu}_{\text{daily}} = \sum_{t=1}^T \omega_t r_t,
  \quad \omega_t \propto (1-\lambda)\lambda^{T-t}
  \]
  This reduces sensitivity to stale history by emphasizing recent observations.

#### Covariance matrix \(\Sigma\)
Risk is modeled via the annualized covariance matrix:
\[
\Sigma = \operatorname{Cov}(r)
\]
Supported estimators:

- **Sample covariance:**
  \[
  \hat{\Sigma}_{\text{daily}} = \frac{1}{T-1}\sum_{t=1}^T (r_t-\hat{\mu})(r_t-\hat{\mu})^\top,
  \quad \hat{\Sigma}_{\text{annual}} = \hat{\Sigma}_{\text{daily}}\cdot A
  \]

- **EWMA covariance (RiskMetrics-style):**
  \[
  \hat{\Sigma}_{\text{daily}} = \sum_{t=1}^T \omega_t (r_t-\hat{\mu})(r_t-\hat{\mu})^\top
  \]

- **Shrinkage covariance (preferred in practice):**
  \[
  \hat{\Sigma}_{\text{shrink}} = (1-\alpha)\hat{\Sigma}_{\text{sample}} + \alpha T
  \]
  where \(T\) is a structured target (e.g., diagonal or identity-scaled), and \(\alpha\in[0,1]\) controls shrinkage.
  If available, **Ledoit–Wolf** shrinkage (via scikit-learn) is used.

#### Winsorization (robust preprocessing)
To reduce the impact of extreme outliers (common in daily returns), returns can be **winsorized** column-wise by clipping tails:
\[
r_{i,t} \leftarrow \min\left(\max(r_{i,t}, q_i(p)),\, q_i(1-p)\right)
\]
where \(q_i(p)\) is the \(p\)-quantile of asset \(i\)’s returns. Typical default: \(p=0.01\) (1% / 99%).

#### PSD enforcement (numerical stability)
Optimizers require \(\Sigma\) to behave like a valid covariance matrix (positive semidefinite).  
If numerical issues occur, the matrix is symmetrized and eigenvalues are clipped:
\[
\Sigma \leftarrow Q \max(\Lambda, \epsilon I) Q^\top
\]
with a small ridge term added to improve conditioning.

---

### 3) Optimization Problems Solved

#### A) Minimum Variance Portfolio (global minimum variance)
\[
\min_{w} \quad w^\top \Sigma w
\]
subject to constraints (see below).

#### B) Maximum Sharpe Ratio Portfolio
Sharpe ratio:
\[
S(w) = \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}}
\]
The solver minimizes the negative Sharpe (equivalent to maximizing Sharpe):
\[
\min_{w} \quad -S(w)
\]
Optionally with **L2 regularization** for stability:
\[
\min_{w} \quad -S(w) + \lambda \sum_{i=1}^N w_i^2
\]
This discourages fragile, concentrated solutions caused by noisy \(\mu\).

---

### 4) Constraints (Feasibility / Mandate Controls)

Common constraints:
- **Full investment:**
  \[
  \sum_{i=1}^N w_i = 1
  \]
- **Bounds (long-only and position limits):**
  \[
  w_i \in [w_{\min}, w_{\max}]
  \]
  In long-only mode, \(w_i \ge 0\).

Optional concentration constraint:
- **HHI cap (Herfindahl–Hirschman Index):**
  \[
  \text{HHI}(w)=\sum_{i=1}^N w_i^2 \le \text{HHI}_{\max}
  \]
Lower HHI implies more diversified allocations.

---

### 5) Numerical Optimization Method
Optimization is performed with **SciPy SLSQP** (Sequential Least Squares Quadratic Programming), which supports:
- equality constraints (e.g., sum-to-one),
- inequality constraints (e.g., HHI cap),
- bound constraints (min/max weights).

To reduce dependence on local minima and starting conditions, the tool uses **multi-start optimization**:
- random feasible initial weights (Dirichlet simplex samples),
- solve the optimization from many starts,
- deduplicate and rank solutions.

---

### 6) Candidate Ranking & Efficient Set Approximation

Each candidate portfolio is evaluated by:
- expected return \(w^\top \mu\),
- volatility \(\sqrt{w^\top\Sigma w}\),
- Sharpe ratio.

A Pareto-efficient subset (“efficient set”) is extracted by discarding dominated portfolios (higher risk and lower return than another candidate).

---

### 7) Risk & Performance Metrics (Reporting)
The reporting layer exports:
- annualized return (based on configured return type and annualization factor),
- annualized volatility,
- Sharpe ratio (using user-supplied \(r_f\)),
- max drawdown (from cumulative equity curve),
- concentration proxy (HHI),
- full weights and portfolio return series for auditability.

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
 
