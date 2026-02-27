
# Tech Stock Forecasting with GARCH Hybrids and Hierarchical Bayesian Models

## Overview

This repository contains a reproducible research framework for forecasting daily returns and volatility of large-cap technology stocks. The project integrates econometric volatility modeling, deep learning architectures, hybrid GARCH–neural network approaches, and hierarchical Bayesian partial pooling across assets.

The primary objective is to build a methodologically rigorous, publication-quality forecasting pipeline that:

* Respects the time-ordered structure of financial data
* Eliminates look-ahead bias and data leakage
* Enables fair model comparison under walk-forward evaluation
* Examines cross-asset generalization and uncertainty calibration
* Produces reproducible experimental artifacts

This repository is intended for educational and research purposes only. It does not constitute financial advice.

---

## Research Motivation

Financial return series exhibit:

* Weak predictability in conditional means
* Strong volatility clustering
* Heavy-tailed behavior
* Regime-dependent dynamics

Traditional econometric models (e.g., GARCH) capture volatility persistence but are typically applied on a per-asset basis. Deep learning models capture nonlinear temporal dependencies but often lack interpretability and principled uncertainty estimation.

This project investigates:

1. Whether hybrid GARCH–deep learning architectures improve predictive stability.
2. Whether hierarchical Bayesian partial pooling improves cross-asset generalization.
3. How different modeling paradigms compare under strict time-based evaluation.

---

## Core Contributions

This framework enables systematic comparison of:

* **GARCH family models** (GARCH, EGARCH, GJR-GARCH)
* **Deep sequence models** (GRU, LSTM)
* **Hybrid models** (GARCH-feature + GRU/LSTM)
* **Hierarchical Bayesian models** with partial pooling across stocks

All models are evaluated using consistent walk-forward testing.

---

## Methodological Standards

This project follows strict research principles:

* No random shuffling of time-series data
* All predictors are properly lagged
* Train/validation/test splits are defined by target date
* Raw data is cached locally for reproducibility
* Experiments are configuration-driven
* Model outputs are stored as versioned artifacts

These standards ensure experimental credibility and reproducibility.

---

## Repository Structure

```
.
├── data/
│   ├── raw/            # Cached raw data downloads
│   └── processed/      # Cleaned panel dataset and splits
├── notebooks/          # Step-by-step research notebooks
├── src/
│   ├── data/           # Data download and feature engineering
│   ├── models/         # GARCH, deep learning, hybrid, and Bayesian models
│   └── eval/           # Walk-forward evaluation and metrics
├── reports/
│   ├── figures/        # Saved plots for analysis and paper
│   └── tables/         # Saved result tables (CSV/LaTeX)
├── configs/            # Experiment configuration files
└── paper/              # Paper outline and manuscript materials
```

---

## Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

Core dependencies:

```bash
pip install -r requirements.txt
```

Deep learning:

```bash
pip install -r requirements-deep.txt
```

Bayesian modeling:

```bash
pip install -r requirements-bayes.txt
```

---

## Data Pipeline

### Download raw data

```bash
python -m src.data.download --config configs/default.yaml
```

This downloads:

* Adjusted close prices for selected technology stocks
* Market proxy (SPY)
* VIX index
* Macro series from FRED

All data is saved under `data/raw/`.

---

## Build the panel dataset

```bash
python -m src.data.build_panel --config configs/default.yaml
```

This produces:

* `data/processed/panel.parquet`
* `data/processed/audit_summary.csv`

The dataset is structured as:

Date × Ticker × Features × Target

Targets are next-day log returns.
All features are constructed to avoid look-ahead leakage.

---

## Modeling Framework

### Econometric Volatility Models

* GARCH(1,1)
* EGARCH
* GJR-GARCH
* Student-t innovations (where applicable)

These models capture volatility clustering and asymmetric shock responses.

---

### Deep Learning Models

* GRU
* LSTM

These architectures model nonlinear temporal dependencies in returns.

---

### Hybrid Models

Hybrid architectures integrate econometric volatility structure into neural models, including:

* GARCH-derived volatility as input features
* Mean–variance decomposition approaches (RNN mean + GARCH variance)

---

### Hierarchical Bayesian Models

Hierarchical models introduce partial pooling across assets:

* Global parameters capture shared sector behavior
* Asset-specific parameters capture idiosyncratic deviations
* Borrowing strength improves generalization and stability
* Provides interpretable parameter structure

---

## Evaluation Strategy

All models are evaluated using:

* Walk-forward (rolling-origin) testing
* Time-respecting splits
* Out-of-sample metrics:

  * MAE
  * RMSE
  * Out-of-sample R²
* Volatility-specific metrics where applicable
* Predictive interval coverage and log scores for probabilistic models

Optional statistical comparisons may include Diebold–Mariano tests and regime-based analysis.

---

## Project Roadmap

Phase 1 — Data engineering and panel construction
Phase 2 — GARCH family volatility modeling
Phase 3 — Deep learning baselines (GRU/LSTM)
Phase 4 — Hybrid GARCH–RNN models
Phase 5 — Hierarchical Bayesian partial pooling
Phase 6 — Walk-forward evaluation and comparative analysis
Phase 7 — Paper drafting and reproducibility validation

---

## Reproducibility Checklist

* Deterministic time splits defined in configuration
* All raw data stored locally after download
* No future data used for feature construction
* All model outputs saved to `reports/`
* Configuration files version-controlled

---

## License

Add a license file before public distribution. The MIT License is recommended for open research repositories.

---

## Disclaimer

This repository is intended for academic and educational research.
It does not provide investment advice or trading recommendations.

