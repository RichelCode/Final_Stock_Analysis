# Forecasting Daily Returns of Large-Cap Technology Stocks (2016–2025)

Where is the predictability in daily equity returns — in the conditional mean,
or in the conditional variance?

This repository holds the data pipeline, models and evaluation code for a study
of eight large-cap US technology stocks over 2016–2025, spanning the COVID-19
dislocation, the 2022 tightening cycle and the subsequent normalisation. It
compares baseline, linear, tree, econometric and Bayesian forecasts of one-day-
ahead returns, scoring them on point accuracy, probabilistic accuracy and
interval calibration, with every headline comparison backed by a
Diebold–Mariano test.

For research and education only. Nothing here is investment advice.

---

## Headline findings

**The conditional mean is not predictable at this horizon.** No model —
ridge, random forest, or hierarchical Bayes — beats a per-ticker mean baseline
on any of the eight stocks. Out-of-sample R² against that baseline is −0.001 for
ridge and −0.003 for the random forest, and Diebold–Mariano cannot distinguish
either from the baseline (p = 0.36 and p = 0.18).

**The conditional variance is predictable, and it is where the value is.**
All three GARCH-family specifications beat a 20-day rolling-variance benchmark
under QLIKE at p < 0.001. Plain GARCH(1,1)-t wins; the asymmetric extensions add
nothing detectable (GARCH vs EGARCH, p = 0.33).

**Calibration comes from the variance model, not the mean model.** Moving from
a constant-variance Gaussian predictive to a GARCH-scaled one is worth 0.155
nats of log score (p < 0.00001). Adding *any* conditional mean on top of that is
worth 0.002 nats and is not statistically detectable (p = 0.06). A zero-mean
forecast with a GARCH scale is as good as the full hierarchical Bayesian model.

**Non-stationary predictors fail silently across a regime break.** See below.

---

## The extrapolation result

An earlier version of this pipeline selected model inputs by dtype, which
admitted three non-stationary level series (the S&P 500 index level, the
effective federal funds rate, the 10-year yield). Standardised on 2016–2019
moments and applied to 2023–2025, the S&P 500 level arrives at **z = +8.7**,
with **100%** of test observations beyond three training standard deviations.

Every model with an unbounded output head extrapolated along that axis. The
LSTM's test predictions correlate with the S&P 500 level at r = 0.88 and imply a
+12% return every trading day for three years. Training error looked normal
throughout and no exception was raised.

Replacing the levels with their stationary counterparts moves ridge from an
out-of-sample R² of −0.044 to −0.001, and the random forest from −0.188 to
−0.003. The failure and its repair are reported as a result, not hidden:
`src/diagnostics/` contains the two audits, and their before/after outputs are
committed under `stock_project/reports/tables/`.

---

## Repository layout

```
src/
  data/build_panel_v2.py        panel construction from the declared feature set
  features/dictionary.py        the explicit feature dictionary (see below)
  models/garch.py               GARCH/GJR/EGARCH estimation with validation
  models/hierarchical_bayes.py  horseshoe hierarchical Bayes sampler
  eval/metrics.py               point, probabilistic and DM scoring
  eval/protocol.py              fixed-origin and expanding-window protocols
  eval/diagnostics.py           R-hat and ESS reporting
  diagnostics/feature_drift.py      train-to-test drift audit
  diagnostics/prediction_sanity.py  extrapolation audit on saved predictions
  experiments/                  runnable experiments, one per model family

stock_project/
  data/processed/               panels (parquet)
  reports/predictions_v2/       per-model predictions
  reports/tables/               metrics, diagnostics and paper tables

deep_analysis.ipynb             original exploratory notebooks, kept for
stock_market_final.ipynb        provenance; superseded by src/
```

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then, in order:

```bash
python -m src.diagnostics.feature_drift               # audit the v1 panel
python -m src.data.build_panel_v2                     # build the corrected panel
python -m src.models.garch                            # re-estimate volatility features
python -m src.experiments.run_baselines               # zero, ticker-mean, AR(1)
python -m src.experiments.run_linear_tree             # ridge, random forest
python -m src.experiments.run_volatility              # QLIKE comparison + DM tests
python -m src.experiments.run_bayes                   # hierarchical Bayes
```

Both diagnostics exit non-zero when they find a problem, so they can gate a
pipeline. `src.models.garch` aborts rather than writing a column if any fit
fails to converge.

---

## Design decisions worth knowing

**Features are declared, not inferred.** `src/features/dictionary.py` lists
every model input by hand with its description, source, transform, the time at
which it is observable, and a stationarity flag backed by an ADF test. Excluded
candidates stay in the dictionary with the reason for exclusion. A column is a
feature because it is listed, never because it happens to be numeric.

**Two evaluation protocols, deliberately.** *Fixed origin* fits once on the
training split and freezes everything. *Expanding window* refits monthly on all
data whose `target_date` precedes the fold. The gap between them separates real
forecast error from the artefact of freezing a scaler across a regime change.
Verified: across all 36 folds no training row reaches into the month being
scored, and both protocols score the identical 6,008 observations.

**GARCH parameters are estimated on training data only**, then held fixed and
filtered forward, giving genuine one-step-ahead σ(t+1|t). The value stored at
date *t* forecasts the same period as the target at date *t*.

**Selection happens inside the posterior.** The Bayesian model places a
horseshoe prior on the coefficients rather than pre-screening predictors by
correlation with the target. Validated on synthetic data: 3 of 3 true signals
recovered, 0 of 9 false positives.

---

## Splits

| Split | Dates | Rows |
|---|---|---|
| Train | 2016-03-02 → 2019-12-30 | 7,720 |
| Validation | 2019-12-31 → 2022-12-29 | 6,048 |
| Test | 2022-12-30 → 2025-12-29 | 6,008 |

Tickers: AAPL, MSFT, AMZN, GOOGL, META, NVDA, TSLA, ORCL.
Sources: Yahoo Finance (prices), FRED (DFF, DGS10, SP500, VIX).

---

## Status

Complete: data pipeline, diagnostics, baselines, linear and tree models,
GARCH family, hierarchical Bayes with convergence diagnostics.

In progress: Student-t innovations, leave-one-ticker-out transfer, per-ticker
result tables, calibration figures, neural network re-runs.

---

## License

MIT. See `LICENSE`.
