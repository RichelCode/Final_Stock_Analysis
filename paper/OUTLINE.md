# Manuscript plan

Working title:

> **Where the Predictability Is: Conditional Variance, Not Conditional Mean,
> Delivers Calibrated Daily Return Forecasts for Large-Cap Technology Stocks,
> 2016–2025**

Every number below is produced by a script in `src/` and written to
`stock_project/reports/`. Nothing is quoted from the v1 notebooks.

---

## The argument in five steps

1. Under a fixed macro-financial information set, **no model class improves on a
   per-ticker mean baseline** for the conditional mean. Zero of 48 model-ticker
   Diebold–Mariano tests survive multiplicity correction, and the count of
   uncorrected rejections is below what chance produces.
2. The **conditional variance is predictable**. All three GARCH-family
   specifications beat a rolling-variance benchmark under QLIKE at p < 0.001.
   Asymmetric extensions add nothing detectable.
3. **Calibration comes from the variance model.** Moving from constant to GARCH
   variance is worth 0.153 nats of log score (p < 0.00001); adding any
   conditional mean on top is worth 0.002 nats (p = 0.06).
4. **Fat tails matter, and they hid a scale error.** Student-t innovations more
   than halve the PIT KS statistic. They also reveal that GARCH parameters
   frozen at end-2019 understate 2023–2025 volatility by 12%, which validation
   data corrects.
5. **A methodological warning with a controlled demonstration.** Standardising
   non-stationary predictors on a fixed origin fails silently and
   catastrophically across a regime break.

---

## Section plan

| § | Title | Source |
|---|---|---|
| 1 | Introduction | existing draft, retarget from AAPL to the panel |
| 2 | Literature review | existing draft + distribution shift, Patton (2011) |
| 3.1 | Data and splits | `build_panel_v2.py` |
| 3.2 | Feature dictionary and stationarity | `features/dictionary.py`, ADF table |
| 3.3 | Evaluation protocols | `eval/protocol.py` |
| 3.4 | Volatility models | `models/garch.py` |
| 3.5 | Hierarchical Bayesian model | `models/hierarchical_bayes.py` |
| 3.6 | Scoring rules and tests | `eval/metrics.py` |
| 4.1 | The conditional mean | Table 2, Table 3 |
| 4.2 | The conditional variance | Table 4 |
| 4.3 | Decomposing the probabilistic gain | Figure 5 |
| 4.4 | Calibration | Figures 2 and 3 |
| 4.5 | Cross-asset transfer | Table 6 |
| 4.6 | Fixed origin vs expanding window | Figure 1 |
| 5 | Conclusion and recommendations | — |

## Tables

| # | Content | File |
|---|---|---|
| 1 | Data, splits, feature dictionary | `feature_dictionary_v2.csv` |
| 2 | Per-ticker OOS R², all models | `v2_per_ticker_oos_r2_fixed.csv` |
| 3 | Pooled point accuracy with DM p-values | `v2_linear_tree_metrics.csv`, `v2_neural_metrics_base.csv` |
| 4 | Volatility QLIKE with pairwise DM | `v2_volatility_qlike_test.csv`, `v2_volatility_dm_test.csv` |
| 5 | Probabilistic scores and coverage | `v2_probabilistic_comparison_t_val_iqr.csv` |
| 6 | LOTO and k-shot transfer | `v2_transfer_kshot.csv` |
| A1 | GARCH fit summary | `v2_garch_fit_summary.csv` |
| A2 | Posterior convergence | `v2_bayes_convergence_t.csv` |

## Figures

| # | Content | File |
|---|---|---|
| 1 | Predictor drift and its repair | `fig1_extrapolation.pdf` |
| 2 | Reliability diagram | `fig2_reliability.pdf` |
| 3 | PIT histograms with KS tests | `fig3_pit.pdf` |
| 4 | Horseshoe shrinkage by predictor | `fig4_shrinkage.pdf` |
| 5 | Decomposition of the log score | `fig5_decomposition.pdf` |

---

## Limitations to state plainly

- Student-t calibration is **improved, not achieved**: PIT uniformity is still
  rejected at p = 0.012. Report it as such.
- GARCH parameters are estimated once on the training split. Rolling
  re-estimation is the obvious extension and is not done here.
- Eight tickers in one sector over one decade. The between-ticker variance is
  identified by eight groups, which is few, and the paper should say so.
- The realised-variance proxy is the squared daily return. QLIKE is robust to
  proxy noise (Patton 2011) but intraday data would be better.
- The scale calibration factor is one scalar estimated on validation data. A
  time-varying calibration is untested.

## What this paper does not claim

- That deep learning fails. On the corrected feature set the sequence models
  land where the linear models land; the v1 failure was a feature-scaling
  artefact and is reported as such.
- That macroeconomic variables are useless. One predictor, the daily change in
  the effective federal funds rate, survives horseshoe shrinkage with a credible
  interval excluding zero. Its economic magnitude is small.
- That the intervals are perfectly calibrated. See limitations.
