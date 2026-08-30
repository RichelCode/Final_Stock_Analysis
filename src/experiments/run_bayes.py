"""
Fit the hierarchical Bayesian model on panel_v2_garch and score its forecasts.

Design
------
The GARCH(1,1)-t conditional SD enters as the known observation scale, not as a
regressor, so the mean equation and the variance equation stay separate and the
horseshoe operates only on genuine mean predictors. GARCH(1,1)-t is used rather
than EGARCH or GJR because it wins the QLIKE comparison in
src.experiments.run_volatility.

Standardisation uses training moments only, and the model is fitted once on the
training split (fixed origin). This is safe here in a way it was not in v1
because every predictor is now stationary.

Competitors scored on the identical sample
------------------------------------------
  bayes_hier_horseshoe   this model
  ridge_garch            ridge mean, GARCH scale calibrated on train
  baseline_garch         zero mean, GARCH scale calibrated on train
  baseline_gaussian      zero mean, constant train SD

The ridge+GARCH benchmark matters: in v1 it beat the Bayesian model on log
score (2.333 against 2.233) while the headline table reported only the
Bayesian row. Any claim made for the Bayesian model has to clear it.

Usage
-----
    python -m src.experiments.run_bayes
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from src.eval.diagnostics import convergence_report, convergence_table, print_report
from src.eval.metrics import (crps_gaussian, diebold_mariano, log_score, oos_r2,
                              point_metrics, probabilistic_metrics)
from src.features.dictionary import TARGET, model_features
from src.models.hierarchical_bayes import fit, inclusion_summary, predict_seen

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJ = REPO_ROOT / "stock_project"
DEFAULT_PANEL = PROJ / "data" / "processed" / "panel_v2_garch.parquet"
PRED_DIR = PROJ / "reports" / "predictions_v2"
TAB_DIR = PROJ / "reports" / "tables"

SIGMA_COL = "garch11_t_sigma1"
SEED = 20260830
ID_COLS = ["Date", "target_date", "Ticker"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--iter", type=int, default=40000)
    ap.add_argument("--burn", type=int, default=20000)
    # The global horseshoe scale tau_b mixes slowly when every
    # coefficient is near zero, which is the situation here. Thinning
    # keeps the retained draws close to independent.
    ap.add_argument("--thin", type=int, default=4)
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel)
    features = model_features(include_garch=False)

    df = panel.dropna(subset=features + [TARGET, SIGMA_COL, "split"]).copy()
    tickers = sorted(df["Ticker"].unique())
    df["g"] = df["Ticker"].map({t: i for i, t in enumerate(tickers)}).astype(int)

    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"]
    print(f"Features ({len(features)}): {', '.join(features)}")
    print(f"Train {len(tr)} rows | Test {len(te)} rows | {len(tickers)} tickers")
    print(f"Scale: {SIGMA_COL}\n")

    mu_x, sd_x = tr[features].mean(), tr[features].std().replace(0.0, 1.0)
    Xtr = ((tr[features] - mu_x) / sd_x).to_numpy(float)
    Xte = ((te[features] - mu_x) / sd_x).to_numpy(float)
    ytr, yte = tr[TARGET].to_numpy(float), te[TARGET].to_numpy(float)
    gtr, gte = tr["g"].to_numpy(int), te["g"].to_numpy(int)
    str_, ste = tr[SIGMA_COL].to_numpy(float), te[SIGMA_COL].to_numpy(float)

    t0 = time.time()
    post = fit(Xtr, ytr, gtr, str_, len(tickers),
               n_chains=args.chains, n_iter=args.iter, burn=args.burn,
               thin=args.thin, seed=SEED)
    print(f"Sampled {post.beta.shape[0]} draws "
          f"({post.n_chains} x {post.n_draws_per_chain}) in {time.time()-t0:.0f}s\n")

    print("Convergence:")
    conv = convergence_table({k: post.by_chain(k) for k in
                              ["beta", "alpha", "mu_alpha", "tau_a2", "s2", "tau_b2"]})
    report = convergence_report(conv)
    print_report(report)
    conv.to_csv(TAB_DIR / "v2_bayes_convergence.csv", index=False)

    inc = inclusion_summary(post, features)
    inc.to_csv(TAB_DIR / "v2_bayes_inclusion.csv", index=False)
    print("\nHorseshoe shrinkage (1 - kappa near 1 means the data overrode the prior):")
    print(inc[["feature", "beta_mean", "beta_q2.5", "beta_q97.5",
               "excludes_zero", "weight_1_minus_kappa"]].round(5).to_string(index=False))

    # ---- competitors, all on the identical test sample ---------------------
    mu_b, sd_b, _ = predict_seen(Xte, gte, ste, post)

    ridge = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(Xtr, ytr)
    mu_r = ridge.predict(Xte)
    # Calibrate the GARCH scale on training residuals: s = sd(resid / sigma).
    s_ridge = float(np.std((ytr - ridge.predict(Xtr)) / str_))
    sd_r = s_ridge * ste

    s_zero = float(np.std(ytr / str_))
    zero_sd = s_zero * ste
    const_sd = np.full(len(yte), float(ytr.std()))

    ref_mean = np.full(len(yte), float(ytr.mean()))

    entries = {
        "bayes_hier_horseshoe": (mu_b, sd_b),
        "ridge_garch": (mu_r, sd_r),
        "baseline_garch": (np.zeros(len(yte)), zero_sd),
        "baseline_gaussian": (np.zeros(len(yte)), const_sd),
    }

    rows = []
    for name, (mu, sd) in entries.items():
        rows.append({
            "model": name,
            **point_metrics(yte, mu),
            "OOS_R2_vs_train_mean": oos_r2(yte, mu, ref_mean),
            **{k: v for k, v in probabilistic_metrics(yte, mu, sd).items() if k != "n"},
        })
    results = pd.DataFrame(rows)

    # Significance on the log-score differential against the Bayesian model.
    ls = {n: log_score(yte, mu, sd) for n, (mu, sd) in entries.items()}
    dm_rows = []
    for name in entries:
        if name == "bayes_hier_horseshoe":
            continue
        # DM on negative log score, so a lower loss is better and a negative
        # statistic favours the Bayesian model.
        r = diebold_mariano(-ls["bayes_hier_horseshoe"], -ls[name])
        dm_rows.append({"model_A": "bayes_hier_horseshoe", "model_B": name,
                        "logscore_A": ls["bayes_hier_horseshoe"].mean(),
                        "logscore_B": ls[name].mean(),
                        "DM_stat": r["DM_stat"], "p_value": r["p_value"],
                        "significant_5pct": bool(r["p_value"] < 0.05)})
    dm = pd.DataFrame(dm_rows)

    show = ["model", "RMSE", "OOS_R2_vs_train_mean", "logscore", "CRPS",
            "coverage_50", "coverage_80", "coverage_90", "coverage_95"]
    with pd.option_context("display.width", 220):
        print(f"\nTest-window comparison, n = {len(yte)}:\n")
        print(results[show].round(5).to_string(index=False))
        print("\nDiebold-Mariano on log score, Bayesian vs each competitor:\n")
        print(dm.round(5).to_string(index=False))

    pred = te[ID_COLS].copy()
    pred["split"] = "test"
    pred["model"] = "bayes_hier_horseshoe"
    pred["y_true"] = yte
    pred["y_pred"] = mu_b
    pred["y_sd"] = sd_b
    pred["residual"] = yte - mu_b
    pred["logscore"] = ls["bayes_hier_horseshoe"]
    pred["crps"] = crps_gaussian(yte, mu_b, sd_b)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(PRED_DIR / "bayes_hier_horseshoe__fixed.parquet", index=False)

    results.to_csv(TAB_DIR / "v2_probabilistic_comparison.csv", index=False)
    dm.to_csv(TAB_DIR / "v2_probabilistic_dm.csv", index=False)
    print(f"\nSaved predictions and 4 tables.")
    return 0 if report["rhat_ok"] and report["ess_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
