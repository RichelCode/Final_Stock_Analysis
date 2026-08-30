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
from src.eval.metrics import (crps_gaussian, crps_t, diebold_mariano, log_score,
                              log_score_t, oos_r2, point_metrics,
                              probabilistic_metrics, probabilistic_metrics_t)
from src.features.dictionary import TARGET, model_features
from src.models.hierarchical_bayes import (fit, inclusion_summary, predict_seen,
                                           predictive_df)

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
    ap.add_argument("--dist", choices=["normal", "t"], default="t",
                    help="Innovation distribution for the observation equation.")
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
               thin=args.thin, seed=SEED, dist=args.dist)
    print(f"Sampled {post.beta.shape[0]} draws "
          f"({post.n_chains} x {post.n_draws_per_chain}) in {time.time()-t0:.0f}s\n")

    print("Convergence:")
    monitored = ["beta", "alpha", "mu_alpha", "tau_a2", "s2", "tau_b2"]
    if post.nu is not None:
        monitored.append("nu")
    conv = convergence_table({k: post.by_chain(k) for k in monitored})
    report = convergence_report(conv)
    print_report(report)
    conv.to_csv(TAB_DIR / f"v2_bayes_convergence_{args.dist}.csv", index=False)
    if post.nu is not None:
        print(f"\n  Student-t degrees of freedom: posterior mean {post.nu.mean():.2f}, "
              f"median {np.median(post.nu):.1f}")

    inc = inclusion_summary(post, features)
    inc.to_csv(TAB_DIR / f"v2_bayes_inclusion_{args.dist}.csv", index=False)
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

    # df = inf denotes a Gaussian predictive. Only the Bayesian model carries a
    # posterior for the degrees of freedom; the benchmarks stay Gaussian so the
    # comparison isolates what the model itself contributes.
    # predict_seen already returns the scale appropriate to the fitted
    # innovation distribution, so no conversion is needed here.
    df_b = predictive_df(post)
    scale_b = sd_b

    entries = {
        "bayes_hier_horseshoe": (mu_b, scale_b, df_b),
        "ridge_garch": (mu_r, sd_r, np.inf),
        "baseline_garch": (np.zeros(len(yte)), zero_sd, np.inf),
        "baseline_gaussian": (np.zeros(len(yte)), const_sd, np.inf),
    }

    def score(y, mu, scale, df):
        if np.isfinite(df):
            return probabilistic_metrics_t(y, mu, scale, df)
        return probabilistic_metrics(y, mu, scale)

    def pointwise_logscore(y, mu, scale, df):
        if np.isfinite(df):
            return log_score_t(y, mu, scale, df)
        return log_score(y, mu, scale)

    rows = []
    for name, (mu, scale, df) in entries.items():
        m = score(yte, mu, scale, df)
        rows.append({
            "model": name,
            "innovations": "t" if np.isfinite(df) else "gaussian",
            "df": df,
            **point_metrics(yte, mu),
            "OOS_R2_vs_train_mean": oos_r2(yte, mu, ref_mean),
            **{k: v for k, v in m.items() if k not in ("n", "df")},
        })
    results = pd.DataFrame(rows)

    ls = {n: pointwise_logscore(yte, mu, sc, df) for n, (mu, sc, df) in entries.items()}
    dm_rows = []
    for name in entries:
        if name == "bayes_hier_horseshoe":
            continue
        # DM on negative log score, so lower is better and a negative statistic
        # favours the Bayesian model.
        r = diebold_mariano(-ls["bayes_hier_horseshoe"], -ls[name])
        dm_rows.append({"model_A": "bayes_hier_horseshoe", "model_B": name,
                        "logscore_A": ls["bayes_hier_horseshoe"].mean(),
                        "logscore_B": ls[name].mean(),
                        "DM_stat": r["DM_stat"], "p_value": r["p_value"],
                        "significant_5pct": bool(r["p_value"] < 0.05)})
    dm = pd.DataFrame(dm_rows)

    show = ["model", "innovations", "RMSE", "OOS_R2_vs_train_mean", "logscore", "CRPS",
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
    pred["y_scale"] = scale_b
    pred["df"] = df_b
    pred["residual"] = yte - mu_b
    pred["logscore"] = ls["bayes_hier_horseshoe"]
    pred["crps"] = (crps_t(yte, mu_b, scale_b, df_b) if np.isfinite(df_b)
                    else crps_gaussian(yte, mu_b, scale_b))
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(PRED_DIR / f"bayes_hier_horseshoe_{args.dist}__fixed.parquet",
                    index=False)

    results.to_csv(TAB_DIR / f"v2_probabilistic_comparison_{args.dist}.csv", index=False)
    dm.to_csv(TAB_DIR / f"v2_probabilistic_dm_{args.dist}.csv", index=False)
    print(f"\nSaved predictions and 4 tables.")
    return 0 if report["rhat_ok"] and report["ess_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
