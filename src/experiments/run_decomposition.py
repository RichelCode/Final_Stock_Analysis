"""
Decompose the probabilistic gain into its sources, with all pairwise tests.

This is the paper's central result. A predictive distribution for daily returns
has a location and a scale, and the question is which of the two carries the
value. The ladder isolates one addition at a time:

  1. zero mean, constant variance        the null predictive
  2. zero mean, GARCH variance           adds the conditional variance
  3. ridge mean, GARCH variance          adds a conditional mean
  4. Bayesian mean, t tails, calibrated  adds fat tails and scale calibration

Every rung is scored on the identical test sample, and every adjacent pair is
tested by Diebold-Mariano on the log-score differential.

A note on interpretation. With 6008 observations the test detects differences
far smaller than any that would matter in practice, so statistical significance
and economic relevance have to be reported separately. Each increment is
therefore expressed both as a p-value and as a share of the total gain.

Usage
-----
    python -m src.experiments.run_decomposition
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from src.eval.metrics import (coverage, coverage_t, crps_gaussian, crps_t,
                              diebold_mariano, log_score, log_score_t)
from src.features.dictionary import TARGET, model_features

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJ = REPO_ROOT / "stock_project"
PANEL = PROJ / "data" / "processed" / "panel_v2_garch.parquet"
PRED = PROJ / "reports" / "predictions_v2"
TAB = PROJ / "reports" / "tables"
SIGMA = "garch11_t_sigma1"
LEVELS = (0.50, 0.80, 0.90, 0.95)


def main() -> int:
    features = model_features(include_garch=False)
    panel = pd.read_parquet(PANEL).dropna(subset=features + [TARGET, SIGMA, "split"])
    tr, te = panel[panel.split == "train"], panel[panel.split == "test"]

    mu_x, sd_x = tr[features].mean(), tr[features].std().replace(0.0, 1.0)
    Xtr = ((tr[features] - mu_x) / sd_x).to_numpy(float)
    Xte = ((te[features] - mu_x) / sd_x).to_numpy(float)
    ytr, yte = tr[TARGET].to_numpy(float), te[TARGET].to_numpy(float)
    str_, ste = tr[SIGMA].to_numpy(float), te[SIGMA].to_numpy(float)

    ridge = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(Xtr, ytr)
    s_ridge = float(np.std((ytr - ridge.predict(Xtr)) / str_))
    s_zero = float(np.std(ytr / str_))

    bayes = pd.read_parquet(PRED / "bayes_hier_horseshoe_t__fixed.parquet")
    # Align to this script's test ordering rather than assuming it matches.
    key = pd.MultiIndex.from_frame(te[["Date", "Ticker"]])
    b = bayes.set_index(pd.MultiIndex.from_frame(bayes[["Date", "Ticker"]])).reindex(key)
    df_b = float(b["df"].iloc[0])

    rungs = {
        "1. zero mean, constant variance": (
            np.zeros(len(yte)), np.full(len(yte), float(ytr.std())), np.inf),
        "2. zero mean, GARCH variance": (
            np.zeros(len(yte)), s_zero * ste, np.inf),
        "3. ridge mean, GARCH variance": (
            ridge.predict(Xte), s_ridge * ste, np.inf),
        "4. Bayesian mean, t tails, calibrated": (
            b["y_pred"].to_numpy(float), b["y_scale"].to_numpy(float), df_b),
    }

    def ls(y, mu, sc, df):
        return log_score_t(y, mu, sc, df) if np.isfinite(df) else log_score(y, mu, sc)

    scores, rows = {}, []
    for name, (mu, sc, df) in rungs.items():
        s = ls(yte, mu, sc, df)
        scores[name] = s
        crps = (crps_t(yte, mu, sc, df) if np.isfinite(df)
                else crps_gaussian(yte, mu, sc))
        cov = {f"coverage_{int(l * 100)}":
               (coverage_t(yte, mu, sc, df, l) if np.isfinite(df)
                else coverage(yte, mu, sc, l)) for l in LEVELS}
        rows.append({"rung": name, "df": df, "logscore": float(s.mean()),
                     "CRPS": float(crps.mean()), **cov})

    table = pd.DataFrame(rows)
    total = table["logscore"].iloc[-1] - table["logscore"].iloc[0]

    names = list(rungs)
    steps = []
    for a, bn in zip(names[:-1], names[1:]):
        r = diebold_mariano(-scores[bn], -scores[a])
        gain = scores[bn].mean() - scores[a].mean()
        steps.append({"from": a, "to": bn, "gain_nats": gain,
                      "share_of_total_pct": 100 * gain / total,
                      "DM_stat": r["DM_stat"], "p_value": r["p_value"],
                      "significant_5pct": bool(r["p_value"] < 0.05)})
    step_table = pd.DataFrame(steps)

    pairs = []
    for a, bn in itertools.combinations(names, 2):
        r = diebold_mariano(-scores[bn], -scores[a])
        pairs.append({"model_A": bn, "model_B": a,
                      "logscore_diff": scores[bn].mean() - scores[a].mean(),
                      "DM_stat": r["DM_stat"], "p_value": r["p_value"]})
    pair_table = pd.DataFrame(pairs)

    TAB.mkdir(parents=True, exist_ok=True)
    table.to_csv(TAB / "v2_decomposition_levels.csv", index=False)
    step_table.to_csv(TAB / "v2_decomposition_steps.csv", index=False)
    pair_table.to_csv(TAB / "v2_decomposition_pairwise.csv", index=False)

    with pd.option_context("display.width", 220):
        print(f"Test sample: {len(yte)} observations\n")
        print(table.round(5).to_string(index=False))
        print(f"\nTotal gain from rung 1 to rung 4: {total:+.5f} nats\n")
        print(step_table.round(5).to_string(index=False))
    print("\nSaved three decomposition tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
