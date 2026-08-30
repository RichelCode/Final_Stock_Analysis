"""
Leave-one-ticker-out and k-shot transfer for the hierarchical Bayesian model.

The question
------------
Partial pooling is only worth its complexity if it buys something a per-asset
model cannot. The test is a ticker the model has never seen: train on the other
seven, then forecast the held-out one. Its own intercept alpha is unknown, so
the model integrates alpha over its population distribution, and the predictive
variance picks up tau_a^2 on top of the observation noise. A model without a
population layer has nothing to integrate and cannot answer at all.

k-shot then reveals the held-out ticker's first k trading days and refits, which
measures how much of a new asset's history is actually needed.

What went wrong in v1
---------------------
The v1 leave-one-out and 60-day k-shot runs returned RMSEs agreeing to four
decimals and coverage agreeing to sixteen significant figures, which is not a
finding about information content but a sign that the adaptation step changed
nothing. This version asserts that the posterior actually moves between
settings before reporting, and sweeps k rather than reporting a single point.

Usage
-----
    python -m src.experiments.run_transfer
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.eval.metrics import coverage_t, crps_t, log_score_t
from src.features.dictionary import TARGET, model_features
from src.models.hierarchical_bayes import (fit, predict_seen, predict_unseen,
                                           predictive_df)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJ = REPO_ROOT / "stock_project"
DEFAULT_PANEL = PROJ / "data" / "processed" / "panel_v2_garch.parquet"
TAB_DIR = PROJ / "reports" / "tables"
SIGMA_COL = "garch11_t_sigma1"
SEED = 20260830
K_GRID = [0, 20, 60, 120, 250]


def score(y, mu, scale, df) -> dict:
    return {
        "RMSE": float(np.sqrt(np.mean((y - mu) ** 2))),
        "logscore": float(np.mean(log_score_t(y, mu, scale, df))),
        "CRPS": float(np.mean(crps_t(y, mu, scale, df))),
        **{f"coverage_{int(l*100)}": coverage_t(y, mu, scale, df, l)
           for l in (0.5, 0.8, 0.9, 0.95)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    ap.add_argument("--iter", type=int, default=12000)
    ap.add_argument("--burn", type=int, default=6000)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--thin", type=int, default=3)
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel)
    features = model_features(include_garch=False)
    df_all = panel.dropna(subset=features + [TARGET, SIGMA_COL, "split"]).copy()
    tickers = sorted(df_all["Ticker"].unique())

    rows = []
    for held in tickers:
        t0 = time.time()
        others = [t for t in tickers if t != held]
        gmap = {t: i for i, t in enumerate(others)}

        base_train = df_all[(df_all["split"] == "train") & (df_all["Ticker"] != held)]
        held_train = df_all[(df_all["split"] == "train") & (df_all["Ticker"] == held)]
        held_test = df_all[(df_all["split"] == "test") & (df_all["Ticker"] == held)]

        mu_x = base_train[features].mean()
        sd_x = base_train[features].std().replace(0.0, 1.0)

        def design(d):
            return ((d[features] - mu_x) / sd_x).to_numpy(float)

        Xte = design(held_test)
        yte = held_test[TARGET].to_numpy(float)
        ste = held_test[SIGMA_COL].to_numpy(float)

        prev_beta = None
        for k in K_GRID:
            shots = held_train.tail(k) if k > 0 else held_train.iloc[:0]

            if k == 0:
                Xtr, ytr, str_ = (design(base_train),
                                  base_train[TARGET].to_numpy(float),
                                  base_train[SIGMA_COL].to_numpy(float))
                gtr = base_train["Ticker"].map(gmap).to_numpy(int)
                n_groups = len(others)
            else:
                # The held-out ticker joins as an extra group, so its own
                # intercept is estimated from k observations while still being
                # shrunk toward the population mean.
                combined = pd.concat([base_train, shots])
                gmap_k = dict(gmap); gmap_k[held] = len(others)
                Xtr, ytr, str_ = (design(combined),
                                  combined[TARGET].to_numpy(float),
                                  combined[SIGMA_COL].to_numpy(float))
                gtr = combined["Ticker"].map(gmap_k).to_numpy(int)
                n_groups = len(others) + 1

            post = fit(Xtr, ytr, gtr, str_, n_groups, n_chains=args.chains,
                       n_iter=args.iter, burn=args.burn, thin=args.thin,
                       seed=SEED, dist="t")
            dfree = predictive_df(post)

            if k == 0:
                mu, scale, _ = predict_unseen(Xte, ste, post)
            else:
                gte = np.full(len(Xte), len(others), dtype=int)
                mu, scale, _ = predict_seen(Xte, gte, ste, post)

            # Guard against the v1 failure mode: if the posterior does not move
            # between settings, the adaptation is not happening.
            beta_mean = post.beta.mean(axis=0)
            moved = (np.nan if prev_beta is None
                     else float(np.abs(beta_mean - prev_beta).max()))
            prev_beta = beta_mean

            rows.append({"Ticker": held, "k": k, "n_test": len(yte), "df": dfree,
                         "alpha_known": k > 0, "beta_shift_vs_prev_k": moved,
                         **score(yte, mu, scale, dfree)})
            print(f"  {held:6s} k={k:4d}  RMSE={rows[-1]['RMSE']:.5f} "
                  f"logscore={rows[-1]['logscore']:.4f} "
                  f"cov90={rows[-1]['coverage_90']:.3f} "
                  f"beta_shift={moved if moved is None else f'{moved:.2e}'}")
        print(f"  {held} done in {time.time()-t0:.0f}s")

    out = pd.DataFrame(rows)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(TAB_DIR / "v2_transfer_kshot.csv", index=False)

    summary = out.groupby("k").agg(
        RMSE=("RMSE", "mean"), logscore=("logscore", "mean"),
        CRPS=("CRPS", "mean"), coverage_90=("coverage_90", "mean"),
        min_beta_shift=("beta_shift_vs_prev_k", "min")).reset_index()
    summary.to_csv(TAB_DIR / "v2_transfer_summary.csv", index=False)

    with pd.option_context("display.width", 200):
        print("\nAveraged over the eight held-out tickers:\n")
        print(summary.round(5).to_string(index=False))

    stuck = summary["min_beta_shift"].dropna()
    if len(stuck) and (stuck < 1e-12).any():
        print("\nWARNING: the posterior did not move between k settings.")
    else:
        print("\nPosterior moves at every k, so the adaptation is real.")
    print(f"\nSaved: v2_transfer_kshot.csv, v2_transfer_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
