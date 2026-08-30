"""
Re-run the pooled linear and tree models on panel_v2 under both protocols.

Both estimators are wrapped in a pipeline whose first step is the scaler, so
under the expanding-window protocol the scaler is refitted inside every fold
alongside the model. That is the specific change this project needed: in v1 a
single scaler was fitted on 2016-2019 and reused through 2025, which is what
sent standardised test inputs to z = +8.7 and drove the extrapolation failure.

Ridge regularisation is selected by RidgeCV over a fixed alpha grid, fitted
only on the fold's own training rows. The random forest uses a large minimum
leaf size, which is the standard guard against fitting noise in low
signal-to-noise return data.

Usage
-----
    python -m src.experiments.run_linear_tree
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.eval.metrics import point_metrics, oos_r2, dm_squared_error
from src.eval.protocol import expanding_window, fixed_origin
from src.features.dictionary import TARGET, model_features

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJ = REPO_ROOT / "stock_project"
DEFAULT_PANEL = PROJ / "data" / "processed" / "panel_v2.parquet"
PRED_DIR = PROJ / "reports" / "predictions_v2"
TAB_DIR = PROJ / "reports" / "tables"

SEED = 42
ID_COLS = ["Date", "target_date", "Ticker"]
ALPHAS = np.logspace(-2, 4, 25)


def ridge_factory() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", RidgeCV(alphas=ALPHAS)),
    ])


def rf_factory() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=300,
            # Returns carry very little signal; a large leaf forces the trees to
            # average over many observations instead of memorising noise.
            min_samples_leaf=50,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])


MODELS = {"ridge_pooled": ridge_factory, "rf_pooled": rf_factory}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    features = model_features(include_garch=False)
    print(f"Features ({len(features)}): {', '.join(features)}\n")

    PRED_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, factory in MODELS.items():
        for protocol, runner in (("fixed", fixed_origin), ("expanding", expanding_window)):
            t0 = time.time()
            result = runner(panel, features, TARGET, factory, name)
            pred = result.predictions

            pred.to_parquet(PRED_DIR / f"{name}__{protocol}.parquet", index=False)

            ref = pd.read_parquet(
                PRED_DIR / f"baseline_ticker_mean__{protocol}.parquet"
            ).set_index(ID_COLS)["y_pred"]
            aligned = pred.set_index(ID_COLS)
            ref = ref.reindex(aligned.index)

            dm = dm_squared_error(aligned["y_true"], aligned["y_pred"], ref)
            rows.append({
                "model": name,
                "protocol": protocol,
                "folds": result.n_folds,
                **point_metrics(aligned["y_true"], aligned["y_pred"]),
                "OOS_R2_vs_ticker_mean": oos_r2(aligned["y_true"], aligned["y_pred"], ref),
                "DM_stat_vs_baseline": dm["DM_stat"],
                "DM_p_value": dm["p_value"],
                "seconds": round(time.time() - t0, 1),
            })
            print(f"  {name:14s} {protocol:10s} {result.n_folds:3d} fold(s)  "
                  f"{rows[-1]['seconds']:6.1f}s  RMSE={rows[-1]['RMSE']:.5f}  "
                  f"OOS R2={rows[-1]['OOS_R2_vs_ticker_mean']:+.4f}")

    table = pd.DataFrame(rows)
    out = TAB_DIR / "v2_linear_tree_metrics.csv"
    table.to_csv(out, index=False)

    show = ["model", "protocol", "RMSE", "Corr", "DirAcc", "pred_mean", "pred_sd",
            "OOS_R2_vs_ticker_mean", "DM_stat_vs_baseline", "DM_p_value"]
    with pd.option_context("display.width", 220):
        print()
        print(table[show].round(5).to_string(index=False))
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
