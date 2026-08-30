"""
Audit saved model predictions for extrapolation failure.

Motivation
----------
When a model is fitted on standardised non-stationary level features, its
test-window predictions drift along whichever level feature moved between
splits. The failure is silent: training error looks normal, no exception is
raised, and a metrics CSV is written as usual. It only shows up if you look at
the predictions themselves.

Two checks catch it:

1. Scale. Daily equity log returns have a standard deviation near 0.025 and a
   mean near zero. A model whose test predictions have mean +0.12 is asserting
   a 12% move every trading day, which is not a forecast but a diverged fit.

2. Level tracking. If predictions correlate strongly with a non-stationary
   level feature that the model was never meant to track, the model is reading
   the trend, not the signal.

Usage
-----
    python -m src.diagnostics.prediction_sanity
    python -m src.diagnostics.prediction_sanity --split test --label after_fix
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Level features whose train-to-test shift drives the extrapolation.
LEVEL_FEATURES = ["SP500_lag1", "DFF_lag1", "DGS10_lag1", "vix_level_lag1"]

# A prediction SD more than this multiple of the realised SD is implausible:
# no honest daily-return forecast is more variable than returns themselves.
MAX_PRED_SD_RATIO = 1.0

# Absolute prediction bias beyond this many realised SDs is implausible.
MAX_BIAS_SD = 0.5

# |corr(prediction, level feature)| above this means the model tracks a trend.
MAX_LEVEL_CORR = 0.30

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJ = REPO_ROOT / "stock_project"
DEFAULT_PRED_DIR = PROJ / "reports" / "predictions"
DEFAULT_PANEL = PROJ / "data" / "processed" / "panel_with_garch_trainfit.parquet"
DEFAULT_OUTDIR = PROJ / "reports" / "tables"


def audit_one(pred: pd.DataFrame, levels: pd.DataFrame, split: str) -> dict | None:
    """Summarise one model's predictions on one split, or None if absent."""
    d = pred[pred["split"] == split]
    if d.empty:
        return None

    d = d.merge(levels, on=["Date", "Ticker"], how="left")
    y, yhat = d["y_true"], d["y_pred"]
    true_sd = y.std()

    row = {
        "model": d["model"].iloc[0],
        "split": split,
        "n": len(d),
        "true_mean": y.mean(),
        "true_sd": true_sd,
        "pred_mean": yhat.mean(),
        "pred_sd": yhat.std(),
        "pred_min": yhat.min(),
        "pred_max": yhat.max(),
        "rmse": float(np.sqrt(((y - yhat) ** 2).mean())),
        # Bias and dispersion expressed in units of realised return SD, so the
        # numbers are comparable across tickers and splits.
        "bias_in_true_sd": (yhat.mean() - y.mean()) / true_sd,
        "pred_sd_ratio": yhat.std() / true_sd,
    }

    for col in LEVEL_FEATURES:
        # Constant or all-missing predictions have undefined correlation.
        row[f"corr_{col}"] = (
            yhat.corr(d[col]) if d[col].notna().any() and yhat.std() > 0 else np.nan
        )

    # A constant predictor (e.g. baseline_zero) has no correlation with anything,
    # so every entry is NaN and there is simply nothing to report.
    level_corrs = [row[f"corr_{c}"] for c in LEVEL_FEATURES]
    finite = [c for c in level_corrs if np.isfinite(c)]
    row["max_abs_level_corr"] = max(abs(c) for c in finite) if finite else np.nan

    row["fail_scale"] = bool(
        abs(row["bias_in_true_sd"]) > MAX_BIAS_SD
        or row["pred_sd_ratio"] > MAX_PRED_SD_RATIO
    )
    row["fail_level_tracking"] = bool(row["max_abs_level_corr"] > MAX_LEVEL_CORR)
    row["suspect"] = row["fail_scale"] or row["fail_level_tracking"]
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--label", default="before_fix")
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    keep = ["Date", "Ticker"] + [c for c in LEVEL_FEATURES if c in panel.columns]
    levels = panel[keep].copy()

    rows = []
    for path in sorted(args.pred_dir.glob("*.parquet")):
        pred = pd.read_parquet(path)
        if not {"split", "y_true", "y_pred", "Date", "Ticker"} <= set(pred.columns):
            print(f"  (skipped {path.name}: unexpected schema)")
            continue
        if "model" not in pred.columns:
            pred = pred.assign(model=path.stem)
        row = audit_one(pred, levels, args.split)
        if row is not None:
            rows.append(row)

    report = pd.DataFrame(rows).sort_values("rmse", ascending=False).reset_index(drop=True)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"diagnostic_prediction_sanity_{args.split}_{args.label}.csv"
    report.to_csv(out, index=False)

    show = [
        "model", "n", "pred_mean", "pred_sd", "rmse",
        "bias_in_true_sd", "pred_sd_ratio", "max_abs_level_corr", "suspect",
    ]
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(f"Split: {args.split}   models audited: {len(report)}")
        print(f"Realised return SD on this split: {report['true_sd'].iloc[0]:.5f}\n")
        print(report[show].round(4).to_string(index=False))

    suspect = report[report["suspect"]]
    print()
    if len(suspect):
        print(f"SUSPECT: {len(suspect)} of {len(report)} models fail a sanity check")
        for _, r in suspect.iterrows():
            reasons = []
            if r["fail_scale"]:
                reasons.append(
                    f"bias {r['bias_in_true_sd']:+.1f} SD, dispersion {r['pred_sd_ratio']:.1f}x"
                )
            if r["fail_level_tracking"]:
                reasons.append(f"tracks a level feature at r={r['max_abs_level_corr']:.2f}")
            print(f"  - {r['model']:46s} {'; '.join(reasons)}")
    else:
        print("All models pass the prediction sanity checks.")
    print(f"\nSaved: {out}")

    return 1 if len(suspect) else 0


if __name__ == "__main__":
    raise SystemExit(main())
