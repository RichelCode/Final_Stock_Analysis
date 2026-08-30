"""
Measure how far the test-window feature distribution drifts from the training
window, in units of training standard deviations.

Motivation
----------
Models in this project standardise features once, using training-split
statistics only. That is the standard leakage-avoidance practice, and it is
safe as long as features are stationary. It is *not* safe for level features
(an index level, a policy rate), whose mean moves permanently between the
training and test windows. When it moves, test inputs arrive at z-scores far
outside anything seen during fitting, and any model with an unbounded linear
output extrapolates along that axis.

This script quantifies that drift so the effect can be reported rather than
discovered by accident. Run it before and after the stationarity fix; the
"after" run is the evidence that the fix worked.

Usage
-----
    python -m src.diagnostics.feature_drift
    python -m src.diagnostics.feature_drift --panel <path> --label after_fix
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Columns that are identifiers or targets, never model inputs.
NON_FEATURES = {"Date", "target_date", "split", "Ticker", "target_ret", "has_garch"}

# A test-window mean beyond this many training SDs is treated as severe drift.
# Two SDs already puts the bulk of the test window outside the training range.
SEVERE_Z = 2.0

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANEL = (
    REPO_ROOT / "stock_project" / "data" / "processed" / "panel_with_garch_trainfit.parquet"
)
DEFAULT_OUTDIR = REPO_ROOT / "stock_project" / "reports" / "tables"


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns that a model would treat as inputs."""
    return sorted(
        c
        for c in df.columns
        if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(df[c])
    )


def drift_report(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """One row per feature: training moments, and where the test window lands."""
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    if train.empty or test.empty:
        raise ValueError("Panel must contain both 'train' and 'test' rows.")

    rows = []
    for col in features:
        mu = train[col].mean()
        sd = train[col].std()

        # A constant training column carries no scale, so a z-score is undefined.
        if not np.isfinite(sd) or sd == 0:
            rows.append(
                {
                    "feature": col,
                    "train_mean": mu,
                    "train_sd": sd,
                    "test_mean": test[col].mean(),
                    "z_test_mean": np.nan,
                    "z_test_absmax": np.nan,
                    "frac_test_beyond_3sd": np.nan,
                    "severe_drift": True,  # degenerate scale is its own problem
                }
            )
            continue

        z = (test[col] - mu) / sd
        rows.append(
            {
                "feature": col,
                "train_mean": mu,
                "train_sd": sd,
                "test_mean": test[col].mean(),
                "z_test_mean": z.mean(),
                "z_test_absmax": z.abs().max(),
                "frac_test_beyond_3sd": (z.abs() > 3).mean(),
                "severe_drift": bool(abs(z.mean()) > SEVERE_Z),
            }
        )

    report = pd.DataFrame(rows)
    return report.reindex(
        report["z_test_mean"].abs().sort_values(ascending=False, na_position="first").index
    ).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--label",
        default="before_fix",
        help="Suffix for the output file, e.g. before_fix / after_fix.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.panel)
    features = feature_columns(df)
    report = drift_report(df, features)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"diagnostic_feature_drift_{args.label}.csv"
    report.to_csv(out, index=False)

    severe = report[report["severe_drift"]]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"Panel:    {args.panel}")
        print(f"Features: {len(features)}")
        print(
            f"Train:    {df.loc[df['split'] == 'train', 'Date'].min().date()} "
            f"-> {df.loc[df['split'] == 'train', 'Date'].max().date()}"
        )
        print(
            f"Test:     {df.loc[df['split'] == 'test', 'Date'].min().date()} "
            f"-> {df.loc[df['split'] == 'test', 'Date'].max().date()}"
        )
        print()
        print(report.round(4).to_string(index=False))
        print()
        if len(severe):
            print(f"SEVERE DRIFT: {len(severe)} feature(s) beyond {SEVERE_Z} training SDs")
            for _, r in severe.iterrows():
                print(f"  - {r['feature']:20s} test mean sits at z = {r['z_test_mean']:+.2f}")
        else:
            print(f"No feature drifts beyond {SEVERE_Z} training SDs.")
    print(f"\nSaved: {out}")

    # Non-zero exit when severe drift is present, so this can gate a pipeline.
    return 1 if len(severe) else 0


if __name__ == "__main__":
    raise SystemExit(main())
