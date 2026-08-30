"""
Build the corrected modelling panel (v2) from the existing GARCH-merged panel.

What changes relative to v1
---------------------------
1. Membership is taken from src.features.dictionary rather than from a dtype
   test, so non-stationary level series can no longer enter by accident.
2. `vix_log_lag1` is derived, replacing the raw VIX level.
3. Excluded candidates are dropped rather than carried along, so no downstream
   step can pick them up.

The identifier, split and target columns are passed through unchanged: this
step re-selects and derives features, it does not re-estimate anything. GARCH
columns were produced by fitting on the training split only and forecasting one
step ahead, and that alignment is preserved.

Usage
-----
    python -m src.data.build_panel_v2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.dictionary import FEATURES, NON_FEATURES, TARGET, model_features

REPO_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = REPO_ROOT / "stock_project" / "data" / "processed"
DEFAULT_IN = PROC_DIR / "panel_with_garch_trainfit.parquet"
DEFAULT_OUT = PROC_DIR / "panel_v2.parquet"


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Create the features the dictionary marks as derived."""
    out = df.copy()

    # VIX is strictly positive and right-skewed; the log is better behaved for a
    # linear predictor and preserves the information in the level.
    if "vix_level_lag1" not in out.columns:
        raise KeyError("vix_level_lag1 is required to derive vix_log_lag1.")
    vix = out["vix_level_lag1"]
    if (vix <= 0).any():
        raise ValueError(f"vix_level_lag1 has {(vix <= 0).sum()} non-positive values.")
    out["vix_log_lag1"] = np.log(vix)

    return out


def validate(df: pd.DataFrame, features: list[str]) -> None:
    """Fail loudly rather than writing a panel a later step will misread."""
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Declared features absent from the panel: {missing}")

    banned = {f.name for f in FEATURES if not f.include} & set(df.columns)
    if banned:
        raise AssertionError(f"Excluded candidates survived into the panel: {sorted(banned)}")

    for col in features:
        values = df[col]
        if not np.isfinite(values.dropna()).all():
            raise ValueError(f"{col} contains non-finite values.")
        # A feature whose training SD is zero or non-finite cannot be
        # standardised, which is exactly the EGARCH failure mode.
        sd = values[df["split"] == "train"].std()
        if not np.isfinite(sd) or sd == 0:
            raise ValueError(f"{col} has a degenerate training SD ({sd}).")

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-panel", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    raw = pd.read_parquet(args.in_panel)
    print(f"Input:  {args.in_panel.name}  {raw.shape[0]} rows x {raw.shape[1]} cols")

    df = derive(raw)
    features = model_features(include_garch=True)

    keep = [c for c in NON_FEATURES if c in df.columns] + features
    panel = df[keep].copy()

    # Rows without a target cannot train or score anything.
    before = len(panel)
    panel = panel.dropna(subset=[TARGET, "split"]).reset_index(drop=True)
    print(f"Dropped {before - len(panel)} rows lacking a target or split label.")

    validate(panel, features)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.out, index=False)

    print(f"Output: {args.out.name}  {panel.shape[0]} rows x {panel.shape[1]} cols")
    print(f"Features kept: {len(features)}  |  dropped from v1: "
          f"{len(set(raw.columns) - set(panel.columns))}")
    print()
    counts = panel.groupby("split").agg(
        rows=("Date", "size"), start=("Date", "min"), end=("Date", "max")
    )
    print(counts.to_string())
    print()
    print("Missing values per feature (non-zero only):")
    na = panel[features].isna().sum()
    print(na[na > 0].to_string() if (na > 0).any() else "  none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
