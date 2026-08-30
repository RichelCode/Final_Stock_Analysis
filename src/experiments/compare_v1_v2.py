"""
Quantify what the stationarity fix changed, per model and per ticker.

This is the evidence for the paper's methodological contribution. The v1
predictions in reports/predictions/ were produced with non-stationary level
features standardised on a fixed 2016-2019 origin; the v2 predictions in
reports/predictions_v2/ use the declared stationary feature set. Both score the
same test window against the same per-ticker mean baseline, so the difference
is attributable to the feature change alone.

Usage
-----
    python -m src.experiments.compare_v1_v2
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import oos_r2

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJ = REPO_ROOT / "stock_project"
V1_DIR = PROJ / "reports" / "predictions"
V2_DIR = PROJ / "reports" / "predictions_v2"
TAB_DIR = PROJ / "reports" / "tables"

# Models present in both generations, under their respective names.
PAIRS = {
    "ridge_pooled": ("ridge_pooled", "ridge_pooled__fixed"),
    "rf_pooled": ("rf_pooled", "rf_pooled__fixed"),
    "hierarchical_bayes": ("bayes_hier_partial_pooling", "bayes_hier_horseshoe_t__fixed"),
}
# v1-only models, kept to show the size of the failure they exhibited.
V1_ONLY = ["lstm_base", "gru_base", "transformer_base",
           "gru_hybrid_garch11_t_sigma1", "lstm_hybrid_garch11_t_sigma1"]


def test_rows(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    d = pd.read_parquet(path)
    return d[d["split"] == "test"][["Date", "Ticker", "y_true", "y_pred"]]


def scored(pred: pd.DataFrame, ref: pd.DataFrame) -> pd.Series:
    j = ref.merge(pred[["Date", "Ticker", "y_pred"]], on=["Date", "Ticker"], how="inner")
    return j.groupby("Ticker").apply(
        lambda g: 100.0 * oos_r2(g["y_true"], g["y_pred"], g["y_ref"]),
        include_groups=False,
    )


def main() -> int:
    ref_v1 = test_rows(V1_DIR / "baseline_ticker_mean.parquet")
    ref_v2 = test_rows(V2_DIR / "baseline_ticker_mean__fixed.parquet")
    if ref_v1 is None or ref_v2 is None:
        raise FileNotFoundError("Both baseline prediction files are required.")
    ref_v1 = ref_v1.rename(columns={"y_pred": "y_ref"})
    ref_v2 = ref_v2.rename(columns={"y_pred": "y_ref"})

    rows = []
    for label, (v1_name, v2_name) in PAIRS.items():
        p1, p2 = test_rows(V1_DIR / f"{v1_name}.parquet"), test_rows(V2_DIR / f"{v2_name}.parquet")
        if p1 is None or p2 is None:
            print(f"  (skipping {label}: missing predictions)")
            continue
        r1, r2 = scored(p1, ref_v1), scored(p2, ref_v2)
        for ticker in sorted(set(r1.index) & set(r2.index)):
            rows.append({"model": label, "Ticker": ticker,
                         "OOS_R2_v1_pct": r1[ticker], "OOS_R2_v2_pct": r2[ticker]})

    per_ticker = pd.DataFrame(rows)
    per_ticker.to_csv(TAB_DIR / "v1_vs_v2_per_ticker.csv", index=False)

    summary = (per_ticker.groupby("model")[["OOS_R2_v1_pct", "OOS_R2_v2_pct"]]
               .mean().reset_index())
    summary["improvement_pp"] = summary["OOS_R2_v2_pct"] - summary["OOS_R2_v1_pct"]

    # v1-only models, to record how large the failure was where it was worst.
    extra = []
    for name in V1_ONLY:
        p = test_rows(V1_DIR / f"{name}.parquet")
        if p is None:
            continue
        r = scored(p, ref_v1)
        extra.append({"model": name, "OOS_R2_v1_pct": r.mean(),
                      "OOS_R2_v2_pct": np.nan, "improvement_pp": np.nan})
    summary = pd.concat([summary, pd.DataFrame(extra)], ignore_index=True)
    summary.to_csv(TAB_DIR / "v1_vs_v2_summary.csv", index=False)

    with pd.option_context("display.width", 200):
        print("Mean out-of-sample R^2 (%) across the eight tickers, test window:\n")
        print(summary.round(2).to_string(index=False))
        print("\nPer ticker:\n")
        print(per_ticker.pivot(index="Ticker", columns="model",
                               values=["OOS_R2_v1_pct", "OOS_R2_v2_pct"]).round(2).to_string())
    print("\nSaved: v1_vs_v2_per_ticker.csv, v1_vs_v2_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
