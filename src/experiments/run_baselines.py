"""
Re-run the baseline forecasts on panel_v2 under both evaluation protocols.

The baselines are deliberately unclever. They exist to set the bar that any
richer model must clear, and to supply the reference forecast for out-of-sample
R^2. They are handled here rather than through src.eval.protocol because each
is defined per ticker, and so needs the group label that the generic
feature-matrix interface does not carry.

  baseline_zero         always forecasts zero
  baseline_ticker_mean  the ticker's own mean return over the information set
  baseline_ar1          per-ticker AR(1) on the previous day's own return

Under the fixed-origin protocol the information set is the training split.
Under the expanding-window protocol it is every observation whose target_date
falls strictly before the month being scored, so the baselines face exactly the
same information constraint as the models they are compared against.

Usage
-----
    python -m src.experiments.run_baselines
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import point_metrics, oos_r2

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJ = REPO_ROOT / "stock_project"
DEFAULT_PANEL = PROJ / "data" / "processed" / "panel_v2.parquet"
PRED_DIR = PROJ / "reports" / "predictions_v2"
TAB_DIR = PROJ / "reports" / "tables"

TARGET = "target_ret"
ID_COLS = ["Date", "target_date", "Ticker"]


def _out(rows: pd.DataFrame, pred: np.ndarray, model: str) -> pd.DataFrame:
    f = rows[ID_COLS].copy()
    f["split"] = "test"
    f["model"] = model
    f["y_true"] = rows[TARGET].to_numpy(float)
    f["y_pred"] = np.asarray(pred, dtype=float)
    f["residual"] = f["y_true"] - f["y_pred"]
    return f


def fixed_origin_baselines(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tr = panel[panel["split"] == "train"]
    te = panel[panel["split"] == "test"].copy()

    means = tr.groupby("Ticker")[TARGET].mean()

    # Per-ticker AR(1): target_ret_t = a + b * ret_t, fitted on the training split.
    coefs = {}
    for tkr, g in tr.groupby("Ticker"):
        g = g.dropna(subset=["ret", TARGET])
        b, a = np.polyfit(g["ret"].to_numpy(float), g[TARGET].to_numpy(float), 1)
        coefs[tkr] = (a, b)

    ar1 = np.array([
        coefs[t][0] + coefs[t][1] * r
        for t, r in zip(te["Ticker"], te["ret"].fillna(0.0))
    ])

    return {
        "baseline_zero": _out(te, np.zeros(len(te)), "baseline_zero"),
        "baseline_ticker_mean": _out(te, te["Ticker"].map(means).to_numpy(float),
                                     "baseline_ticker_mean"),
        "baseline_ar1": _out(te, ar1, "baseline_ar1"),
    }


def expanding_baselines(panel: pd.DataFrame, min_obs: int = 250) -> dict[str, pd.DataFrame]:
    df = panel.dropna(subset=[TARGET]).copy()
    df["target_date"] = pd.to_datetime(df["target_date"])
    scored = df[df["split"] == "test"]

    zero, mean, ar1 = [], [], []
    for month in sorted(scored["target_date"].dt.to_period("M").unique()):
        start, end = month.start_time, month.end_time
        hist = df[df["target_date"] < start]
        te = scored[(scored["target_date"] >= start) & (scored["target_date"] <= end)]
        if te.empty or len(hist) < min_obs:
            continue

        means = hist.groupby("Ticker")[TARGET].mean()

        coefs = {}
        for tkr, g in hist.groupby("Ticker"):
            g = g.dropna(subset=["ret", TARGET])
            if len(g) < 30:
                coefs[tkr] = (means.get(tkr, 0.0), 0.0)
                continue
            b, a = np.polyfit(g["ret"].to_numpy(float), g[TARGET].to_numpy(float), 1)
            coefs[tkr] = (a, b)

        zero.append(_out(te, np.zeros(len(te)), "baseline_zero"))
        mean.append(_out(te, te["Ticker"].map(means).fillna(0.0).to_numpy(float),
                         "baseline_ticker_mean"))
        ar1.append(_out(te, np.array([
            coefs.get(t, (0.0, 0.0))[0] + coefs.get(t, (0.0, 0.0))[1] * r
            for t, r in zip(te["Ticker"], te["ret"].fillna(0.0))
        ]), "baseline_ar1"))

    return {
        "baseline_zero": pd.concat(zero, ignore_index=True),
        "baseline_ticker_mean": pd.concat(mean, ignore_index=True),
        "baseline_ar1": pd.concat(ar1, ignore_index=True),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    protocols = {
        "fixed": fixed_origin_baselines(panel),
        "expanding": expanding_baselines(panel),
    }

    rows = []
    for protocol, results in protocols.items():
        # Out-of-sample R^2 is measured against the ticker-mean baseline
        # computed under the same protocol, so the reference never has an
        # information advantage over the model being scored.
        ref = results["baseline_ticker_mean"].set_index(ID_COLS)["y_pred"]

        for name, pred in results.items():
            pred.to_parquet(PRED_DIR / f"{name}__{protocol}.parquet", index=False)
            aligned = pred.set_index(ID_COLS)
            rows.append({
                "model": name,
                "protocol": protocol,
                **point_metrics(aligned["y_true"], aligned["y_pred"]),
                "OOS_R2_vs_ticker_mean": oos_r2(
                    aligned["y_true"], aligned["y_pred"], ref.reindex(aligned.index)
                ),
            })

    table = pd.DataFrame(rows)
    out = TAB_DIR / "v2_baseline_metrics.csv"
    table.to_csv(out, index=False)

    with pd.option_context("display.width", 200):
        print(table.round(5).to_string(index=False))
    print(f"\nSaved: {out}")
    print(f"Predictions: {PRED_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
