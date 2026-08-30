"""
Assemble the per-ticker result tables the manuscript needs.

Why per ticker
--------------
Pooled metrics hide the only thing a reader wants to know: whether a model
helps for *any* asset. A pooled out-of-sample R^2 near zero is consistent with
a model that helps two tickers and hurts six. The v1 reports contained no
per-ticker breakdown at all, so that question could not be answered from them.

Each model is compared against the per-ticker mean baseline computed under the
same protocol, and a Diebold-Mariano test is run per ticker on the squared-error
differential. With 751 observations per ticker the test has enough power to
detect an economically meaningful edge, so a table of insignificant results is
informative rather than merely inconclusive.

Usage
-----
    python -m src.experiments.make_paper_tables
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import dm_squared_error, oos_r2, point_metrics


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg step-up. Returns a boolean rejection mask.

    One test per model-ticker pair means dozens of simultaneous tests, and at
    the 5% level a handful of rejections is what chance alone produces. Any
    claim of predictability for an individual ticker has to survive a
    multiplicity correction, or it is an artefact of the number of tests.
    """
    p_values = np.asarray(p_values, dtype=float)
    finite = np.isfinite(p_values)
    reject = np.zeros(p_values.shape, dtype=bool)
    if not finite.any():
        return reject

    idx = np.flatnonzero(finite)
    order = idx[np.argsort(p_values[idx])]
    m = order.size
    thresholds = alpha * np.arange(1, m + 1) / m
    passed = p_values[order] <= thresholds
    if passed.any():
        reject[order[: np.flatnonzero(passed)[-1] + 1]] = True
    return reject

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJ = REPO_ROOT / "stock_project"
PRED_DIR = PROJ / "reports" / "predictions_v2"
TAB_DIR = PROJ / "reports" / "tables"

ID_COLS = ["Date", "Ticker"]
BASELINE = "baseline_ticker_mean"


def load(name: str) -> pd.DataFrame | None:
    path = PRED_DIR / f"{name}.parquet"
    if not path.exists():
        return None
    d = pd.read_parquet(path)
    return d[d["split"] == "test"]


def per_ticker(protocol: str) -> pd.DataFrame:
    ref = load(f"{BASELINE}__{protocol}")
    if ref is None:
        raise FileNotFoundError(f"Missing baseline for protocol {protocol!r}")
    ref = ref[ID_COLS + ["y_true", "y_pred"]].rename(columns={"y_pred": "y_ref"})

    names = sorted(p.stem for p in PRED_DIR.glob(f"*__{protocol}.parquet"))
    rows = []
    for name in names:
        pred = load(name)
        if pred is None:
            continue
        model = name.replace(f"__{protocol}", "")
        j = ref.merge(pred[ID_COLS + ["y_pred"]], on=ID_COLS, how="inner")
        if j.empty:
            continue

        for ticker, g in j.groupby("Ticker"):
            dm = dm_squared_error(g["y_true"], g["y_pred"], g["y_ref"])
            rows.append({
                "model": model,
                "protocol": protocol,
                "Ticker": ticker,
                "n": len(g),
                "RMSE": float(np.sqrt(((g["y_true"] - g["y_pred"]) ** 2).mean())),
                "OOS_R2_pct": 100.0 * oos_r2(g["y_true"], g["y_pred"], g["y_ref"]),
                "DirAcc": point_metrics(g["y_true"], g["y_pred"])["DirAcc"],
                "DM_stat": dm["DM_stat"],
                "DM_p": dm["p_value"],
                "beats_baseline_5pct": bool(
                    np.isfinite(dm["p_value"]) and dm["p_value"] < 0.05
                    and dm["favours"] == "A"
                ),
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocols", nargs="+", default=["fixed", "expanding"])
    args = ap.parse_args()

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    frames = [per_ticker(p) for p in args.protocols]
    long = pd.concat(frames, ignore_index=True)

    # Multiplicity correction across every model-ticker test, per protocol.
    long["significant_bh"] = False
    long["significant_bonferroni"] = False
    for protocol in long["protocol"].unique():
        m = (long["protocol"] == protocol) & (long["model"] != BASELINE)
        sub = long.loc[m]
        long.loc[m, "significant_bh"] = benjamini_hochberg(sub["DM_p"].to_numpy())
        long.loc[m, "significant_bonferroni"] = (
            sub["DM_p"].to_numpy() < 0.05 / max(len(sub), 1)
        )
    long.to_csv(TAB_DIR / "v2_per_ticker_long.csv", index=False)

    for protocol in args.protocols:
        sub = long[long["protocol"] == protocol]
        if sub.empty:
            continue
        wide = sub.pivot(index="Ticker", columns="model", values="OOS_R2_pct")
        wide.columns.name = None
        wide.loc["MEAN"] = wide.mean()
        wide.to_csv(TAB_DIR / f"v2_per_ticker_oos_r2_{protocol}.csv")

        print(f"\n{'='*78}\nOut-of-sample R^2 (%) vs per-ticker mean baseline "
              f"— {protocol} protocol\n{'='*78}")
        with pd.option_context("display.width", 220):
            print(wide.round(2).to_string())

        tested = sub[sub["model"] != BASELINE]
        total = len(tested)
        raw = tested[tested["beats_baseline_5pct"]]
        bh = tested[tested["significant_bh"] & (tested["DM_stat"] < 0)]
        bonf = tested[tested["significant_bonferroni"] & (tested["DM_stat"] < 0)]

        print(f"\nTests run: {total} model-ticker pairs")
        print(f"  beating the baseline, uncorrected p < 0.05 : {len(raw)}"
              f"  (expected by chance: {0.05 * total:.1f})")
        print(f"  surviving Benjamini-Hochberg FDR at 5%      : {len(bh)}")
        print(f"  surviving Bonferroni at 5%                  : {len(bonf)}")
        for _, r in raw.iterrows():
            verdict = "SURVIVES correction" if r["significant_bh"] else "does not survive correction"
            print(f"    {r['model']} / {r['Ticker']}: R2 = {r['OOS_R2_pct']:+.2f}%, "
                  f"p = {r['DM_p']:.4f}  -> {verdict}")

    print(f"\nSaved: v2_per_ticker_long.csv and one wide table per protocol")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
