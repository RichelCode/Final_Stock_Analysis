"""
Compare conditional volatility forecasts under QLIKE, with significance tests.

Loss
----
QLIKE, log(h) + rv/h, is used because the realised-variance proxy available at
the daily frequency is the squared return, which is unbiased but extremely
noisy. Patton (2011) shows QLIKE and MSE are the loss functions that preserve
the ranking a perfect proxy would give; QLIKE is the less proxy-sensitive of
the two and is reported as the headline. MSE against the same proxy is
reported alongside it.

Significance
------------
The v1 volatility table ranked EGARCH ahead of GJR by 0.024 QLIKE and GARCH by
0.024, with no standard errors, so nothing could be concluded from it. Here
every specification is tested against the rolling-variance benchmark and
against the best performer, using Diebold-Mariano on the per-observation QLIKE
differential with a Newey-West HAC variance.

Usage
-----
    python -m src.experiments.run_volatility
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import diebold_mariano, qlike

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJ = REPO_ROOT / "stock_project"
DEFAULT_PANEL = PROJ / "data" / "processed" / "panel_v2_garch.parquet"
TAB_DIR = PROJ / "reports" / "tables"

SPEC_COLS = {
    "garch11_t": "garch11_t_sigma1",
    "gjr11_t": "gjr11_t_sigma1",
    "egarch11_t": "egarch11_t_sigma1",
}
# Trailing realised volatility, the benchmark any conditional model must beat.
BENCHMARK = ("RollingVol20", "ret_vol20")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    models = {**SPEC_COLS, BENCHMARK[0]: BENCHMARK[1]}

    # Score every model on the identical sample: rows where the proxy and all
    # forecasts are available. Comparing models on different samples is how the
    # v1 table ended up with 6024 observations for GARCH and 6020 for EGARCH.
    need = list(models.values()) + ["target_ret"]
    d = panel[panel["split"] == args.split].dropna(subset=need).copy()
    d["rv"] = d["target_ret"] ** 2

    losses, rows = {}, []
    for name, col in models.items():
        var_hat = d[col].to_numpy(float) ** 2
        loss = qlike(d["rv"].to_numpy(float), var_hat)
        losses[name] = loss
        rows.append({
            "model": name,
            "n": len(d),
            "QLIKE": float(loss.mean()),
            "MSE_RV": float(np.mean((d["rv"].to_numpy(float) - var_hat) ** 2)),
            "mean_sigma": float(d[col].mean()),
        })

    table = pd.DataFrame(rows).sort_values("QLIKE").reset_index(drop=True)
    best = table.loc[0, "model"]

    with pd.option_context("display.width", 200):
        print(f"Split: {args.split}   n = {len(d)}   (identical sample for every model)\n")
        print(table.round(6).to_string(index=False))

    # Pairwise DM on the QLIKE differential. A negative statistic favours A.
    pairs = []
    for a, b in itertools.combinations(models, 2):
        r = diebold_mariano(losses[a], losses[b])
        pairs.append({
            "model_A": a, "model_B": b,
            "QLIKE_A": losses[a].mean(), "QLIKE_B": losses[b].mean(),
            "diff_A_minus_B": r["mean_diff"],
            "DM_stat": r["DM_stat"], "p_value": r["p_value"],
            "significant_5pct": bool(r["p_value"] < 0.05) if np.isfinite(r["p_value"]) else False,
            "favours": r["favours"],
        })
    pair_table = pd.DataFrame(pairs)

    print(f"\nPairwise Diebold-Mariano on QLIKE (best by mean loss: {best})\n")
    with pd.option_context("display.width", 220):
        print(pair_table.round(5).to_string(index=False))

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    t1 = TAB_DIR / f"v2_volatility_qlike_{args.split}.csv"
    t2 = TAB_DIR / f"v2_volatility_dm_{args.split}.csv"
    table.to_csv(t1, index=False)
    pair_table.to_csv(t2, index=False)

    sig = pair_table[pair_table["significant_5pct"]]
    print(f"\n{len(sig)} of {len(pair_table)} pairwise comparisons significant at 5%.")
    for _, r in sig.iterrows():
        winner = r["model_A"] if r["favours"] == "A" else r["model_B"]
        loser = r["model_B"] if r["favours"] == "A" else r["model_A"]
        print(f"  {winner} beats {loser}  (DM = {r['DM_stat']:+.2f}, p = {r['p_value']:.4f})")

    print(f"\nSaved: {t1.name}\nSaved: {t2.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
