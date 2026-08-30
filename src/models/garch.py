"""
Re-estimate the GARCH-family conditional volatility features, with validation.

Protocol
--------
For each ticker, parameters are estimated on the training split only, via
`last_obs`. The fitted parameters are then held fixed and the recursion is
filtered forward through the validation and test windows to produce genuine
one-step-ahead forecasts sigma_{t+1|t}. No test-window observation influences
any parameter.

Returns are passed in percent. The arch package's optimiser is poorly scaled on
raw decimal returns, and percent is the documented convention.

Alignment
---------
The value stored at (Date = t, Ticker) is the conditional standard deviation
forecast for t+1. The panel's target at that row is `target_ret`, the return
from t to t+1. Feature and target therefore refer to the same period, and the
forecast uses only information through t.

Why the validation exists
-------------------------
The v1 feature file contained nine training rows, spread over three tickers in
2017, where the EGARCH recursion produced sigma up to 1.3e+152. Those values
gave the training column an infinite standard deviation and a QLIKE of
+124894, while the test window looked healthy, so EGARCH was still reported as
the best volatility model. Every fit here is now checked for optimiser
convergence, finite parameters, and a conditional SD inside a plausible range,
and the run aborts rather than writing a column a later step will misread.

Usage
-----
    python -m src.models.garch
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJ = REPO_ROOT / "stock_project"
DEFAULT_PANEL = PROJ / "data" / "processed" / "panel_v2.parquet"
DEFAULT_OUT = PROJ / "data" / "processed" / "panel_v2_garch.parquet"
TAB_DIR = PROJ / "reports" / "tables"

TRAIN_END = pd.Timestamp("2019-12-31")

# Returns are modelled in percent; a one-day conditional SD above 100% is not a
# volatility estimate but a diverged recursion.
MAX_PLAUSIBLE_SIGMA_PCT = 100.0

SPECS: dict[str, dict] = {
    "garch11_t": dict(mean="Zero", vol="GARCH", p=1, o=0, q=1, dist="t"),
    "gjr11_t": dict(mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="t"),
    "egarch11_t": dict(mean="Zero", vol="EGARCH", p=1, o=1, q=1, dist="t"),
}


def fit_one(y_pct: pd.Series, spec: dict, n_train: int) -> tuple[pd.Series, dict]:
    """Fit on the first `n_train` observations, filter forward, return sigma in decimal."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(y_pct, rescale=False, **spec).fit(
            disp="off", update_freq=0, last_obs=n_train
        )
        forecast = res.forecast(horizon=1, start=0, reindex=True, align="origin")

    var_pct2 = forecast.variance.iloc[:, 0]
    sigma_pct = np.sqrt(var_pct2)

    diagnostics = {
        "converged": int(res.convergence_flag) == 0,
        "params_finite": bool(np.isfinite(res.params).all()),
        "max_sigma_pct": float(sigma_pct.max(skipna=True)),
        "n_train": n_train,
        "loglik": float(res.loglikelihood),
        "aic": float(res.aic),
        **{f"param_{k}": float(v) for k, v in res.params.items()},
    }
    # Convert percent SD back to the decimal scale the panel's returns use.
    return sigma_pct / 100.0, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    panel["Date"] = pd.to_datetime(panel["Date"])

    sigma_frames, summary, failures = [], [], []

    for ticker in sorted(panel["Ticker"].unique()):
        g = panel[panel["Ticker"] == ticker].sort_values("Date").set_index("Date")
        y = (g["ret"].dropna() * 100.0).astype(float)
        n_train = int((y.index <= TRAIN_END).sum())

        for name, spec in SPECS.items():
            sigma, diag = fit_one(y, spec, n_train)

            diag |= {"ticker": ticker, "spec": name}
            summary.append(diag)

            if not diag["converged"]:
                failures.append(f"{ticker}/{name}: optimiser did not converge")
            if not diag["params_finite"]:
                failures.append(f"{ticker}/{name}: non-finite parameters")
            if diag["max_sigma_pct"] > MAX_PLAUSIBLE_SIGMA_PCT:
                failures.append(
                    f"{ticker}/{name}: max conditional SD {diag['max_sigma_pct']:.3g}% "
                    f"exceeds {MAX_PLAUSIBLE_SIGMA_PCT}%"
                )

            sigma_frames.append(
                pd.DataFrame({
                    "Date": sigma.index,
                    "Ticker": ticker,
                    "spec": name,
                    "sigma1": sigma.to_numpy(float),
                })
            )

            print(f"  {ticker:6s} {name:11s} converged={diag['converged']!s:5s} "
                  f"max_sigma={diag['max_sigma_pct']:7.3f}%  aic={diag['aic']:9.1f}")

    if failures:
        print("\nVALIDATION FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    # Long -> wide, one column per specification.
    wide = (
        pd.concat(sigma_frames, ignore_index=True)
        .pivot(index=["Date", "Ticker"], columns="spec", values="sigma1")
        .rename(columns=lambda c: f"{c}_sigma1")
        .reset_index()
    )
    wide.columns.name = None

    # The v1 GARCH column is superseded by this re-estimation; drop it so the
    # merge cannot leave a stale duplicate behind.
    stale = [c for c in panel.columns if c.endswith("_sigma1") or c.endswith("_var1")]
    if stale:
        print(f"\nDropping superseded v1 columns: {', '.join(stale)}")
        panel = panel.drop(columns=stale)

    merged = panel.merge(wide, on=["Date", "Ticker"], how="left", validate="one_to_one")
    if merged.columns.duplicated().any():
        raise AssertionError(f"Duplicate columns after merge: "
                             f"{merged.columns[merged.columns.duplicated()].tolist()}")

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    fit_table = TAB_DIR / "v2_garch_fit_summary.csv"
    pd.DataFrame(summary).to_csv(fit_table, index=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.out, index=False)

    sig_cols = [f"{n}_sigma1" for n in SPECS]
    print(f"\nAll {len(summary)} fits converged with finite parameters.")
    print(f"\nConditional SD by specification (decimal, all splits):")
    print(merged[sig_cols].describe().loc[["mean", "std", "min", "max"]].round(5).to_string())
    print(f"\nSaved: {args.out.name}  ({merged.shape[0]} rows x {merged.shape[1]} cols)")
    print(f"Saved: {fit_table.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
