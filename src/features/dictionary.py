"""
The explicit feature dictionary for this project.

Why this file exists
--------------------
Earlier versions selected model inputs with a dtype test: every numeric column
not on a short exclusion list became a feature. That rule admitted three
non-stationary level series (SP500_lag1, DFF_lag1, DGS10_lag1) whose mean moves
permanently between the training and test windows. Standardised on 2016-2019
moments and applied to 2023-2025, they arrive at z-scores of +8.7, +4.7 and
+3.9, and every model with an unbounded output head extrapolates along them.

Feature membership is therefore declared here, by hand, and the pipeline reads
this file. A column is a feature because it is listed, not because it happens
to be numeric.

Evidence behind the `stationary` flag
-------------------------------------
Augmented Dickey-Fuller on the full sample (macro series at the date level,
own-return series for a representative ticker):

    SP500_lag1         stat  +0.602   p 0.9877   non-stationary
    DFF_lag1           stat  -0.762   p 0.8301   non-stationary
    DGS10_lag1         stat  -0.858   p 0.8016   non-stationary
    vix_level_lag1     stat  -5.222   p 0.0000   stationary
    DFF_diff_lag1      stat -51.489   p 0.0000   stationary
    DGS10_diff_lag1    stat -36.897   p 0.0000   stationary
    vix_ret_lag1       stat -20.849   p 0.0000   stationary
    mkt_ret_lag1       stat -15.914   p 0.0000   stationary
    ret                stat -16.094   p 0.0000   stationary
    ret_vol20          stat  -5.523   p 0.0000   stationary
    garch11_t_sigma1   stat  -8.042   p 0.0000   stationary

Timing convention
-----------------
A panel row is keyed by (Date = t, Ticker). Its target is `target_ret`, the log
return from t to t+1, realised on `target_date` = t+1. Every feature listed
here is observable at the close of day t, so no feature uses information from
the target period. `observable_at` records this explicitly per feature.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Feature:
    """One model input, or one deliberately excluded candidate."""

    name: str
    family: str
    description: str
    source: str
    transform: str
    observable_at: str
    stationary: bool
    include: bool
    reason: str = ""


# Observability shorthand: every included feature is known by the close of t.
AT_T = "close of t"

FEATURES: list[Feature] = [
    # ---- own return dynamics -------------------------------------------------
    Feature("ret", "own return", "Own log return realised on day t",
            "panel", "identity", AT_T, True, True),
    Feature("ret_lag1", "own return", "Own log return on day t-1",
            "panel", "lag 1 of ret", AT_T, True, True),
    Feature("ret_lag2", "own return", "Own log return on day t-2",
            "panel", "lag 2 of ret", AT_T, True, True),
    Feature("ret_lag3", "own return", "Own log return on day t-3",
            "panel", "lag 3 of ret", AT_T, True, True),
    Feature("ret_lag5", "own return", "Own log return on day t-5",
            "panel", "lag 5 of ret", AT_T, True, True),
    Feature("ret_lag10", "own return", "Own log return on day t-10",
            "panel", "lag 10 of ret", AT_T, True, True),

    # ---- own realised volatility --------------------------------------------
    Feature("ret_vol5", "own volatility", "Trailing 5-day SD of own returns through t",
            "panel", "rolling sd, window 5", AT_T, True, True),
    Feature("ret_vol10", "own volatility", "Trailing 10-day SD of own returns through t",
            "panel", "rolling sd, window 10", AT_T, True, True),
    Feature("ret_vol20", "own volatility", "Trailing 20-day SD of own returns through t",
            "panel", "rolling sd, window 20", AT_T, True, True),

    # ---- market ---------------------------------------------------------------
    Feature("mkt_ret_lag1", "market", "Market proxy (SPY) log return, most recent close",
            "panel", "identity", AT_T, True, True),

    # ---- uncertainty ----------------------------------------------------------
    Feature("vix_log_lag1", "uncertainty", "Log of the CBOE VIX at the most recent close",
            "derived from vix_level_lag1", "natural log", AT_T, True, True,
            "Level is stationary by ADF, but right-skewed; the log is better "
            "behaved for a linear model and leaves the information intact."),
    Feature("vix_ret_lag1", "uncertainty", "Log change in VIX at the most recent close",
            "panel", "identity", AT_T, True, True),

    # ---- macro (changes, never levels) ----------------------------------------
    Feature("DFF_diff_lag1", "macro", "Daily change in the effective federal funds rate",
            "panel", "identity", AT_T, True, True),
    Feature("DGS10_diff_lag1", "macro", "Daily change in the 10-year Treasury yield",
            "panel", "identity", AT_T, True, True),

    # ---- conditional volatility (hybrid specifications only) -------------------
    Feature("garch11_t_sigma1", "garch", "GARCH(1,1)-t one-step-ahead conditional SD for t+1",
            "panel", "identity", AT_T, True, True,
            "Parameters estimated on the training split only, then filtered "
            "forward. Used by hybrid specifications and as the known scale in "
            "the hierarchical Bayesian model."),

    # ==== excluded candidates ==================================================
    # Kept in the dictionary so the paper can state what was considered and why.

    Feature("SP500_lag1", "macro", "S&P 500 index level at the most recent close",
            "panel", "identity", AT_T, False, False,
            "Non-stationary level (ADF p=0.99). Test-window mean sits at z=+8.73 "
            "under training moments and 100% of test rows exceed 3 training SDs. "
            "Primary driver of the out-of-sample extrapolation failure. Its "
            "stationary counterpart mkt_ret_lag1 is retained."),
    Feature("DFF_lag1", "macro", "Effective federal funds rate, level",
            "panel", "identity", AT_T, False, False,
            "Non-stationary level (ADF p=0.83). Test-window mean at z=+4.72, "
            "100% of test rows beyond 3 training SDs. Replaced by DFF_diff_lag1."),
    Feature("DGS10_lag1", "macro", "10-year Treasury constant maturity yield, level",
            "panel", "identity", AT_T, False, False,
            "Non-stationary level (ADF p=0.80). Test-window mean at z=+3.86, "
            "85% of test rows beyond 3 training SDs. Replaced by DGS10_diff_lag1."),
    Feature("vix_level_lag1", "uncertainty", "CBOE VIX level at the most recent close",
            "panel", "identity", AT_T, True, False,
            "Stationary, but superseded by its log transform vix_log_lag1."),
    Feature("ret_t", "own return", "Own log return realised on day t",
            "panel", "target_ret shifted by 1", AT_T, True, False,
            "Bit-identical to `ret` (max absolute difference 0.0). Exact duplicate."),
    Feature("sp500_ret_lag1", "market", "S&P 500 index log return",
            "panel", "identity", AT_T, True, False,
            "Correlates 0.9985 with mkt_ret_lag1; the two measure the same "
            "quantity. Retaining both inflates collinearity and makes the "
            "Bayesian coefficient posteriors uninterpretable."),
    Feature("garch11_t_var1", "garch", "GARCH(1,1)-t one-step-ahead conditional variance",
            "panel", "identity", AT_T, True, False,
            "Square of garch11_t_sigma1; redundant."),
    Feature("gjr11_t_sigma1", "garch", "GJR-GARCH(1,1)-t one-step-ahead conditional SD",
            "panel", "identity", AT_T, True, False,
            "Held back for the volatility-model robustness check rather than "
            "entered alongside garch11_t_sigma1, with which it is near-collinear."),
    Feature("gjr11_t_var1", "garch", "GJR-GARCH(1,1)-t one-step-ahead conditional variance",
            "panel", "identity", AT_T, True, False, "Square of gjr11_t_sigma1; redundant."),
    Feature("egarch11_t_sigma1", "garch", "EGARCH(1,1)-t one-step-ahead conditional SD",
            "panel", "identity", AT_T, True, False,
            "Training-split fit diverged: values reach 1e+148 and the training "
            "SD is infinite. Excluded pending re-estimation."),
    Feature("egarch11_t_var1", "garch", "EGARCH(1,1)-t one-step-ahead conditional variance",
            "panel", "identity", AT_T, True, False,
            "Square of a diverged fit; training SD is infinite."),
]

# Identifier and target columns. Never model inputs.
NON_FEATURES = ("Date", "target_date", "split", "Ticker", "target_ret", "has_garch")

TARGET = "target_ret"


def model_features(include_garch: bool = False) -> list[str]:
    """Feature names for a model.

    `include_garch` selects the hybrid specification, which adds the GARCH
    conditional SD to the base set. The base set is deliberately free of any
    GARCH-derived column so that base and hybrid differ by exactly one input.
    """
    return [
        f.name
        for f in FEATURES
        if f.include and (include_garch or f.family != "garch")
    ]


def excluded_features() -> list[str]:
    return [f.name for f in FEATURES if not f.include]


def to_frame() -> pd.DataFrame:
    """The dictionary as a table, for export into the manuscript."""
    return pd.DataFrame([asdict(f) for f in FEATURES])


def export_csv(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    to_frame().to_csv(path, index=False)
    return path


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[2]
    out = export_csv(repo / "stock_project" / "reports" / "tables" / "feature_dictionary_v2.csv")

    base, hybrid = model_features(False), model_features(True)
    print(f"Base features   ({len(base):2d}): {', '.join(base)}")
    print(f"Hybrid features ({len(hybrid):2d}): {', '.join(hybrid)}")
    print(f"Excluded        ({len(excluded_features()):2d}): {', '.join(excluded_features())}")
    print(f"\nSaved: {out}")
