"""
Emit the manuscript's tables as LaTeX, straight from the result CSVs.

Tables are generated rather than transcribed so the manuscript cannot drift
from the numbers in reports/. Re-run this after any experiment and the .tex
files update; \\input them from the Overleaf project.

Requires booktabs and siunitx in the preamble:

    \\usepackage{booktabs}
    \\usepackage{siunitx}

Usage
-----
    python -m src.experiments.make_latex_tables
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJ = REPO_ROOT / "stock_project"
TAB = PROJ / "reports" / "tables"
OUT = REPO_ROOT / "paper" / "tables"

PRETTY = {
    "baseline_ticker_mean": "Ticker mean (baseline)",
    "baseline_zero": "Zero",
    "baseline_ar1": "AR(1), per ticker",
    "ridge_pooled": "Ridge, pooled",
    "rf_pooled": "Random forest, pooled",
    "bayes_hier_horseshoe": "Hierarchical Bayes (t)",
    "bayes_hier_horseshoe_t": "Hierarchical Bayes (t)",
    "bayes_hier_uncalibrated": "Hierarchical Bayes (t), uncalibrated",
    "bayes_hier_calib_sd": "Hierarchical Bayes (t), SD calibration",
    "ridge_garch": "Ridge $+$ GARCH scale",
    "baseline_garch": "Zero mean $+$ GARCH scale",
    "baseline_gaussian": "Zero mean $+$ constant variance",
    "lstm_base": "LSTM", "gru_base": "GRU",
    "cnn1d_base": "1D CNN", "transformer_base": "Transformer",
    "garch11_t": "GARCH(1,1)-$t$", "gjr11_t": "GJR-GARCH(1,1)-$t$",
    "egarch11_t": "EGARCH(1,1)-$t$", "RollingVol20": "Rolling variance (20d)",
}


def label(x: str) -> str:
    return PRETTY.get(x, x.replace("_", r"\_"))


def stars(p) -> str:
    """Conventional significance marks; blank when the test is undefined."""
    if p is None or not np.isfinite(p):
        return ""
    return "$^{***}$" if p < 0.01 else "$^{**}$" if p < 0.05 else "$^{*}$" if p < 0.10 else ""


def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(body)
    print(f"  {name}.tex")


def read(name: str) -> pd.DataFrame | None:
    path = TAB / name
    return pd.read_csv(path) if path.exists() else None


# ------------------------------------------------------------------ table 2
def table_per_ticker():
    d = read("v2_per_ticker_oos_r2_fixed.csv")
    if d is None:
        return
    d = d.rename(columns={d.columns[0]: "Ticker"}).set_index("Ticker")
    keep = [c for c in d.columns if c != "baseline_ticker_mean"]
    d = d[keep]

    cols = " ".join(["S[table-format=-2.2]"] * len(keep))
    header = " & ".join(label(c) for c in keep)
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Out-of-sample $R^2$ (\%) against the per-ticker mean "
        r"baseline, test window 2023--2025, $n=751$ per ticker. No entry is "
        r"significant after multiplicity correction across the 48 "
        r"model--ticker Diebold--Mariano tests.}",
        r"\label{tab:per-ticker}",
        rf"\begin{{tabular}}{{l {cols}}}", r"\toprule",
        rf"Ticker & {header} \\", r"\midrule",
    ]
    for tkr, row in d.iterrows():
        if str(tkr) == "MEAN":
            lines.append(r"\midrule")
            tkr = r"\textbf{Mean}"
        lines.append(f"{tkr} & " + " & ".join(f"{v:.2f}" for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_per_ticker", "\n".join(lines))


# ------------------------------------------------------------------ table 4
def table_volatility():
    q, dm = read("v2_volatility_qlike_test.csv"), read("v2_volatility_dm_test.csv")
    if q is None:
        return
    bench = "RollingVol20"
    pvals = {}
    if dm is not None:
        for _, r in dm.iterrows():
            if r["model_B"] == bench:
                pvals[r["model_A"]] = r["p_value"]

    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Volatility forecast comparison on an identical sample of "
        r"6008 test observations, scored by QLIKE against the squared return. "
        r"$p$-values are Diebold--Mariano against the rolling-variance "
        r"benchmark with a Newey--West HAC variance. "
        r"$^{***}$, $^{**}$, $^{*}$ denote 1\%, 5\% and 10\%.}",
        r"\label{tab:volatility}",
        r"\begin{tabular}{l S[table-format=-1.4] S[table-format=1.2e-1] c}",
        r"\toprule",
        r"Specification & {QLIKE} & {MSE vs proxy} & {DM $p$} \\", r"\midrule",
    ]
    for _, r in q.sort_values("QLIKE").iterrows():
        p = pvals.get(r["model"])
        pstr = "--" if p is None else f"{p:.4f}{stars(p)}"
        lines.append(f"{label(r['model'])} & {r['QLIKE']:.4f} & "
                     f"{r['MSE_RV']:.2e} & {pstr} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_volatility", "\n".join(lines))


# ------------------------------------------------------------------ table 5
def table_probabilistic():
    d = read("v2_probabilistic_comparison_t_val_iqr.csv")
    dm = read("v2_probabilistic_dm_t_val_iqr.csv")
    if d is None:
        return
    pv = {} if dm is None else dict(zip(dm["model_B"], dm["p_value"]))

    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Probabilistic accuracy and interval calibration, 6008 test "
        r"observations. Log score and CRPS are means; coverage is empirical "
        r"against the nominal level. $p$-values are Diebold--Mariano on the "
        r"log-score differential against the hierarchical model.}",
        r"\label{tab:probabilistic}",
        r"\begin{tabular}{l S[table-format=1.4] S[table-format=1.5] "
        r"S[table-format=1.3] S[table-format=1.3] S[table-format=1.3] "
        r"S[table-format=1.3] c}",
        r"\toprule",
        r"Model & {Log score} & {CRPS} & {50\%} & {80\%} & {90\%} & {95\%} "
        r"& {DM $p$} \\", r"\midrule",
    ]
    for _, r in d.sort_values("logscore", ascending=False).iterrows():
        p = pv.get(r["model"])
        pstr = "--" if p is None else f"{p:.4f}{stars(p)}"
        lines.append(
            f"{label(r['model'])} & {r['logscore']:.4f} & {r['CRPS']:.5f} & "
            f"{r['coverage_50']:.3f} & {r['coverage_80']:.3f} & "
            f"{r['coverage_90']:.3f} & {r['coverage_95']:.3f} & {pstr} " + r"\\")
    lines += [r"\midrule",
              r"\textit{Nominal} & & & 0.500 & 0.800 & 0.900 & 0.950 & \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_probabilistic", "\n".join(lines))


# ------------------------------------------------------------------ table 6
def table_transfer():
    d = read("v2_transfer_summary.csv")
    if d is None:
        return
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Cross-asset transfer, averaged over the eight held-out "
        r"tickers. At $k=0$ the held-out ticker's intercept is integrated over "
        r"its population distribution; at $k>0$ its last $k$ training days are "
        r"revealed and the model refitted.}",
        r"\label{tab:transfer}",
        r"\begin{tabular}{S[table-format=3.0] S[table-format=1.5] "
        r"S[table-format=1.5] S[table-format=1.5] S[table-format=1.5]}",
        r"\toprule",
        r"{$k$ days} & {RMSE} & {Log score} & {CRPS} & {Coverage 90\%} \\",
        r"\midrule",
    ]
    for _, r in d.iterrows():
        lines.append(f"{r['k']:.0f} & {r['RMSE']:.5f} & {r['logscore']:.5f} & "
                     f"{r['CRPS']:.5f} & {r['coverage_90']:.5f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_transfer", "\n".join(lines))


# ------------------------------------------------------------------ table 1
def table_features():
    d = read("feature_dictionary_v2.csv")
    if d is None:
        return
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Feature dictionary. Every model input is declared here; a "
        r"column is a feature because it is listed, not because it is numeric. "
        r"Stationarity is from augmented Dickey--Fuller tests. All retained "
        r"features are observable at the close of $t$, and the target is the "
        r"return from $t$ to $t+1$.}",
        r"\label{tab:features}",
        r"\begin{tabular}{l l l p{0.36\linewidth}}", r"\toprule",
        r"Feature & Family & Status & Note \\", r"\midrule",
    ]
    for _, r in d[d["include"]].iterrows():
        lines.append(f"{label(r['name'])} & {r['family']} & Retained & "
                     f"{r['description']} " + r"\\")
    lines.append(r"\midrule")
    for _, r in d[~d["include"]].iterrows():
        reason = str(r["reason"]).replace("%", r"\%")
        reason = (reason[:118] + "...") if len(reason) > 118 else reason
        lines.append(f"{label(r['name'])} & {r['family']} & Excluded & {reason} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_features", "\n".join(lines))


def main() -> int:
    print("Writing LaTeX tables:")
    table_features()
    table_per_ticker()
    table_volatility()
    table_probabilistic()
    table_transfer()
    print(f"\nSaved to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
