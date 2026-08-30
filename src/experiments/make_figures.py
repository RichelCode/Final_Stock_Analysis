"""
Build the manuscript's figure set.

Five figures, each carrying one claim:

  fig1  the extrapolation failure and its repair       (methodological result)
  fig2  reliability: nominal against empirical coverage (calibration result)
  fig3  PIT histograms, Gaussian against Student-t      (why the t is needed)
  fig4  horseshoe shrinkage by predictor                (what survives selection)
  fig5  where the log score comes from                  (the central result)

Design notes. The categorical palette is the Okabe-Ito subset
blue / green / vermillion / pink, ordered so the worst adjacent pair is
separated by dE 11.0 under deuteranopia and 16.4 under normal vision. Series are
additionally distinguished by marker and line style, so the figures survive
greyscale printing and the pink slot's sub-3:1 contrast against white. Grid and
axes are recessive; no chart uses two y-scales.

Usage
-----
    python -m src.experiments.make_figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJ = REPO_ROOT / "stock_project"
TAB = PROJ / "reports" / "tables"
PRED = PROJ / "reports" / "predictions_v2"
FIG = PROJ / "reports" / "figures_v2"

# Validated categorical palette; see module docstring.
BLUE, GREEN, VERM, PINK = "#0072B2", "#009E73", "#D55E00", "#CC79A7"
INK, MUTED, GRID = "#1a1a1a", "#666666", "#d8d8d8"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.dpi": 160,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def style(ax, ygrid=True):
    ax.spines[["top", "right"]].set_visible(False)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}.pdf / .png")


# ---------------------------------------------------------------- fig 1
def fig1_extrapolation():
    before = pd.read_csv(TAB / "diagnostic_feature_drift_before_fix.csv")
    after = pd.read_csv(TAB / "diagnostic_feature_drift_after_fix.csv")
    summary = pd.read_csv(TAB / "v1_vs_v2_summary.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # (a) v1 drift per feature, coloured by whether v2 kept the feature. Drawing
    # a v2 bar for a removed feature would read as missing data rather than as
    # the deliberate removal it was.
    kept = set(after["feature"])
    b = before.dropna(subset=["z_test_mean"]).nlargest(8, "z_test_mean", keep="all")
    b = b.reindex(b["z_test_mean"].abs().sort_values().index)
    colors = [BLUE if f in kept else VERM for f in b["feature"]]
    ax1.barh(b["feature"], b["z_test_mean"], color=colors, height=0.66, zorder=3)
    ax1.axvline(2.0, color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=2)

    worst_after = after["z_test_mean"].abs().max()
    ax1.set_xlim(0, max(b["z_test_mean"]) * 1.30)
    ax1.text(2.22, len(b) - 0.55, "severe-drift\nthreshold", fontsize=6.3,
             color=MUTED, va="top")
    handles = [plt.Rectangle((0, 0), 1, 1, color=VERM),
               plt.Rectangle((0, 0), 1, 1, color=BLUE)]
    ax1.legend(handles, ["removed in v2", "retained in v2"],
               frameon=False, loc="center right", fontsize=7)
    ax1.set_xlabel("test-window mean, in training SDs")
    ax1.set_title("(a)  Predictor drift, v1 feature set", loc="left", fontsize=8.5)
    ax1.text(0.97, 0.06, f"worst drift after the fix: {worst_after:.2f} SD",
             transform=ax1.transAxes, ha="right", fontsize=6.8, color=INK,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.5))
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="x", color=GRID, lw=0.5, zorder=0); ax1.set_axisbelow(True)

    # (b) what the drift cost, and what removing it recovered
    sm = summary.dropna(subset=["OOS_R2_v2_pct"]).copy()
    sm = sm.loc[sm["OOS_R2_v1_pct"].sort_values().index]
    ypos = np.arange(len(sm))
    ax2.barh(ypos - 0.19, sm["OOS_R2_v1_pct"], height=0.36, color=VERM, zorder=3, label="v1")
    ax2.barh(ypos + 0.19, sm["OOS_R2_v2_pct"], height=0.36, color=BLUE, zorder=3, label="v2")
    ax2.set_yticks(ypos, [m.replace("_", " ") for m in sm["model"]])
    ax2.axvline(0, color=MUTED, lw=0.8, zorder=2)
    ax2.set_xlim(min(sm["OOS_R2_v1_pct"]) * 1.14, 9)
    ax2.set_xlabel("mean out-of-sample $R^2$ vs baseline (%)")
    ax2.set_title("(b)  Cost of the drift, and its repair", loc="left", fontsize=8.5)
    ax2.legend(frameon=False, loc="upper left", fontsize=7, ncol=2)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="x", color=GRID, lw=0.5, zorder=0); ax2.set_axisbelow(True)
    for y, v in zip(ypos, sm["OOS_R2_v2_pct"]):
        ax2.text(v + 0.7, y + 0.19, f"{v:+.2f}", va="center", fontsize=6.5, color=INK)

    fig.tight_layout()
    save(fig, "fig1_extrapolation")


# ---------------------------------------------------------------- fig 2
def fig2_reliability():
    comp = pd.read_csv(TAB / "v2_probabilistic_comparison_t_val_iqr.csv")
    levels = [50, 80, 90, 95]
    want = {
        "bayes_hier_horseshoe": ("Hierarchical Bayes, t + calibration", BLUE, "o", "-"),
        "bayes_hier_uncalibrated": ("Hierarchical Bayes, t, uncalibrated", GREEN, "s", "--"),
        "ridge_garch": ("Ridge + GARCH, Gaussian", VERM, "^", "-."),
        "baseline_gaussian": ("Constant variance, Gaussian", PINK, "D", ":"),
    }

    fig, ax = plt.subplots(figsize=(3.9, 3.9))
    ax.plot([45, 99], [45, 99], color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.text(62, 57.5, "perfect calibration", fontsize=6.5, color=MUTED, rotation=41)

    for key, (label, color, marker, ls) in want.items():
        row = comp[comp["model"] == key]
        if row.empty:
            continue
        emp = [100 * row[f"coverage_{lv}"].iloc[0] for lv in levels]
        ax.plot(levels, emp, color=color, marker=marker, ls=ls, lw=1.6,
                ms=5, mec="white", mew=0.8, label=label, zorder=4)

    ax.set_xlabel("nominal interval level (%)")
    ax.set_ylabel("empirical coverage (%)")
    ax.set_title("Interval calibration", loc="left")
    ax.set_xticks(levels)
    ax.set_xlim(45, 99)
    ax.set_ylim(43, 99)
    # Four series will not fit beside the diagonal without collision, so the
    # legend sits under the axes.
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17),
              handlelength=2.6, ncol=1, borderaxespad=0)
    style(ax)
    fig.tight_layout()
    save(fig, "fig2_reliability")


# ---------------------------------------------------------------- fig 3
def fig3_pit():
    t = pd.read_parquet(PRED / "bayes_hier_horseshoe_t__fixed.parquet")
    g = pd.read_parquet(PRED / "bayes_hier_horseshoe__fixed.parquet")

    df = float(t["df"].iloc[0])
    pit_t = stats.t.cdf((t.y_true - t.y_pred) / t.y_scale, df=df)
    pit_g = stats.norm.cdf((g.y_true - g.y_pred) / g.y_sd)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, (vals, title, color) in zip(axes, [
        (pit_g, "(a)  Gaussian innovations", VERM),
        (pit_t, "(b)  Student-t + calibration", BLUE),
    ]):
        n, _, _ = ax.hist(vals, bins=20, range=(0, 1), color=color, zorder=3,
                          edgecolor="white", linewidth=0.7)
        expected = len(vals) / 20
        ax.axhline(expected, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=4)
        ax.set_ylim(0, max(n) * 1.28)
        ax.set_title(title, loc="left")
        ax.set_xlabel("PIT value")
        # Kolmogorov-Smirnov against the uniform: the formal version of the
        # visual test this panel presents.
        ks = stats.kstest(vals, "uniform")
        ax.text(0.02, 0.94, f"KS = {ks.statistic:.4f}   p = {ks.pvalue:.3g}",
                transform=ax.transAxes, fontsize=6.8, color=INK, va="top")
        style(ax)
    axes[0].set_ylabel("count")
    axes[1].text(0.99, expected * 1.04, "uniform", fontsize=6.5,
                 color=MUTED, ha="right", va="bottom")
    fig.tight_layout()
    save(fig, "fig3_pit")


# ---------------------------------------------------------------- fig 4
def fig4_shrinkage():
    inc = pd.read_csv(TAB / "v2_bayes_inclusion_t.csv").sort_values("weight_1_minus_kappa")

    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    y = np.arange(len(inc))
    colors = [BLUE if e else MUTED for e in inc["excludes_zero"]]
    ax.barh(y, inc["weight_1_minus_kappa"], color=colors, height=0.62, zorder=3)
    ax.set_yticks(y, inc["feature"])
    ax.set_xlabel(r"horseshoe weight  $1-\kappa$")
    ax.set_title("What survives shrinkage", loc="left")
    ax.set_xlim(0, 1)

    hit = inc[inc["excludes_zero"]]
    for yy, f in zip(y, inc["feature"]):
        if f in set(hit["feature"]):
            w = inc.loc[inc.feature == f, "weight_1_minus_kappa"].iloc[0]
            ax.text(w - 0.03, yy, "95% CI excludes 0", va="center", ha="right",
                    fontsize=6.5, color="white", zorder=5)
    style(ax, ygrid=False)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig4_shrinkage")


# ---------------------------------------------------------------- fig 5
def fig5_decomposition():
    levels = pd.read_csv(TAB / "v2_decomposition_levels.csv")
    steps = pd.read_csv(TAB / "v2_decomposition_steps.csv")

    labels = ["Zero mean,\nconstant variance", "+ GARCH\nvariance",
              "+ conditional\nmean (ridge)", "+ t tails and\ncalibration"]
    colors = [PINK, VERM, GREEN, BLUE]
    vals = levels["logscore"].to_numpy()

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    x = np.arange(len(vals))
    ax.bar(x, vals, color=colors, width=0.62, zorder=3)
    ax.set_xticks(x, labels)
    ax.set_ylim(2.15, vals.max() + 0.085)
    ax.set_ylabel("mean log predictive density")
    ax.set_title("Where the probabilistic gain comes from", loc="left")

    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=7.5, color=INK)

    # Each increment is labelled with its share of the total gain as well as its
    # p-value: at n = 6008 the test detects differences far too small to act on,
    # so significance and relevance have to be read separately.
    for i, r in steps.iterrows():
        xi = i + 1
        ax.annotate("", xy=(xi - 0.08, vals[xi]), xytext=(xi - 0.92, vals[xi - 1]),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8,
                                    shrinkA=2, shrinkB=2))
        pstr = "p < 0.00001" if r["p_value"] < 1e-5 else f"p = {r['p_value']:.3f}"
        lift = 0.030 if xi % 2 else 0.058
        ax.text(xi - 0.5, max(vals[xi], vals[xi - 1]) + lift,
                f"{r['gain_nats']:+.3f} nats\n{r['share_of_total_pct']:.1f}% of total\n{pstr}",
                ha="center", fontsize=6.4, color=MUTED, linespacing=1.35)
    style(ax)
    fig.tight_layout()
    save(fig, "fig5_decomposition")


def main() -> int:
    print("Building figures:")
    fig1_extrapolation()
    fig2_reliability()
    fig3_pit()
    fig4_shrinkage()
    fig5_decomposition()
    print(f"\nSaved to {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
