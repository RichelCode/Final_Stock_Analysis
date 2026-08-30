#!/usr/bin/env bash
#
# Reproduce every number, table and figure in the paper, in dependency order.
#
#   ./run_all.sh              full reproduction (about 40 minutes)
#   ./run_all.sh --fast       skip the neural sweep and shorten the sampler
#   ./run_all.sh --stage 4    run one stage only
#
# Expects the virtual environment described in README.md:
#
#   python -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt
#   pip install -r requirements-deep.txt     # stage 9 only
#
set -euo pipefail

PY="${PYTHON:-python}"
FAST=0
ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fast)  FAST=1; shift ;;
    --stage) ONLY="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

run () {  # run <stage-number> <description> <command...>
  local n="$1" desc="$2"; shift 2
  if [[ -n "$ONLY" && "$ONLY" != "$n" ]]; then return 0; fi
  printf '\n\033[1m[%s] %s\033[0m\n' "$n" "$desc"
  "$@"
}

# Stages 1 and 2 audit the uncorrected pipeline and are expected to exit
# non-zero: that is the finding, not a failure, so their status is not fatal.
audit () {
  local n="$1" desc="$2"; shift 2
  if [[ -n "$ONLY" && "$ONLY" != "$n" ]]; then return 0; fi
  printf '\n\033[1m[%s] %s\033[0m\n' "$n" "$desc"
  if "$@"; then echo "  (clean)"; else echo "  (findings reported, as expected)"; fi
}

BAYES_ARGS=(--dist t --calibrate val --calib-stat iqr)
if [[ "$FAST" == "1" ]]; then
  BAYES_ARGS+=(--iter 8000 --burn 4000 --thin 2)
fi

audit 1 "Audit the v1 panel for predictor drift" \
      "$PY" -m src.diagnostics.feature_drift
audit 2 "Audit v1 predictions for extrapolation" \
      "$PY" -m src.diagnostics.prediction_sanity

run 3 "Export the feature dictionary" \
    "$PY" -m src.features.dictionary
run 4 "Build the corrected panel" \
    "$PY" -m src.data.build_panel_v2
run 5 "Confirm the drift is resolved" \
    "$PY" -m src.diagnostics.feature_drift \
    --panel stock_project/data/processed/panel_v2.parquet --label after_fix
run 6 "Estimate GARCH features" \
    "$PY" -m src.models.garch
run 7 "Baselines" \
    "$PY" -m src.experiments.run_baselines
run 8 "Ridge and random forest" \
    "$PY" -m src.experiments.run_linear_tree

if [[ "$FAST" == "1" ]]; then
  echo $'\n\033[1m[9] Sequence models — skipped (--fast)\033[0m'
else
  run 9 "Sequence models, tuned and untuned" bash -c \
      "$PY -m src.experiments.run_neural && $PY -m src.experiments.run_neural --no-tune"
fi

run 10 "Volatility comparison" \
    "$PY" -m src.experiments.run_volatility
run 11 "Hierarchical Bayesian model" \
    "$PY" -m src.experiments.run_bayes "${BAYES_ARGS[@]}"
run 12 "Cross-asset transfer" \
    "$PY" -m src.experiments.run_transfer
run 13 "Decomposition of the predictive gain" \
    "$PY" -m src.experiments.run_decomposition
run 14 "Per-ticker tables" \
    "$PY" -m src.experiments.make_paper_tables
run 15 "v1 against v2 comparison" \
    "$PY" -m src.experiments.compare_v1_v2
run 16 "Figures" \
    "$PY" -m src.experiments.make_figures
run 17 "LaTeX tables" \
    "$PY" -m src.experiments.make_latex_tables

printf '\n\033[1mDone.\033[0m Tables in stock_project/reports/tables,'
printf ' figures in stock_project/reports/figures_v2,\n'
printf 'LaTeX in paper/tables. Build the paper with: cd paper && tectonic main.tex\n'
