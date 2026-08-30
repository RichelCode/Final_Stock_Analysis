"""
The two evaluation protocols used throughout the project.

Fixed origin
------------
Fit once on the training split, freeze everything (including the feature
scaler), and score the whole test window. This is the protocol the original
notebooks used. It is the correct choice only when features are stationary: any
permanent shift in a predictor's mean between fitting and scoring arrives as an
extreme standardised input the model has never seen.

Expanding window
----------------
Walk forward month by month. At each fold, refit on everything strictly before
the fold start and predict that month only. The scaler is refitted with the
model, so a predictor whose mean has moved is re-centred rather than
extrapolated. This is what a forecaster could actually have done in real time.

Running both is deliberate. The gap between them measures how much of a
model's out-of-sample error is genuine forecast error and how much is an
artefact of freezing a scaler across a regime change, which is a reportable
result in its own right.

Leakage rules enforced here
---------------------------
* A fold trains only on rows whose `target_date` falls strictly before the fold
  start, so the outcome being predicted is never in the training sample.
* Scalers and imputers are fitted inside the fold, never on the full panel.
* Rows are never shuffled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import pandas as pd


class Estimator(Protocol):
    """Minimal sklearn-style interface a model must expose."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Estimator": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


EstimatorFactory = Callable[[], Estimator]


@dataclass
class RunResult:
    predictions: pd.DataFrame  # Date, target_date, Ticker, split, model, y_true, y_pred
    n_folds: int


ID_COLS = ["Date", "target_date", "Ticker"]


def _frame(rows: pd.DataFrame, y_true, y_pred, model: str, split: str) -> pd.DataFrame:
    out = rows[ID_COLS].copy()
    out["split"] = split
    out["model"] = model
    out["y_true"] = np.asarray(y_true, dtype=float)
    out["y_pred"] = np.asarray(y_pred, dtype=float)
    out["residual"] = out["y_true"] - out["y_pred"]
    return out


def fixed_origin(
    panel: pd.DataFrame,
    features: list[str],
    target: str,
    factory: EstimatorFactory,
    model_name: str,
    train_splits: tuple[str, ...] = ("train",),
    score_split: str = "test",
) -> RunResult:
    """Fit once on `train_splits`, score `score_split`."""
    cols = features + [target]
    tr = panel[panel["split"].isin(train_splits)].dropna(subset=cols)
    te = panel[panel["split"] == score_split].dropna(subset=cols)
    if tr.empty or te.empty:
        raise ValueError(f"{model_name}: empty train ({len(tr)}) or score ({len(te)}) sample.")

    est = factory()
    est.fit(tr[features].to_numpy(float), tr[target].to_numpy(float))
    pred = est.predict(te[features].to_numpy(float))

    return RunResult(_frame(te, te[target], pred, model_name, score_split), n_folds=1)


def expanding_window(
    panel: pd.DataFrame,
    features: list[str],
    target: str,
    factory: EstimatorFactory,
    model_name: str,
    score_split: str = "test",
    min_train_rows: int = 250,
) -> RunResult:
    """Walk forward monthly over `score_split`, refitting on all prior data."""
    cols = features + [target]
    df = panel.dropna(subset=cols).copy()
    df["target_date"] = pd.to_datetime(df["target_date"])

    scored = df[df["split"] == score_split]
    if scored.empty:
        raise ValueError(f"{model_name}: no rows in split '{score_split}'.")

    months = sorted(scored["target_date"].dt.to_period("M").unique())

    frames, n_folds = [], 0
    for month in months:
        start, end = month.start_time, month.end_time

        # Strictly before the fold start: the outcome being predicted is the
        # return realised on target_date, so that is the date that must be
        # excluded, not the feature date.
        tr = df[df["target_date"] < start]
        te = scored[(scored["target_date"] >= start) & (scored["target_date"] <= end)]

        if len(tr) < min_train_rows or te.empty:
            continue

        est = factory()
        est.fit(tr[features].to_numpy(float), tr[target].to_numpy(float))
        pred = est.predict(te[features].to_numpy(float))

        fold = _frame(te, te[target], pred, model_name, score_split)
        fold["fold"] = str(month)
        frames.append(fold)
        n_folds += 1

    if not frames:
        raise ValueError(f"{model_name}: no usable folds.")
    return RunResult(pd.concat(frames, ignore_index=True), n_folds=n_folds)


def run_both(
    panel: pd.DataFrame,
    features: list[str],
    target: str,
    factory: EstimatorFactory,
    model_name: str,
    **kwargs,
) -> dict[str, RunResult]:
    """Run a model under both protocols and label each result."""
    return {
        "fixed_origin": fixed_origin(
            panel, features, target, factory, f"{model_name}__fixed", **kwargs
        ),
        "expanding_window": expanding_window(
            panel, features, target, factory, f"{model_name}__expanding", **kwargs
        ),
    }
