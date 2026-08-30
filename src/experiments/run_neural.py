"""
Re-run the sequence models on the corrected feature set.

Purpose
-------
The v1 neural results were catastrophic: per-ticker mean out-of-sample R^2 of
-3334% for the GRU, -4618% for the LSTM and -10425% for the GRU-GARCH hybrid.
Those numbers measured extrapolation along non-stationary level features, not
the architectures. This module re-runs the same four families on the declared
stationary feature set so the comparison in the paper is about model class
rather than about a scaling artefact.

Expectation, stated in advance
------------------------------
The linear and tree models land within 0.5 percentage points of the per-ticker
mean baseline, and the Diebold-Mariano tests cannot separate any of them from
it. A sequence model that reaches the same place is the correct result, not a
disappointing one, and it closes off the obvious objection that the null result
is an artefact of using models too simple to find the signal.

Design
------
Sequences of 30 trading days per ticker, features standardised on the training
split only, targets left on their natural scale. Early stopping on the
validation split, fixed seeds, and identical training budget across
architectures so the comparison is about inductive bias rather than tuning
effort. Predictions are audited for scale before any metric is reported.

Usage
-----
    python -m src.experiments.run_neural
    python -m src.experiments.run_neural --models lstm gru
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTHONHASHSEED", "0")

import tensorflow as tf
from tensorflow import keras

from src.eval.metrics import dm_squared_error, oos_r2, point_metrics
from src.features.dictionary import TARGET, model_features

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJ = REPO_ROOT / "stock_project"
DEFAULT_PANEL = PROJ / "data" / "processed" / "panel_v2_garch.parquet"
PRED_DIR = PROJ / "reports" / "predictions_v2"
TAB_DIR = PROJ / "reports" / "tables"

SEQ_LEN = 30
SEED = 20260830
ID_COLS = ["Date", "target_date", "Ticker"]

# A daily return forecast whose mean is more than half a return SD from the
# realised mean, or whose dispersion exceeds the realised SD, is not a forecast.
MAX_BIAS_SD = 0.5
MAX_DISPERSION_RATIO = 1.0


def make_sequences(df: pd.DataFrame, features: list[str], seq_len: int):
    """Windows of length seq_len ending at t-1, predicting the target at t.

    The window deliberately stops one day short of the target row, so no feature
    observed on the target's own date enters the input.
    """
    X, y, meta = [], [], []
    for ticker, g in df.sort_values(["Ticker", "Date"]).groupby("Ticker", sort=False):
        vals = g[features].to_numpy(np.float32)
        targ = g[TARGET].to_numpy(np.float32)
        rows = g[ID_COLS + ["split"]].reset_index(drop=True)
        for i in range(seq_len, len(g)):
            X.append(vals[i - seq_len:i])
            y.append(targ[i])
            meta.append(rows.iloc[i])
    return np.stack(X), np.asarray(y, np.float32), pd.DataFrame(meta).reset_index(drop=True)


def build_lstm(shape):
    return keras.Sequential([keras.Input(shape=shape),
                             keras.layers.LSTM(48), keras.layers.Dropout(0.2),
                             keras.layers.Dense(1)])


def build_gru(shape):
    return keras.Sequential([keras.Input(shape=shape),
                             keras.layers.GRU(48), keras.layers.Dropout(0.2),
                             keras.layers.Dense(1)])


def build_cnn(shape):
    return keras.Sequential([keras.Input(shape=shape),
                             keras.layers.Conv1D(48, 3, activation="relu", padding="causal"),
                             keras.layers.GlobalAveragePooling1D(),
                             keras.layers.Dropout(0.2), keras.layers.Dense(1)])


def build_transformer(shape):
    inp = keras.Input(shape=shape)
    x = keras.layers.Dense(32)(inp)
    attn = keras.layers.MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
    x = keras.layers.LayerNormalization()(x + attn)
    ff = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.LayerNormalization()(x + ff)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(0.2)(x)
    return keras.Model(inp, keras.layers.Dense(1)(x))


BUILDERS = {"lstm": build_lstm, "gru": build_gru,
            "cnn1d": build_cnn, "transformer": build_transformer}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    ap.add_argument("--models", nargs="+", default=list(BUILDERS))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--garch", action="store_true",
                    help="Add the GARCH conditional SD as an input (hybrid).")
    args = ap.parse_args()

    keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    features = model_features(include_garch=args.garch)
    panel = pd.read_parquet(args.panel).dropna(subset=features + [TARGET, "split"])
    tag = "hybrid" if args.garch else "base"
    print(f"Features ({len(features)}): {', '.join(features)}\n")

    X, y, meta = make_sequences(panel, features, SEQ_LEN)
    tr, va, te = (meta["split"] == s for s in ("train", "val", "test"))
    tr, va, te = tr.to_numpy(), va.to_numpy(), te.to_numpy()

    # Standardise on training windows only.
    flat = X[tr].reshape(-1, X.shape[-1])
    mu, sd = flat.mean(0), flat.std(0)
    sd[sd == 0] = 1.0
    Xs = ((X - mu) / sd).astype(np.float32)
    print(f"Sequences: train {tr.sum()}  val {va.sum()}  test {te.sum()}")

    ref = pd.read_parquet(PRED_DIR / "baseline_ticker_mean__fixed.parquet")
    ref = ref[ref.split == "test"].set_index(["Date", "Ticker"])["y_pred"]

    rows = []
    for name in args.models:
        keras.utils.set_random_seed(SEED)
        t0 = time.time()
        model = BUILDERS[name]((SEQ_LEN, X.shape[-1]))
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        model.fit(Xs[tr], y[tr], validation_data=(Xs[va], y[va]),
                  epochs=args.epochs, batch_size=args.batch, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(
                      monitor="val_loss", patience=8, restore_best_weights=True)])

        pred = model.predict(Xs[te], verbose=0).ravel()
        yte = y[te]
        model_name = f"{name}_{tag}"

        out = meta.loc[te, ID_COLS].copy()
        out["split"] = "test"
        out["model"] = model_name
        out["y_true"] = yte
        out["y_pred"] = pred
        out["residual"] = yte - pred
        out.to_parquet(PRED_DIR / f"{model_name}__fixed.parquet", index=False)

        idx = pd.MultiIndex.from_frame(out[["Date", "Ticker"]])
        r = ref.reindex(idx).to_numpy()
        ok = np.isfinite(r)

        bias_sd = (pred.mean() - yte.mean()) / yte.std()
        disp = pred.std() / yte.std()
        sane = abs(bias_sd) <= MAX_BIAS_SD and disp <= MAX_DISPERSION_RATIO

        m = point_metrics(yte, pred)
        dm = dm_squared_error(yte[ok], pred[ok], r[ok])
        rows.append({"model": model_name, **m,
                     "OOS_R2_pct": 100 * oos_r2(yte[ok], pred[ok], r[ok]),
                     "bias_in_true_sd": bias_sd, "dispersion_ratio": disp,
                     "scale_sane": sane,
                     "DM_stat_vs_baseline": dm["DM_stat"], "DM_p": dm["p_value"],
                     "seconds": round(time.time() - t0, 1)})
        print(f"  {model_name:20s} RMSE={m['RMSE']:.5f} OOS_R2={rows[-1]['OOS_R2_pct']:+.3f}% "
              f"bias={bias_sd:+.3f}sd disp={disp:.3f} "
              f"{'OK' if sane else 'SCALE FAILURE'}  ({rows[-1]['seconds']:.0f}s)")

    table = pd.DataFrame(rows)
    out_path = TAB_DIR / f"v2_neural_metrics_{tag}.csv"
    table.to_csv(out_path, index=False)

    show = ["model", "RMSE", "OOS_R2_pct", "DirAcc", "bias_in_true_sd",
            "dispersion_ratio", "scale_sane", "DM_p"]
    with pd.option_context("display.width", 220):
        print(f"\n{table[show].round(5).to_string(index=False)}")
    print(f"\nSaved: {out_path.name}")
    return 0 if table["scale_sane"].all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
