"""
Shared evaluation metrics for point and probabilistic return forecasts.

Every model in the project scores through this module, so comparisons are made
on identical definitions and an identical sample. Earlier versions computed
metrics inline in notebook cells, which is how two variants of the same LSTM
ended up in reports/ with incompatible numbers.

Point accuracy
--------------
RMSE, MAE, correlation and directional accuracy, plus out-of-sample R^2 against
a stated reference forecast. For daily returns the reference must be a real
baseline (the per-ticker training mean), not the test-sample mean, which would
use information unavailable at forecast time.

Probabilistic accuracy
----------------------
Gaussian log score, CRPS and interval coverage. Log score rewards sharpness,
CRPS is less sensitive to outliers, and coverage measures calibration. A model
can win on one and lose on another, and the paper should report all three.

Significance
------------
`diebold_mariano` tests equal predictive accuracy between two forecasts using a
Newey-West HAC variance, which is required because daily loss differentials are
serially correlated. Without it, a difference in the third decimal place of
RMSE is not evidence of anything.

References
----------
Diebold & Mariano (1995), Comparing predictive accuracy, JBES 13(3).
Gneiting & Raftery (2007), Strictly proper scoring rules, JASA 102(477).
Patton (2011), Volatility forecast comparison using imperfect proxies, J. Econometrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import special, stats

SQRT_PI = np.sqrt(np.pi)


# --------------------------------------------------------------------------
# point accuracy
# --------------------------------------------------------------------------
def point_metrics(y_true, y_pred) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float).ravel()
    f = np.asarray(y_pred, dtype=float).ravel()
    err = y - f
    return {
        "n": int(y.size),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        # Correlation is undefined for a constant forecast, which several
        # baselines are by construction.
        "Corr": float(np.corrcoef(y, f)[0, 1]) if f.std() > 0 and y.std() > 0 else np.nan,
        "DirAcc": float(np.mean((y > 0) == (f > 0))),
        "pred_mean": float(f.mean()),
        "pred_sd": float(f.std()),
    }


def oos_r2(y_true, y_pred, y_reference) -> float:
    """Out-of-sample R^2 against a reference forecast (Campbell-Thompson).

    Positive means the model beats the reference; zero means it matches it.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    f = np.asarray(y_pred, dtype=float).ravel()
    r = np.asarray(y_reference, dtype=float).ravel()
    sse_ref = np.sum((y - r) ** 2)
    if sse_ref == 0:
        return np.nan
    return float(1.0 - np.sum((y - f) ** 2) / sse_ref)


# --------------------------------------------------------------------------
# probabilistic accuracy
# --------------------------------------------------------------------------
def log_score(y_true, mu, sigma) -> np.ndarray:
    """Pointwise Gaussian log predictive density. Higher is better."""
    y = np.asarray(y_true, dtype=float).ravel()
    m = np.asarray(mu, dtype=float).ravel()
    s = np.maximum(np.asarray(sigma, dtype=float).ravel(), 1e-12)
    return stats.norm.logpdf(y, loc=m, scale=s)


def crps_gaussian(y_true, mu, sigma) -> np.ndarray:
    """Pointwise CRPS for a Gaussian predictive distribution. Lower is better.

    Closed form: sigma * [ z(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ].
    """
    y = np.asarray(y_true, dtype=float).ravel()
    m = np.asarray(mu, dtype=float).ravel()
    s = np.maximum(np.asarray(sigma, dtype=float).ravel(), 1e-12)
    z = (y - m) / s
    return s * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1.0 / SQRT_PI)


def log_score_t(y_true, mu, scale, df) -> np.ndarray:
    """Pointwise Student-t log predictive density. Higher is better.

    `scale` is the distribution's scale parameter, not its standard deviation.
    A t with df degrees of freedom has SD = scale * sqrt(df / (df - 2)).
    """
    y = np.asarray(y_true, dtype=float).ravel()
    m = np.asarray(mu, dtype=float).ravel()
    sc = np.maximum(np.asarray(scale, dtype=float).ravel(), 1e-12)
    return stats.t.logpdf(y, df=df, loc=m, scale=sc)


def crps_t(y_true, mu, scale, df) -> np.ndarray:
    """Pointwise CRPS for a Student-t predictive distribution. Lower is better.

    Closed form from Jordan, Kruger & Lerch (2019). Requires df > 1.
    """
    if df <= 1:
        raise ValueError(f"CRPS for a Student-t requires df > 1 (got {df}).")

    y = np.asarray(y_true, dtype=float).ravel()
    m = np.asarray(mu, dtype=float).ravel()
    sc = np.maximum(np.asarray(scale, dtype=float).ravel(), 1e-12)
    z = (y - m) / sc

    # The final term is constant in z and centres the score.
    const = (2.0 * np.sqrt(df) / (df - 1.0)) * (
        special.beta(0.5, df - 0.5) / special.beta(0.5, df / 2.0) ** 2
    )
    return sc * (
        z * (2.0 * stats.t.cdf(z, df) - 1.0)
        + 2.0 * stats.t.pdf(z, df) * (df + z**2) / (df - 1.0)
        - const
    )


def coverage_t(y_true, mu, scale, df, level: float = 0.90) -> float:
    y = np.asarray(y_true, dtype=float).ravel()
    m = np.asarray(mu, dtype=float).ravel()
    sc = np.maximum(np.asarray(scale, dtype=float).ravel(), 1e-12)
    q = stats.t.ppf(0.5 + level / 2.0, df)
    return float(np.mean((y >= m - q * sc) & (y <= m + q * sc)))


def pit_t(y_true, mu, scale, df) -> np.ndarray:
    y = np.asarray(y_true, dtype=float).ravel()
    m = np.asarray(mu, dtype=float).ravel()
    sc = np.maximum(np.asarray(scale, dtype=float).ravel(), 1e-12)
    return stats.t.cdf(y, df=df, loc=m, scale=sc)


def probabilistic_metrics_t(y_true, mu, scale, df,
                            levels=(0.50, 0.80, 0.90, 0.95)) -> dict:
    out = {
        "n": int(np.asarray(y_true).size),
        "df": float(df),
        "logscore": float(np.mean(log_score_t(y_true, mu, scale, df))),
        "CRPS": float(np.mean(crps_t(y_true, mu, scale, df))),
    }
    for lv in levels:
        out[f"coverage_{int(lv * 100)}"] = coverage_t(y_true, mu, scale, df, lv)
    return out


def coverage(y_true, mu, sigma, level: float = 0.90) -> float:
    """Empirical coverage of the central interval at `level`."""
    y = np.asarray(y_true, dtype=float).ravel()
    m = np.asarray(mu, dtype=float).ravel()
    s = np.maximum(np.asarray(sigma, dtype=float).ravel(), 1e-12)
    z = stats.norm.ppf(0.5 + level / 2.0)
    return float(np.mean((y >= m - z * s) & (y <= m + z * s)))


def pit(y_true, mu, sigma) -> np.ndarray:
    """Probability integral transform values.

    Under a correctly calibrated predictive distribution these are uniform on
    [0, 1]; a histogram of them is the standard calibration diagnostic.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    m = np.asarray(mu, dtype=float).ravel()
    s = np.maximum(np.asarray(sigma, dtype=float).ravel(), 1e-12)
    return stats.norm.cdf(y, loc=m, scale=s)


def probabilistic_metrics(y_true, mu, sigma, levels=(0.50, 0.80, 0.90, 0.95)) -> dict:
    out = {
        "n": int(np.asarray(y_true).size),
        "logscore": float(np.mean(log_score(y_true, mu, sigma))),
        "CRPS": float(np.mean(crps_gaussian(y_true, mu, sigma))),
    }
    for lv in levels:
        out[f"coverage_{int(lv * 100)}"] = coverage(y_true, mu, sigma, lv)
    return out


# --------------------------------------------------------------------------
# significance
# --------------------------------------------------------------------------
def newey_west_var(d: np.ndarray, lag: int | None = None) -> float:
    """HAC variance of the mean of `d`, Bartlett kernel.

    Default lag follows the common rule floor(4*(n/100)^(2/9)).
    """
    d = np.asarray(d, dtype=float).ravel()
    n = d.size
    if lag is None:
        lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    lag = max(0, min(lag, n - 1))

    dc = d - d.mean()
    gamma0 = float(dc @ dc) / n
    total = gamma0
    for k in range(1, lag + 1):
        gamma_k = float(dc[k:] @ dc[:-k]) / n
        total += 2.0 * (1.0 - k / (lag + 1.0)) * gamma_k
    # A negative HAC estimate is possible in small samples; fall back to gamma0.
    return total if total > 0 else gamma0


def diebold_mariano(loss_a, loss_b, lag: int | None = None) -> dict:
    """Test equal predictive accuracy between two forecasts.

    Pass per-observation losses (squared error, absolute error, QLIKE, negative
    log score). The null is E[loss_a - loss_b] = 0. A negative statistic favours
    model A, i.e. A has the lower loss.

    Uses the Harvey-Leybourne-Newbold small-sample correction and a t
    reference distribution with n-1 degrees of freedom.
    """
    a = np.asarray(loss_a, dtype=float).ravel()
    b = np.asarray(loss_b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"Loss vectors must align: {a.shape} vs {b.shape}")

    d = a - b
    n = d.size
    if n < 10:
        raise ValueError(f"Too few observations for a DM test (n={n}).")

    # Identical forecasts give a degenerate differential: there is nothing to
    # test, and NaN is the honest answer rather than an arbitrary statistic.
    var_d = newey_west_var(d, lag=lag)
    if var_d <= 0 or np.allclose(d, 0.0):
        return {
            "DM_stat": np.nan, "p_value": np.nan,
            "mean_diff": float(d.mean()), "n": n, "favours": "tie",
        }

    dm = d.mean() / np.sqrt(var_d / n)

    # Harvey, Leybourne & Newbold (1997) correction for one-step forecasts.
    h = 1
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * correction
    p = 2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1))

    return {
        "DM_stat": float(dm_hln),
        "p_value": float(p),
        "mean_diff": float(d.mean()),
        "n": n,
        "favours": "A" if d.mean() < 0 else "B",
    }


def dm_squared_error(y_true, pred_a, pred_b, lag: int | None = None) -> dict:
    """Diebold-Mariano on squared-error loss."""
    y = np.asarray(y_true, dtype=float).ravel()
    return diebold_mariano((y - np.asarray(pred_a).ravel()) ** 2,
                           (y - np.asarray(pred_b).ravel()) ** 2, lag=lag)


def qlike(realised_var, forecast_var, eps: float = 1e-12) -> np.ndarray:
    """QLIKE loss, robust to a noisy variance proxy (Patton 2011). Lower is better."""
    rv = np.asarray(realised_var, dtype=float).ravel()
    vh = np.maximum(np.asarray(forecast_var, dtype=float).ravel(), eps)
    return np.log(vh) + rv / vh


def summarise(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
