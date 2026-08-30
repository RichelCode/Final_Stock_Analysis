"""
Hierarchical Bayesian return regression with a horseshoe prior and GARCH scale.

Model
-----
For ticker i on day t, with x_it the standardised predictor vector and
sigma_it the GARCH(1,1)-t one-step-ahead conditional SD forecast for t+1:

    y_it   ~ N(alpha_i + x_it' beta,  s^2 * sigma_it^2)
    alpha_i ~ N(mu_alpha, tau_a^2)                       partial pooling
    beta_j | lambda_j, tau_b ~ N(0, lambda_j^2 tau_b^2)  horseshoe
    lambda_j ~ C+(0, 1),  tau_b ~ C+(0, 1)
    mu_alpha ~ N(0, v0),  tau_a^2 ~ InvGamma,  s^2 ~ InvGamma

The GARCH forecast enters as a *known* observation-level scale, so the model
does not have to relearn volatility clustering; s^2 is a single scalar that
calibrates how well that forecast is scaled overall. s^2 near 1 means the GARCH
scale is already right.

Why the horseshoe replaces the v1 correlation screen
----------------------------------------------------
v1 selected the ten predictors with the highest absolute correlation with the
target on the training split, then fitted a flat-prior regression on those. A
trending predictor correlates with almost anything over a four-year window, so
that screen preferentially selected the non-stationary level features that
caused the extrapolation failure. The horseshoe keeps every predictor in the
model and shrinks weak ones toward zero while leaving genuinely strong ones
nearly unshrunk, so selection happens inside the posterior and carries
uncertainty. It is the continuous analogue of the spike-and-slab selection the
manuscript's literature review identifies as the state of the art.

Sampling
--------
Blocked Gibbs. Every conditional is available in closed form, including the
horseshoe, via the auxiliary-variable representation of Makalic & Schmidt
(2016): a half-Cauchy is a scale mixture of inverse gammas, which makes
lambda_j, tau_b and their auxiliaries conditionally inverse gamma.

References
----------
Carvalho, Polson & Scott (2010), The horseshoe estimator for sparse signals.
Makalic & Schmidt (2016), A simple sampler for the horseshoe estimator.
Gelman & Rubin (1992); Vehtari et al. (2021), rank-normalised R-hat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _inv_gamma(rng: np.random.Generator, shape, scale):
    """Draw from InvGamma(shape, scale) as the reciprocal of a Gamma(shape, rate=scale)."""
    return 1.0 / rng.gamma(shape=shape, scale=1.0 / np.asarray(scale))


@dataclass
class Posterior:
    """Retained draws. Leading axis is the draw index, chains already stacked."""

    beta: np.ndarray       # (D, p)
    alpha: np.ndarray      # (D, K)
    mu_alpha: np.ndarray   # (D,)
    tau_a2: np.ndarray     # (D,)  between-ticker variance
    s2: np.ndarray         # (D,)  scale on the GARCH variance
    tau_b2: np.ndarray     # (D,)  global horseshoe shrinkage
    lambda2: np.ndarray    # (D, p) local horseshoe shrinkage
    n_chains: int
    n_draws_per_chain: int
    # Diagonal of X'WX, the data's information about each coefficient. Needed to
    # express shrinkage as a fraction rather than as a raw prior variance.
    xtwx_diag: np.ndarray | None = None

    def stacked(self, name: str) -> np.ndarray:
        return getattr(self, name)

    def by_chain(self, name: str) -> np.ndarray:
        """Reshape a parameter to (chain, draw, ...) for convergence diagnostics."""
        arr = getattr(self, name)
        return arr.reshape(self.n_chains, self.n_draws_per_chain, *arr.shape[1:])


def gibbs_chain(
    X: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    sigma: np.ndarray,
    n_groups: int,
    n_iter: int = 4000,
    burn: int = 2000,
    thin: int = 1,
    seed: int = 0,
    v0: float = 1e4,
    tau_a_scale: float = 0.01,
    a_s: float = 2.0,
    b_s: float = 1e-4,
    init_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Run one chain. `init_scale` disperses the starting point across chains."""
    rng = np.random.default_rng(seed)
    n, p = X.shape

    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-8)
    w = 1.0 / sigma**2                       # known observation precisions
    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    XtWX = Xw.T @ Xw

    group_idx = [np.where(g == j)[0] for j in range(n_groups)]
    group_w = np.array([w[idx].sum() for idx in group_idx])

    # Dispersed inits: chains must start apart for R-hat to be informative.
    beta = rng.normal(0.0, 0.01 * init_scale, p)
    alpha = rng.normal(0.0, 0.01 * init_scale, n_groups)
    mu_alpha = 0.0
    tau_a2 = 0.01 * init_scale
    s2 = 1.0 * init_scale
    lambda2 = np.ones(p)
    tau_b2 = 1.0
    nu = np.ones(p)
    xi = 1.0
    a_aux = 1.0   # auxiliary variable for the half-Cauchy prior on tau_a

    keep = {k: [] for k in
            ("beta", "alpha", "mu_alpha", "tau_a2", "s2", "tau_b2", "lambda2")}

    for it in range(n_iter):
        # ---- beta | rest, horseshoe prior precision on the diagonal ---------
        resid = y - alpha[g]
        XtWr = Xw.T @ (resid * sqrt_w)
        prior_prec = 1.0 / np.maximum(lambda2 * tau_b2, 1e-12)
        prec = XtWX / s2 + np.diag(prior_prec)
        L = np.linalg.cholesky(prec)
        mean = np.linalg.solve(L.T, np.linalg.solve(L, XtWr / s2))
        beta = mean + np.linalg.solve(L.T, rng.standard_normal(p))

        # ---- alpha_i | rest -------------------------------------------------
        y_minus_xb = y - X @ beta
        for j in range(n_groups):
            idx = group_idx[j]
            if idx.size == 0:
                alpha[j] = mu_alpha + np.sqrt(tau_a2) * rng.standard_normal()
                continue
            prec_j = group_w[j] / s2 + 1.0 / tau_a2
            var_j = 1.0 / prec_j
            mean_j = var_j * ((w[idx] @ y_minus_xb[idx]) / s2 + mu_alpha / tau_a2)
            alpha[j] = mean_j + np.sqrt(var_j) * rng.standard_normal()

        # ---- mu_alpha | rest ------------------------------------------------
        prec_mu = n_groups / tau_a2 + 1.0 / v0
        var_mu = 1.0 / prec_mu
        mu_alpha = var_mu * (alpha.sum() / tau_a2) + np.sqrt(var_mu) * rng.standard_normal()

        # ---- tau_a^2 | rest, half-Cauchy(0, tau_a_scale) prior ---------------
        # With only eight tickers the between-ticker SD is weakly identified, and
        # a conventional InvGamma(2, 1e-4) prior simply reproduces its own mode.
        # A half-Cauchy on the SD is the standard recommendation for a variance
        # parameter with few groups (Gelman 2006), and the same auxiliary-variable
        # trick used for the horseshoe keeps it conjugate.
        tau_a2 = max(_inv_gamma(rng, (n_groups + 1) / 2.0,
                                1.0 / a_aux + 0.5 * np.sum((alpha - mu_alpha) ** 2)), 1e-14)
        a_aux = float(_inv_gamma(rng, 1.0, 1.0 / tau_a_scale**2 + 1.0 / tau_a2))

        # ---- s^2 | rest -----------------------------------------------------
        mu_vec = alpha[g] + X @ beta
        rss = float(np.sum(w * (y - mu_vec) ** 2))
        s2 = max(_inv_gamma(rng, a_s + n / 2.0, b_s + 0.5 * rss), 1e-12)

        # ---- horseshoe: lambda_j, nu_j, tau_b, xi ---------------------------
        # Makalic & Schmidt (2016): each half-Cauchy becomes a pair of
        # conditionally inverse-gamma draws, so the whole block is conjugate.
        lambda2 = _inv_gamma(rng, 1.0, 1.0 / nu + beta**2 / (2.0 * tau_b2))
        lambda2 = np.maximum(lambda2, 1e-12)
        nu = _inv_gamma(rng, 1.0, 1.0 + 1.0 / lambda2)

        tau_b2 = max(_inv_gamma(rng, (p + 1) / 2.0,
                                1.0 / xi + 0.5 * np.sum(beta**2 / lambda2)), 1e-12)
        xi = float(_inv_gamma(rng, 1.0, 1.0 + 1.0 / tau_b2))

        if it >= burn and (it - burn) % thin == 0:
            keep["beta"].append(beta.copy())
            keep["alpha"].append(alpha.copy())
            keep["mu_alpha"].append(mu_alpha)
            keep["tau_a2"].append(tau_a2)
            keep["s2"].append(s2)
            keep["tau_b2"].append(tau_b2)
            keep["lambda2"].append(lambda2.copy())

    out = {k: np.asarray(v) for k, v in keep.items()}
    out["_xtwx_diag"] = np.diag(XtWX)
    return out


def fit(
    X: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    sigma: np.ndarray,
    n_groups: int,
    n_chains: int = 4,
    n_iter: int = 4000,
    burn: int = 2000,
    thin: int = 1,
    seed: int = 0,
    **kwargs,
) -> Posterior:
    """Run `n_chains` independent chains from dispersed starting points."""
    chains = []
    for c in range(n_chains):
        chains.append(gibbs_chain(
            X, y, g, sigma, n_groups,
            n_iter=n_iter, burn=burn, thin=thin,
            seed=seed + 1000 * c,
            # Starting points spread over roughly one order of magnitude:
            # dispersed enough for R-hat to detect poor mixing, without pushing
            # the global horseshoe scale into a region it takes thousands of
            # iterations to leave.
            init_scale=float(3.0 ** (c - (n_chains - 1) / 2.0)),
            **kwargs,
        ))

    per_chain = len(chains[0]["mu_alpha"])
    xtwx_diag = chains[0].pop("_xtwx_diag")
    for c in chains[1:]:
        c.pop("_xtwx_diag")
    merged = {k: np.concatenate([c[k] for c in chains], axis=0) for k in chains[0]}
    return Posterior(n_chains=n_chains, n_draws_per_chain=per_chain,
                     xtwx_diag=xtwx_diag, **merged)


# --------------------------------------------------------------------------
# posterior predictive
# --------------------------------------------------------------------------
def predict_seen(X: np.ndarray, g: np.ndarray, sigma: np.ndarray, post: Posterior):
    """Predictive mean and SD for tickers present in the training sample.

    The SD uses the law of total variance, so it includes uncertainty in
    alpha and beta rather than plugging in their posterior means. v1 used a
    plug-in scale and noted in a comment that it omitted alpha uncertainty,
    which made its stated intervals narrower than the model implies.
    """
    sigma = np.maximum(sigma, 1e-8)
    mu_draws = post.alpha[:, g] + post.beta @ X.T          # (D, n)
    mean = mu_draws.mean(axis=0)

    var_param = mu_draws.var(axis=0)                       # uncertainty in the mean
    var_noise = post.s2.mean() * sigma**2                  # expected observation noise
    return mean, np.sqrt(var_param + var_noise), mu_draws


def predict_unseen(X: np.ndarray, sigma: np.ndarray, post: Posterior):
    """Predictive mean and SD for a ticker with no training history.

    alpha for a new ticker is integrated out over its population distribution,
    so the predictive variance picks up tau_a^2 on top of the observation noise.
    This is what partial pooling buys: a usable interval for an asset the model
    has never seen.
    """
    sigma = np.maximum(sigma, 1e-8)
    mu_draws = post.beta @ X.T + post.mu_alpha[:, None]    # (D, n)
    mean = mu_draws.mean(axis=0)

    var_param = mu_draws.var(axis=0)
    var_new_alpha = post.tau_a2.mean()
    var_noise = post.s2.mean() * sigma**2
    return mean, np.sqrt(var_param + var_new_alpha + var_noise), mu_draws


def inclusion_summary(post: Posterior, feature_names: list[str]):
    """Per-coefficient posterior summary plus a horseshoe shrinkage weight.

    kappa_j = 1 / (1 + lambda_j^2 tau_b^2) is the shrinkage factor: near 1 the
    coefficient is shrunk to zero, near 0 it is left alone. 1 - kappa is the
    natural horseshoe analogue of a posterior inclusion probability.
    """
    import pandas as pd

    if post.xtwx_diag is None:
        raise ValueError("Posterior lacks xtwx_diag; refit with the current sampler.")

    # kappa_j = 1 / (1 + I_j * lambda_j^2 * tau_b^2), where I_j = (X'WX)_jj / s^2
    # is the data's information about beta_j. Comparing the prior variance to
    # that information is what makes kappa a shrinkage *fraction*; comparing it
    # to nothing, as an earlier version did, just reports a tiny prior variance
    # and reads as total shrinkage for every coefficient.
    info = post.xtwx_diag[None, :] / post.s2[:, None]
    kappa = 1.0 / (1.0 + info * post.lambda2 * post.tau_b2[:, None])
    lo, hi = np.percentile(post.beta, [2.5, 97.5], axis=0)
    return pd.DataFrame({
        "feature": feature_names,
        "beta_mean": post.beta.mean(axis=0),
        "beta_sd": post.beta.std(axis=0),
        "beta_q2.5": lo,
        "beta_q97.5": hi,
        "excludes_zero": (lo > 0) | (hi < 0),
        "shrinkage_kappa": kappa.mean(axis=0),
        "weight_1_minus_kappa": 1.0 - kappa.mean(axis=0),
    }).sort_values("weight_1_minus_kappa", ascending=False).reset_index(drop=True)
