#   Copyright 2024 - 2026 Brian Curry / Vector1 Research
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Optional JAX backend: the model's log posterior as a differentiable function, and NUTS.

Installed with ``pip install "papilon[jax]"``. Nothing here is imported unless a model
is fitted with ``sampler="nuts"`` (or ``"auto"`` picks it), so the default install
needs only NumPy and SciPy.

The functions built here compute the same collapsed posterior as
``MMM._log_post_batch`` and ``GeoMMM._log_post_batch``, in the same unconstrained
parametrization: baselines and noise (and, in regional models, each region's effects)
are integrated out in closed form, and only the curve parameters are sampled. Draws
from NUTS are therefore unpacked by the same NumPy code as draws from the gate, and
the two samplers can be checked against each other on any model small enough for both.

Importing the backend switches JAX to 64-bit floats (``jax_enable_x64``) for the whole
process: the closed-form marginal likelihood loses too much precision in 32 bits.
"""
from __future__ import annotations

import importlib.util

import numpy as np

from . import model as _m

__all__ = ["available", "log_posterior", "sample"]

CHAIN_METHOD = "vectorized"   # chains advance together in one compiled program
DENSE_MASS_DIM = 30  # up to this many sampled parameters NUTS adapts a dense mass matrix, above it a diagonal one

_HINT = ('the NUTS sampler needs the optional JAX backend; install it with pip install "papilon[jax]", '
         'or fit with sampler="gate"')
_LOG_2PI = float(np.log(2 * np.pi))


def available() -> bool:
    """True when the optional backend (JAX and NumPyro) can be imported."""
    return all(importlib.util.find_spec(m) is not None for m in ("jax", "numpyro"))


def _jax():
    try:
        import jax
        import numpyro  # noqa: F401
    except ImportError as err:
        raise ImportError(_HINT) from err
    jax.config.update("jax_enable_x64", True)
    return jax


# ----------------------------------------------------------------- pieces
def _pieces(model, fam):
    """Unpack, carryover kernel and saturation for one parameter vector, traced by JAX."""
    jax = _jax()
    jnp = jax.numpy
    from jax.scipy.special import betaln, expit

    chans, lay = model.channels, model._layout(fam)
    K, L, max_peak = len(chans), model.l_max, model.l_max - 2
    lags, edges = jnp.arange(L, dtype=float), jnp.arange(L + 1, dtype=float)

    def norm(z, mu, sd):
        return -np.log(sd) - 0.5 * _LOG_2PI - (z - mu) ** 2 / (2 * sd * sd)

    def beta_lpdf(x, a, b):
        return (a - 1) * jnp.log(x) + (b - 1) * jnp.log1p(-x) - betaln(a, b)

    def power(x, s):
        """x ** s for x >= 0, with a finite gradient in s where x is zero."""
        pos = x > 0
        return jnp.where(pos, jnp.exp(s * jnp.log(jnp.where(pos, x, 1.0))), 0.0)

    def unpack(z):
        P = {k: [None] * K for k in ("alpha", "peak", "wk", "wl", "half_sat", "beta", "shape", "rf_k", "rf_s", "tau")}
        tv = [[] for _ in range(K)]
        lp, i = 0.0, 0
        for j, (c, names) in enumerate(zip(chans, lay)):
            for nm in names:
                x = z[i]; i += 1
                if nm == "za":
                    a = expit(x); P["alpha"][j] = a
                    pa, pb = model.alpha_prior[c]
                    lp = lp + beta_lpdf(a, pa, pb) + jnp.log(a) + jnp.log1p(-a)
                elif nm == "zu":
                    u = expit(x); P["peak"][j] = max_peak * u
                    lp = lp + beta_lpdf(u, 2.0, 2.0) + jnp.log(u) + jnp.log1p(-u)
                elif nm == "zwk":
                    P["wk"][j] = jnp.exp(x); lp = lp + norm(x, _m._WK_MU, _m._WK_SD)
                elif nm == "zwl":
                    P["wl"][j] = jnp.exp(x); lp = lp + norm(x, _m._WL_MU, _m._WL_SD)
                elif nm == "zh":
                    P["half_sat"][j] = jnp.exp(x)          # log-normal(0, 0.5) on half_sat, Jacobian included
                    lp = lp + (-x - np.log(0.5) - 0.5 * _LOG_2PI - x * x / (2 * 0.25)) + x
                elif nm == "zb":
                    b = jnp.exp(x); P["beta"][j] = b
                    s = model._b_scale[c]
                    lp = lp + (np.log(2.0) - 0.5 * _LOG_2PI - np.log(s) - b * b / (2 * s * s)) + x
                elif nm == "zs":
                    P["shape"][j] = jnp.exp(x); lp = lp + norm(x, _m._SHAPE_MU, _m._SHAPE_SD)
                elif nm == "zrk":
                    P["rf_k"][j] = jnp.exp(x); lp = lp + norm(x, np.log(model._rf_stats[c][1]), 0.5)
                elif nm == "zrs":
                    P["rf_s"][j] = jnp.exp(x); lp = lp + norm(x, _m._RFS_MU, _m._RFS_SD)
                elif nm == "zg":
                    tv[j].append(x); lp = lp + norm(x, 0.0, _m._TV_SD)
                elif nm == "zt":
                    P["tau"][j] = jnp.exp(x); lp = lp + norm(x, np.log(0.3), 0.5)
        return P, tv, lp

    def kernel(P, j, kind):
        if kind == "weibull":
            cdf = 1.0 - jnp.exp(-power(edges / P["wl"][j], P["wk"][j]))
            kern = jnp.diff(cdf)
        elif P["peak"][j] is None:
            kern = P["alpha"][j] ** lags
        else:
            kern = P["alpha"][j] ** ((lags - P["peak"][j]) ** 2)
        kern = kern + 1e-12
        return kern / kern.sum()

    def adstock(P, j, kind, src, rows):
        kern = kernel(P, j, kind)
        if src[0] == "rf":
            _, R, F, r, f = src
            kap, s = P["rf_k"][j], P["rf_s"][j]
            ks = kap ** s
            fs = power(jnp.asarray(F[rows]), s)
            g_all = power(jnp.asarray(f), s)
            g_all = g_all / (g_all + ks)
            nrm = jnp.maximum(jnp.mean(jnp.asarray(r) * g_all), 1e-12)
            return (jnp.asarray(R[rows]) * (fs / (fs + ks))) @ kern / nrm
        return jnp.asarray(src[1][rows]) @ kern

    def saturate(ad, P, j, kind):
        hs = P["half_sat"][j]
        if kind == "logistic":
            e = jnp.exp(-ad / hs)
            return (1 - e) / (1 + e)
        if kind == "hill":
            return ad / (ad + hs)
        s = P["shape"][j]
        a_s = power(ad, s)
        return a_s / (a_s + hs ** s)

    return jnp, unpack, kernel, adstock, saturate


# ------------------------------------------------------------ national
def _national(model, fam):
    jnp, unpack, kernel, adstock, saturate = _pieces(model, fam)
    chans, K, T = model.channels, len(model.channels), len(model.y)
    all_rows = np.arange(T)
    y, proj, vn_inv = jnp.asarray(model.y), jnp.asarray(model._proj), jnp.asarray(model._vn_inv)

    def effect(P, tv, j, src, rows):
        c = chans[j]
        out = P["beta"][j] * saturate(adstock(P, j, fam.carry[c], src, rows), P, j, fam.sat[c])
        if c in fam.tv:
            out = out * jnp.exp(jnp.asarray(model._tv_rows(rows)) @ jnp.stack(tv[j]))
        return out

    def logp(z):
        P, tv, lp = unpack(z)
        eff = [effect(P, tv, j, model._src[j], all_rows) for j in range(K)]
        for j, m, sd in model._roi_idx:
            roas = eff[j].sum() * model.y_scale / model._spend_tot[j]
            lp = lp - 0.5 * ((roas - m) / sd) ** 2 - np.log(sd) - 0.5 * _LOG_2PI
        for t in model._lift:
            pred = (effect(P, tv, t["j"], t["src_cf"], t["rows"]) - effect(P, tv, t["j"], t["src"], t["rows"])).sum()
            lp = lp - 0.5 * ((pred - t["lift"]) / t["se"]) ** 2 - np.log(t["se"]) - 0.5 * _LOG_2PI
        resid = y - sum(eff)
        mn = proj @ resid
        bn = model._b0 + 0.5 * (resid @ resid - mn @ (vn_inv @ mn))
        out = model._lm_const - model._an * jnp.log(bn) + lp
        return jnp.where(jnp.isfinite(out), out, -jnp.inf)

    return logp


# ---------------------------------------------------------------- regional
def _geo(model, fam):
    jnp, unpack, kernel, adstock, saturate = _pieces(model, fam)
    jax = _jax()
    from jax.scipy.linalg import cho_solve
    from jax.scipy.special import gammaln

    media, G, K, p, T = model.media, len(model.geos), len(model.media), model._p, model._T_fit
    d0 = sum(len(x) for x in model._layout(fam))
    W = jnp.asarray(np.stack(model._Wg))                         # (G, K, T, L)
    X, Yn = jnp.asarray(np.stack(model._X)), jnp.asarray(model._Yn)
    XtX, Xty = jnp.asarray(np.stack(model._XtX)), jnp.asarray(np.stack(model._Xty))
    yty, s_hat, b0 = jnp.asarray(model._yty), jnp.asarray(model._s_hat), jnp.asarray(model._b0)
    prior_prec = jnp.diag(1.0 / jnp.asarray(model._sd_b2))
    logdet_b = float(np.sum(np.log(model._sd_b2)))
    a0 = model._a0
    from .geo import _REGIONAL_SD as REGIONAL_SD

    # lift tests, padded to the same number per region and the same number of rows per test
    M = max((len(t) for t in model._geo_lift), default=0)
    if M:
        R = max(len(t["rows"]) for ts in model._geo_lift for t in ts)
        Wcf, Wac = np.zeros((G, M, R, model.l_max)), np.zeros((G, M, R, model.l_max))
        hot, wt, obs = np.zeros((G, M, K)), np.zeros((G, M)), np.zeros((G, M))
        for g, ts in enumerate(model._geo_lift):
            for m, t in enumerate(ts):
                r = t["rows"]
                Wcf[g, m, :len(r)], Wac[g, m, :len(r)] = t["W_cf"][r], model._Wg[g][t["j"]][r]
                hot[g, m, t["j"]], wt[g, m], obs[g, m] = 1.0, t["w"], t["lift"]
        lift = tuple(jnp.asarray(a) for a in (Wcf, Wac, hot, wt, obs))
    else:
        lift = tuple(jnp.zeros((G, 0)) for _ in range(5))

    def region(zg, Wg, Xg, XtXg, Xtyg, yg, ytyg, sg, b0g, Lcf, Lac, Lhot, Lw, Lobs):
        P, _, _ = unpack(zg)
        kerns = [kernel(P, j, fam.carry[c]) for j, c in enumerate(media)]
        F = jnp.stack([saturate(Wg[j] @ kerns[j], P, j, fam.sat[c]) for j, c in enumerate(media)], axis=1)   # (T, K)
        mu, tau = jnp.stack(P["beta"]), jnp.stack(P["tau"])
        v = (tau * mu / sg) ** 2 + 1e-10
        FtF, Fty, n_obs, const = F.T @ F, F.T @ yg, float(T), 0.0
        for m in range(M):
            d = jnp.stack([jnp.sum(saturate(Lcf[m] @ kerns[j], P, j, fam.sat[c])
                                   - saturate(Lac[m] @ kerns[j], P, j, fam.sat[c])) for j, c in enumerate(media)])
            row = Lw[m] * d * Lhot[m]
            FtF, Fty = FtF + jnp.outer(row, row), Fty + row * (Lw[m] * Lobs[m])
            ytyg = ytyg + (Lw[m] * Lobs[m]) ** 2
            used = (Lw[m] > 0).astype(float)
            n_obs, const = n_obs + used, const + used * jnp.log(jnp.where(Lw[m] > 0, Lw[m], 1.0))
        XtF = Xg.T @ F
        A = jnp.block([[XtXg + prior_prec, XtF], [XtF.T, FtF + jnp.diag(1.0 / v)]])
        rhs = jnp.concatenate([Xtyg, Fty + mu / v])
        chol = jnp.linalg.cholesky(A)
        mn = cho_solve((chol, True), rhs)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol)))
        an = a0 + n_obs / 2
        bn = b0g + 0.5 * (ytyg + jnp.sum(mu * mu / v) - mn @ rhs)
        return (-n_obs / 2 * _LOG_2PI - 0.5 * logdet - 0.5 * (logdet_b + jnp.sum(jnp.log(v))) + a0 * jnp.log(b0g)
                - an * jnp.log(bn) + gammaln(an) - gammaln(a0) + const)

    data = (W, X, XtX, Xty, Yn, yty, s_hat, b0) + lift
    if not model.regional_curves:
        over = jax.vmap(region, in_axes=(None,) + (0,) * len(data))

        def logp(z):
            _, _, lp = unpack(z)
            out = lp + jnp.sum(over(z, *data))
            return jnp.where(jnp.isfinite(out), out, -jnp.inf)
        return logp

    ridx = jnp.asarray(model._regional_index(fam))
    nr = len(model._regional_index(fam))
    over = jax.vmap(region, in_axes=(0,) * (len(data) + 1))

    def logp(z):
        _, _, lp = unpack(z[:d0])
        log_sd = z[d0:d0 + nr]
        sd = jnp.exp(log_sd)
        eps = z[d0 + nr:].reshape(G, nr)
        lp = lp + jnp.sum(np.log(2.0) - 0.5 * _LOG_2PI - np.log(REGIONAL_SD) - sd * sd / (2 * REGIONAL_SD ** 2) + log_sd)
        lp = lp + jnp.sum(-0.5 * eps * eps - 0.5 * _LOG_2PI)
        zg = jnp.tile(z[:d0], (G, 1)).at[:, ridx].add(sd[None, :] * eps)
        out = lp + jnp.sum(over(zg, *data))
        return jnp.where(jnp.isfinite(out), out, -jnp.inf)
    return logp


def log_posterior(model, fam=None):
    """The model's log posterior as a JAX function of one unconstrained parameter vector."""
    fam = model._fam if fam is None else fam
    return _geo(model, fam) if hasattr(model, "geos") else _national(model, fam)


# ------------------------------------------------------------------- NUTS
def sample(model, fam, chains: int = 4, warmup: int = 1000, samples: int = 1000,
           target_accept: float = 0.9, max_tree_depth: int = 10, seed: int = 0) -> dict:
    """Run NUTS on the model's log posterior.

    Returns ``draws`` (chains * samples, dim) in the model's unconstrained
    parametrization, with ``rhat`` (largest split R-hat over parameters), ``ess_bulk``
    (smallest effective sample size), ``divergences`` and ``passed``.
    """
    jax = _jax()
    from numpyro.infer import MCMC, NUTS

    from ._diagnostics import summarize

    logp = log_posterior(model, fam)
    z0 = np.asarray(model._init_point(fam), dtype=float)
    rng = np.random.default_rng(seed)
    init = z0[None, :] + 0.1 * rng.standard_normal((chains, len(z0)))
    kernel = NUTS(potential_fn=lambda z: -logp(z), target_accept_prob=target_accept,
                  max_tree_depth=max_tree_depth, dense_mass=len(z0) <= DENSE_MASS_DIM)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=chains,
                chain_method=CHAIN_METHOD, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), init_params=jax.numpy.asarray(init if chains > 1 else init[0]),
             extra_fields=("diverging",))
    z = np.asarray(mcmc.get_samples(group_by_chain=True), dtype=float)          # (chains, samples, dim)
    out = summarize(z, int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()))
    out.update(draws=z.reshape(-1, z.shape[-1]), chains=chains, samples=samples)
    return out
