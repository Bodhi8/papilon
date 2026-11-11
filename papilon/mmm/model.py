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
"""Bayesian marketing mix model with gated importance sampling (Papilon RFC 001).

Inference follows the path in ``papilon-amortized-inference-design.md``:

1. a proposal distribution over the media parameters (here a Laplace fit at the
   posterior mode, standing in for the amortized network described in the RFC),
2. importance sampling against the exact posterior,
3. an effective-sample-size gate that accepts the proposal, refines it from the
   weighted draws, or escalates,
4. marginal-likelihood estimates from the same weights, so model families
   (geometric vs delayed adstock, concave vs S-shaped saturation) can be compared
   by Bayes factor.

Baseline terms (intercept, trend, seasonality, controls) and the noise variance
are integrated out in closed form with a conjugate prior, so sampling runs only
over the nonlinear media parameters. Channel effects are constrained positive.

Since 0.5 the posterior is evaluated for whole batches of draws at once, the
saturation curve can be S-shaped per channel (Hill with a fitted shape), and the
model can check itself: ``backtest()`` on held-out periods, ``self_check()`` on
simulated copies of the brand with known returns, and ``value_of_lift_test()``
for which experiment is worth running next.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import betaln, expit, gammaln, logsumexp

__all__ = ["MMM", "GateResult"]

_SHAPE_MU, _SHAPE_SD = float(np.log(1.5)), 0.4  # prior on the Hill shape of S-curve channels


@dataclass
class GateResult:
    """Outcome of the importance-sampling gate for one model family."""

    family: str
    delayed: list[str]
    log_evidence: float
    ess: float
    verdict: str
    history: list[tuple[int, float, str]] = field(default_factory=list)
    pareto_k: float = float("nan")  # PSIS tail shape; above 0.7 the importance weights are unreliable
    s_curve: list[str] = field(default_factory=list)


class MMM:
    """Marketing mix model fit by Papilon's gated importance sampling.

    Parameters
    ----------
    data : DataFrame with one row per period.
    target : column with the outcome (for example revenue).
    media : columns with spend per channel, in the same currency as ``target``
        if you want returns on ad spend.
    date_col : column with the period dates.
    controls : optional columns for non-media drivers (price, promotions).
    yearly_seasonality : number of Fourier harmonics for yearly seasonality.
    trend : include a linear trend.
    l_max : length of the carryover kernel, in periods.
    delayed : channels whose effect peaks after the period of spend.
    s_curve : channels whose saturation curve may be S-shaped: Hill with a
        fitted shape s, ``x**s / (x**s + K**s)``, prior median 1.5. Other
        channels keep shape 1 (concave from the first dollar).
    alpha_prior : Beta(a, b) prior on carryover retention, either one pair for
        every channel or a dict of pairs by channel. Default (2, 2).
    roi_priors : optional dict mapping a channel to (mean, sd) of its return on
        spend over the fitted window, typically from a lift test or geo
        experiment. Added to the posterior as a normal prior on that ROAS.
    roas_prior_scale : scale of the half-normal prior on each channel's return on
        spend at its typical spend level, either one number for every channel or a
        dict by channel (channels left out keep 2.0). Default 2.0 (prior median
        ROAS about 1.35). The prior is
        set on the ROAS scale so a small channel is not given a larger prior
        return than a big one. ``None`` restores the 0.3/0.4 prior, a half-normal
        with scale 0.5 on the effect in units of mean sales, which implied ROAS of
        roughly 4 to 10 for channels with a small share of spend.
    period : periods per year (52.18 for weekly data).
    seed : random seed for reproducible results.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target: str,
        media: list[str],
        date_col: str,
        controls: list[str] | None = None,
        yearly_seasonality: int = 2,
        trend: bool = True,
        l_max: int = 8,
        delayed: list[str] | None = None,
        period: float = 52.18,
        seed: int = 0,
        alpha_prior: tuple[float, float] | dict[str, tuple[float, float]] = (2.0, 2.0),
        roi_priors: dict[str, tuple[float, float]] | None = None,
        roas_prior_scale: float | dict[str, float] | None = 2.0,
        s_curve: list[str] | None = None,
    ) -> None:
        self._config = dict(target=target, media=list(media), date_col=date_col,
                            controls=list(controls or []), yearly_seasonality=yearly_seasonality,
                            trend=trend, l_max=l_max, period=period, alpha_prior=alpha_prior,
                            roi_priors=dict(roi_priors or {}), roas_prior_scale=roas_prior_scale)
        df = data.sort_values(date_col).reset_index(drop=True)
        self.data, self.target, self.media = df, target, list(media)
        self.date_col, self.controls = date_col, list(controls or [])
        self.l_max, self.period = int(l_max), float(period)
        self.delayed = list(delayed or [])
        self.s_curve = list(s_curve or [])
        self._check_channels(self.delayed, "delayed")
        self._check_channels(self.s_curve, "s_curve")
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        if isinstance(alpha_prior, dict):
            self.alpha_prior = {c: tuple(alpha_prior.get(c, (2.0, 2.0))) for c in self.media}
        else:
            self.alpha_prior = {c: tuple(alpha_prior) for c in self.media}
        self.roi_priors = dict(roi_priors or {})
        unknown = set(self.roi_priors) - set(self.media)
        if unknown:
            raise ValueError(f"roi_priors channels not in media: {sorted(unknown)}")

        self.y_raw = df[target].to_numpy(dtype=float)
        self.spend_raw = df[self.media].to_numpy(dtype=float)
        if np.any(self.spend_raw < 0):
            raise ValueError("media spend must be non-negative")
        self.y_scale = float(np.mean(self.y_raw))
        self.spend_scale = self.spend_raw.mean(axis=0)
        self.spend_scale[self.spend_scale == 0] = 1.0
        self.y = self.y_raw / self.y_scale
        self.spend = self.spend_raw / self.spend_scale
        self._spend_tot = self.spend_raw.sum(axis=0)
        self._W = self._windows(self.spend)
        # Prior scale for each channel's effect b (in units of mean sales). Contribution at
        # typical spend is about b * y_scale * 0.5 (half-saturation prior median 1, kernel
        # normalized to sum to 1), so ROAS ~ b * y_scale * 0.5 / spend_scale.
        rs = ({c: roas_prior_scale.get(c, 2.0) for c in self.media} if isinstance(roas_prior_scale, dict)
              else {c: roas_prior_scale for c in self.media})
        self._b_scale = {c: (0.5 if rs.get(c) is None else
                             float(rs[c]) * self.spend_scale[j] / (0.5 * self.y_scale))
                         for j, c in enumerate(self.media)}
        self._roi_idx = [(self.media.index(c), float(m), float(sd)) for c, (m, sd) in
                         (roi_priors or {}).items() if c in self.media]

        self._trend, self._harmonics = bool(trend), int(yearly_seasonality)
        self._ctrl_stats = []
        for c in self.controls:
            v = df[c].to_numpy(dtype=float)
            self._ctrl_stats.append((float(v.mean()), float(v.std() or 1.0)))
        n = len(df)
        self._base_x = self._design(df, 0)
        sds = [100.0] + ([10.0] if trend else []) + [10.0] * (2 * yearly_seasonality) + [20.0] * len(self.controls)
        self._prior_sd = np.array(sds)  # baseline coefficients, in units of sigma
        v0_inv = np.diag(1 / self._prior_sd ** 2)
        self._vn_inv = v0_inv + self._base_x.T @ self._base_x
        self._vn = np.linalg.inv(self._vn_inv)
        self._proj = self._vn @ self._base_x.T
        self._logdet_ratio = np.linalg.slogdet(self._vn)[1] - np.sum(np.log(self._prior_sd ** 2))
        self._a0 = 2.0
        self._b0 = 0.1 * float(np.var(self.y))  # weakly informative noise prior
        self._an = self._a0 + n / 2
        self._lags = np.arange(self.l_max)
        self._lm_const = (-n / 2 * np.log(2 * np.pi) + 0.5 * self._logdet_ratio + self._a0 * np.log(self._b0)
                          + gammaln(self._an) - gammaln(self._a0))

        self.gate_results: dict[str, GateResult] = {}
        self.family_probabilities: dict[str, float] = {}
        self._draws: dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------------ model
    def _check_channels(self, names: list[str], what: str = "delayed") -> None:
        unknown = set(names) - set(self.media)
        if unknown:
            raise ValueError(f"{what} channels not in media: {sorted(unknown)}")

    _check_delayed = _check_channels

    def _design(self, df: pd.DataFrame, offset: int) -> np.ndarray:
        """Baseline design matrix for rows of ``df`` that start ``offset`` periods into the fit window."""
        n = len(df)
        t = (offset + np.arange(n)) / self.period
        cols = [np.ones(n)]
        if self._trend:
            cols.append(t)
        for k in range(1, self._harmonics + 1):
            cols += [np.sin(2 * np.pi * k * t), np.cos(2 * np.pi * k * t)]
        for c, (mu, sd) in zip(self.controls, self._ctrl_stats):
            cols.append((df[c].to_numpy(dtype=float) - mu) / sd)
        return np.column_stack(cols)

    def _windows(self, spend: np.ndarray) -> np.ndarray:
        """Lagged spend, shape (channels, periods, l_max): W[j, t, l] = spend[t - l, j]."""
        T, k = spend.shape
        W = np.zeros((k, T, self.l_max))
        for l in range(min(self.l_max, T)):
            W[:, l:, l] = spend[: T - l].T
        return W

    def _dim(self, delayed: list[str], shaped: list[str] | None = None) -> int:
        return 3 * len(self.media) + len(delayed) + len(shaped or [])

    def _unpack_batch(self, Z: np.ndarray, delayed: list[str], shaped: list[str]):
        """Unconstrained draws (n, d) -> parameter arrays (n, channels), plus log prior with Jacobian."""
        Z = np.atleast_2d(Z)
        n, k = len(Z), len(self.media)
        P = {"alpha": np.empty((n, k)), "half_sat": np.empty((n, k)), "beta": np.empty((n, k)),
             "peak": np.full((n, k), np.nan), "shape": np.ones((n, k))}
        lp = np.zeros(n)
        i, max_peak = 0, self.l_max - 2
        with np.errstate(all="ignore"):
            for j, c in enumerate(self.media):
                za, zh, zb = Z[:, i], Z[:, i + 1], Z[:, i + 2]
                i += 3
                a, hs, b = expit(za), np.exp(zh), np.exp(zb)
                pa, pb = self.alpha_prior[c]
                lp += (_beta_lpdf(a, pa, pb) + np.log(a) + np.log1p(-a)
                       + _lognorm_lpdf(hs, 0.5) + zh
                       + _halfnorm_lpdf(b, self._b_scale[c]) + zb)
                P["alpha"][:, j], P["half_sat"][:, j], P["beta"][:, j] = a, hs, b
                if c in delayed:
                    u = expit(Z[:, i]); i += 1
                    P["peak"][:, j] = max_peak * u
                    lp += _beta_lpdf(u, 2.0, 2.0) + np.log(u) + np.log1p(-u)
                if c in shaped:
                    zs = Z[:, i]; i += 1
                    P["shape"][:, j] = np.exp(zs)
                    lp += -np.log(_SHAPE_SD) - _HALF_LOG_2PI - (zs - _SHAPE_MU) ** 2 / (2 * _SHAPE_SD ** 2)
        return P, lp

    def _effects(self, P: dict, W: np.ndarray | None = None, scale: float = 1.0) -> np.ndarray:
        """Channel contributions in model units, shape (periods, draws, channels).

        ``scale`` multiplies spend in every period (adstock is linear in spend).
        """
        W = self._W if W is None else W
        lags = np.arange(self.l_max)[None, :]
        out = np.empty((W.shape[1], len(P["beta"]), len(self.media)))
        with np.errstate(all="ignore"):
            for j in range(len(self.media)):
                a, pk = P["alpha"][:, j][:, None], P["peak"][:, j][:, None]
                kern = np.where(np.isnan(pk), a ** lags, a ** ((lags - np.nan_to_num(pk)) ** 2))
                kern = kern / kern.sum(axis=1, keepdims=True)
                ad = (W[j] @ kern.T) * scale
                hs, s = P["half_sat"][:, j], P["shape"][:, j]
                if np.all(s == 1.0):
                    out[:, :, j] = P["beta"][:, j] * ad / (ad + hs)
                else:
                    ads = ad ** s
                    out[:, :, j] = P["beta"][:, j] * ads / (ads + hs ** s)
        return out

    def _marginal_parts(self, resid: np.ndarray):
        """Posterior mean of the baseline coefficients and the noise scale term, per draw."""
        mn = self._proj @ resid                                        # (p, n)
        bn = self._b0 + 0.5 * (np.sum(resid * resid, axis=0) - np.sum(mn * (self._vn_inv @ mn), axis=0))
        return mn, bn

    def _log_post_batch(self, Z: np.ndarray, delayed: list[str], shaped: list[str] | None = None) -> np.ndarray:
        shaped = shaped or []
        P, lp = self._unpack_batch(Z, delayed, shaped)
        eff = self._effects(P)
        with np.errstate(all="ignore"):
            for j, m, sd in self._roi_idx:
                roas = eff[:, :, j].sum(axis=0) * self.y_scale / self._spend_tot[j]
                lp = lp + (-0.5 * ((roas - m) / sd) ** 2 - np.log(sd) - _HALF_LOG_2PI)
            _, bn = self._marginal_parts(self.y[:, None] - eff.sum(axis=2))
            n = len(self.y)
            out = (-n / 2 * np.log(2 * np.pi) + 0.5 * self._logdet_ratio + self._a0 * np.log(self._b0)
                   - self._an * np.log(bn) + gammaln(self._an) - gammaln(self._a0)) + lp
        return np.where(np.isfinite(out), out, -np.inf)

    def _log_post(self, z: np.ndarray, delayed: list[str], shaped: list[str] | None = None) -> float:
        """One draw, without the batch overhead: this is what the mode search calls."""
        shaped = shaped or ()
        i, lp, max_peak, lags = 0, 0.0, self.l_max - 2, self._lags
        media = np.zeros(len(self.y))
        roas = {}
        with np.errstate(all="ignore"):
            for j, c in enumerate(self.media):
                za, zh, zb = z[i], z[i + 1], z[i + 2]
                i += 3
                a, hs, b = expit(za), np.exp(zh), np.exp(zb)
                pa, pb = self.alpha_prior[c]
                lp += (_beta_lpdf(a, pa, pb) + np.log(a) + np.log1p(-a)
                       + _lognorm_lpdf(hs, 0.5) + zh
                       + _halfnorm_lpdf(b, self._b_scale[c]) + zb)
                if c in delayed:
                    u = expit(z[i]); i += 1
                    lp += _beta_lpdf(u, 2.0, 2.0) + np.log(u) + np.log1p(-u)
                    kern = a ** ((lags - max_peak * u) ** 2)
                else:
                    kern = a ** lags
                ad = self._W[j] @ (kern / kern.sum())
                if c in shaped:
                    zs = z[i]; i += 1
                    lp += -np.log(_SHAPE_SD) - _HALF_LOG_2PI - (zs - _SHAPE_MU) ** 2 / (2 * _SHAPE_SD ** 2)
                    s = np.exp(zs)
                    ad, hs = ad ** s, hs ** s
                eff = b * ad / (ad + hs)
                roas[j] = eff
                media += eff
            if not np.isfinite(lp):
                return -np.inf
            for j, m, sd in self._roi_idx:
                r = roas[j].sum() * self.y_scale / self._spend_tot[j]
                lp += -0.5 * ((r - m) / sd) ** 2 - np.log(sd) - _HALF_LOG_2PI
            resid = self.y - media
            mn = self._proj @ resid
            bn = self._b0 + 0.5 * (resid @ resid - mn @ self._vn_inv @ mn)
            out = self._lm_const - self._an * np.log(bn) + lp
        return float(out) if np.isfinite(out) else -np.inf

    # -------------------------------------------------------------- inference
    def _laplace(self, delayed: list[str], shaped: list[str], n_starts: int):
        f = lambda z: -self._log_post(z, delayed, shaped)
        init = []
        for c in self.media:
            init += [0.0, 0.0, np.log(0.4 * self._b_scale[c])]
            init += [0.0] * (c in delayed) + [_SHAPE_MU] * (c in shaped)
        z_init = np.array(init)
        best = None
        for _ in range(n_starts):
            z0 = z_init + self.rng.normal(0, 0.3, len(z_init))
            r = optimize.minimize(f, z0, method="Nelder-Mead",
                                  options={"maxiter": 20000, "maxfev": 20000, "xatol": 1e-4, "fatol": 1e-6})
            r = optimize.minimize(f, r.x, method="BFGS")
            if best is None or r.fun < best.fun:
                best = r
        d, h = len(best.x), 1e-3
        pts, where = [], []
        for a in range(d):
            for b in range(a, d):
                ea = np.zeros(d); ea[a] = h
                eb = np.zeros(d); eb[b] = h
                pts += [best.x + ea + eb, best.x + ea - eb, best.x - ea + eb, best.x - ea - eb]
                where.append((a, b))
        v = -self._log_post_batch(np.array(pts), delayed, shaped).reshape(-1, 4)
        hess = np.zeros((d, d))
        for (a, b), (pp_, pm, mp, mm) in zip(where, v):
            hess[a, b] = hess[b, a] = (pp_ - pm - mp + mm) / (4 * h * h)
        hess = np.nan_to_num(hess, nan=0.0, posinf=0.0, neginf=0.0)
        vals, vecs = np.linalg.eigh(0.5 * (hess + hess.T))
        return best.x, vecs @ np.diag(1 / np.maximum(vals, 0.5)) @ vecs.T

    def _importance(self, delayed, shaped, mean, cov, n_draws):
        q = stats.multivariate_t(loc=mean, shape=cov, df=5, seed=self.rng)
        z = q.rvs(n_draws)
        log_w = self._log_post_batch(z, delayed, shaped) - q.logpdf(z)
        log_ev = float(logsumexp(log_w) - np.log(n_draws))
        w = np.exp(log_w - log_w.max()); w /= w.sum()
        self._last_k = _pareto_k(log_w)
        return z, w, log_ev, float(1 / np.sum(w ** 2) / n_draws)

    def _gate(self, name, delayed, shaped, n_draws, n_starts, accept, refine, max_refine):
        mean, cov = self._laplace(delayed, shaped, n_starts)
        history = []
        for step in range(max_refine + 1):
            z, w, log_ev, ess = self._importance(delayed, shaped, mean, 1.5 * cov, n_draws)
            verdict = "ACCEPT" if ess > accept else "REFINE" if ess > refine else "ESCALATE"
            history.append((step, round(ess, 3), verdict))
            if verdict == "ACCEPT":
                break
            mean = np.average(z, axis=0, weights=w)  # refine: moment-match the weighted draws
            cov = np.cov(z.T, aweights=w) + 1e-6 * np.eye(len(mean))
        return GateResult(name, list(delayed), log_ev, ess, verdict, history,
                          pareto_k=float(getattr(self, "_last_k", np.nan)), s_curve=list(shaped)), z, w

    def _store_draws(self, delayed, shaped, z, w, n_keep=2000):
        idx = self.rng.choice(len(w), n_keep, p=w)
        self._draws, _ = self._unpack_batch(z[idx], delayed, shaped)
        self.delayed, self.s_curve = list(delayed), list(shaped)
        self._roi_cache = self._roas_cache = None

    def _family_name(self) -> str:
        name = "delayed:" + ",".join(self.delayed) if self.delayed else "all_geometric"
        return name + ("+s_curve:" + ",".join(self.s_curve) if self.s_curve else "")

    def fit(self, n_draws: int = 4000, n_starts: int = 3, accept: float = 0.10,
            refine: float = 0.01, max_refine: int = 3) -> "MMM":
        """Fit the model with the family set by ``delayed`` and ``s_curve``."""
        name = self._family_name()
        res, z, w = self._gate(name, self.delayed, self.s_curve, n_draws, n_starts, accept, refine, max_refine)
        self.gate_results = {name: res}
        self.family_probabilities = {name: 1.0}
        self._store_draws(self.delayed, self.s_curve, z, w)
        return self

    def compare_models(self, candidates: dict[str, dict], n_draws: int = 4000,
                       n_starts: int = 3, accept: float = 0.10, refine: float = 0.01,
                       max_refine: int = 3) -> pd.DataFrame:
        """Fit each model family, compare them by marginal likelihood, keep the winner.

        ``candidates`` maps a family name to ``{"delayed": [...], "s_curve": [...]}``
        (either key may be left out), for example::

            {"concave": {"delayed": ["online_video"]},
             "video_s_curve": {"delayed": ["online_video"], "s_curve": ["online_video"]}}

        Families get equal prior odds. A wrong curve can fit history and still
        extrapolate badly, which is what a reallocation asks of it; this lets the
        data vote on the shape before a plan is built on it.
        """
        fitted = {}
        for name, spec in candidates.items():
            delayed, shaped = list(spec.get("delayed", [])), list(spec.get("s_curve", []))
            self._check_channels(delayed, "delayed"); self._check_channels(shaped, "s_curve")
            fitted[name] = self._gate(name, delayed, shaped, n_draws, n_starts, accept, refine, max_refine)
        names = list(fitted)
        ev = np.array([fitted[k][0].log_evidence for k in names])
        probs = np.exp(ev - logsumexp(ev))
        self.gate_results = {k: fitted[k][0] for k in names}
        self.family_probabilities = dict(zip(names, map(float, probs)))
        win = names[int(np.argmax(probs))]
        res, z, w = fitted[win]
        self._store_draws(res.delayed, res.s_curve, z, w)
        return pd.DataFrame({
            "log_evidence": ev,
            "log_bayes_factor_vs_best": ev - ev.max(),
            "probability": probs,
            "ess_fraction": [fitted[k][0].ess for k in names],
            "verdict": [fitted[k][0].verdict for k in names],
        }, index=pd.Index(names, name="family"))

    def compare_carryover(self, candidates: dict[str, list[str]], n_draws: int = 4000,
                          n_starts: int = 3, accept: float = 0.10, refine: float = 0.01,
                          max_refine: int = 3) -> pd.DataFrame:
        """Fit each carryover family, compare them by marginal likelihood, keep the winner.

        ``candidates`` maps a family name to the channels that use delayed adstock,
        for example ``{"all_geometric": [], "video_delayed": ["online_video"]}``.
        Families get equal prior odds. Channels in ``s_curve`` keep their S-curve
        in every family; use :meth:`compare_models` to compare saturation shapes.
        """
        return self.compare_models({k: {"delayed": v, "s_curve": self.s_curve} for k, v in candidates.items()},
                                   n_draws, n_starts, accept, refine, max_refine)

    # ---------------------------------------------------------------- results
    def _require_fit(self) -> None:
        if self._draws is None:
            raise RuntimeError("call fit() or compare_carryover() first")

    def parameters(self) -> pd.DataFrame:
        """Posterior means of the media parameters (half-saturation in raw spend units)."""
        self._require_fit()
        d = self._draws
        out = pd.DataFrame({
            "alpha": d["alpha"].mean(0),
            "half_saturation": d["half_sat"].mean(0) * self.spend_scale,
            "max_weekly_effect": d["beta"].mean(0) * self.y_scale,
            "peak_lag": [float(np.mean(d["peak"][:, j])) if not np.isnan(d["peak"][0, j]) else np.nan
                         for j in range(len(self.media))],
            "shape": d["shape"].mean(0),
        }, index=pd.Index(self.media, name="channel"))
        return out

    def roas_draws(self) -> dict:
        """Posterior draws, shape (draws, channels), of ROAS, marginal ROAS and share of sales."""
        self._require_fit()
        if getattr(self, "_roas_cache", None) is None:
            m = self._effects(self._draws).sum(axis=0) * self.y_scale
            bumped = self._effects(self._draws, scale=1.01).sum(axis=0) * self.y_scale
            self._roas_cache = {"roas": m / self._spend_tot, "mroas": (bumped - m) / (0.01 * self._spend_tot),
                                "share": m / self.y_raw.sum()}
        return self._roas_cache

    def get_roi(self) -> pd.DataFrame:
        """Return on spend per channel: posterior mean, 90% interval and marginal return.

        ROAS is the channel's modeled contribution over the fitted window divided
        by its spend. Marginal ROAS is the extra contribution from 1% more spend in
        every period, per extra unit of spend.
        """
        self._require_fit()
        if getattr(self, "_roi_cache", None) is not None:
            return self._roi_cache
        r = self.roas_draws()
        self._roi_cache = pd.DataFrame({
            "spend": self._spend_tot.copy(),
            "roas_mean": r["roas"].mean(0),
            "roas_p05": np.quantile(r["roas"], 0.05, axis=0),
            "roas_p95": np.quantile(r["roas"], 0.95, axis=0),
            "mroas_mean": r["mroas"].mean(0),
            "share_of_sales": r["share"].mean(0),
        }, index=pd.Index(self.media, name="channel"))
        return self._roi_cache

    def contributions(self) -> pd.DataFrame:
        """Posterior-mean contribution of each channel per period, in target units."""
        self._require_fit()
        out = pd.DataFrame(self._effects(self._draws).mean(axis=1) * self.y_scale, columns=self.media)
        out.insert(0, self.date_col, self.data[self.date_col].to_numpy())
        out["baseline_and_controls"] = self.y_raw - out[self.media].sum(axis=1)
        return out

    def _curve(self, xs: np.ndarray) -> np.ndarray:
        """Steady-state contribution, model units, per draw and channel, at scaled spend xs (..., channels)."""
        b, hs, s = self._draws["beta"], self._draws["half_sat"], self._draws["shape"]
        if np.all(s == 1.0):
            return b * xs / (xs + hs)
        xp = np.asarray(xs, dtype=float) ** s
        return b * xp / (xp + hs ** s)

    def response_curve(self, channel: str, spend_levels: np.ndarray) -> pd.DataFrame:
        """Steady-state contribution per period at constant spend levels, with a 90% interval."""
        self._require_fit()
        j = self.media.index(channel)
        x = (np.asarray(spend_levels, dtype=float) / self.spend_scale[j])[None, :]
        b = self._draws["beta"][:, j][:, None]; hs = self._draws["half_sat"][:, j][:, None]
        s = self._draws["shape"][:, j][:, None]
        r = b * x ** s / (x ** s + hs ** s) * self.y_scale
        return pd.DataFrame({"spend": spend_levels, "contribution_mean": r.mean(0),
                             "contribution_p05": np.quantile(r, 0.05, axis=0),
                             "contribution_p95": np.quantile(r, 0.95, axis=0)})

    def posterior_curves(self) -> dict:
        """Posterior draws of each channel's response parameters, in raw units.

        Returns a dict with ``media`` (channel names) and arrays of shape
        (draws, channels): ``beta`` (maximum contribution per period, target
        units), ``half_sat`` (spend per period at half of beta), ``alpha``
        (carryover retention), ``peak`` (delay of the peak, NaN for
        geometric channels) and ``shape`` (Hill shape, 1 unless the channel is in
        ``s_curve``). Steady-state contribution at constant spend x is
        ``beta * x**shape / (x**shape + half_sat**shape)``. This is the hand-off
        format used by simulators such as ``pycausalsim.futures.marketing``.
        """
        self._require_fit()
        d = self._draws
        return {"media": list(self.media),
                "beta": d["beta"] * self.y_scale,
                "half_sat": d["half_sat"] * self.spend_scale,
                "alpha": d["alpha"].copy(), "peak": d["peak"].copy(), "shape": d["shape"].copy(),
                "l_max": self.l_max,
                "historical_spend": self.spend_raw.mean(axis=0)}

    def _steady(self, x):
        """Steady-state media contribution per period for each posterior draw."""
        xs = np.asarray(x, dtype=float) / self.spend_scale
        return np.sum(self._curve(xs), axis=-1) * self.y_scale

    def evaluate_plan(self, spend, against=None) -> dict:
        """Posterior distribution of the change in per-period contribution from a plan.

        ``spend`` and ``against`` are per-period spend by channel (dict or array in
        ``media`` order). ``against`` defaults to the historical mix at the same total.
        """
        self._require_fit()
        x = self._as_vector(spend)
        hist = self.spend_raw.mean(axis=0)
        base = hist / hist.sum() * x.sum() if against is None else self._as_vector(against)
        lift = self._steady(x) - self._steady(base)
        return {"expected_lift": float(lift.mean()),
                "lift_p05": float(np.quantile(lift, 0.05)),
                "lift_p95": float(np.quantile(lift, 0.95)),
                "prob_better": float(np.mean(lift > 0)),
                "draws": lift}

    def _as_vector(self, spend):
        if isinstance(spend, dict):
            return np.array([float(spend[c]) for c in self.media])
        if isinstance(spend, pd.Series):
            return spend.reindex(self.media).to_numpy(dtype=float)
        return np.asarray(spend, dtype=float)

    def _plan_limits(self, total, bounds, max_change, support, support_quantile):
        hist = self.spend_raw.mean(axis=0)
        current = hist / hist.sum() * total
        # a plan is a sustained level, so compare it with sustained history:
        # rolling means of spend over the carryover window
        k = min(self.l_max, len(self.spend_raw))
        roll = np.array([np.convolve(self.spend_raw[:, j], np.ones(k) / k, mode="valid")
                         for j in range(len(self.media))]).T
        observed_hi = np.quantile(roll, support_quantile, axis=0)
        lims = []
        for j, c in enumerate(self.media):
            lo, hi = (bounds or {}).get(c, (0.0, 3.0 * hist[j]))
            if max_change is not None:
                lo, hi = max(lo, current[j] * (1 - max_change)), min(hi, current[j] * (1 + max_change))
            if support == "limit":
                hi = min(hi, max(observed_hi[j], current[j]))
            lims.append((lo, hi))
        if sum(lo for lo, _ in lims) > total + 1e-9 or sum(hi for _, hi in lims) < total - 1e-9:
            raise ValueError("bounds cannot be met with this total budget")
        return current, lims, observed_hi

    def _solve_plan(self, total, current, lims, objective, quantile, weights=None):
        """Best spend vector under the limits; ``weights`` reweights the posterior draws."""
        base = self._steady(current)
        if weights is None:
            mean = lambda v: float(v.mean())
            quant = lambda v: float(np.quantile(v, quantile))
        else:
            mean = lambda v: float(v @ weights)
            quant = lambda v: _weighted_quantile(v, weights, quantile)
        f_mean = lambda x: -mean(self._steady(x))
        f = f_mean if objective == "mean" else (lambda x: -quant(self._steady(x) - base))
        cons = [{"type": "eq", "fun": lambda x: x.sum() - total}]
        x0 = np.clip(current, [lo for lo, _ in lims], [hi for _, hi in lims])
        starts = [x0]
        concave = bool(np.all(self._draws["shape"] == 1.0))
        if not concave:  # S-curves make the surface multi-modal: also start from each channel pushed up
            for j in range(len(x0)):
                s = x0.copy(); s[j] = lims[j][1]
                rest = [i for i in range(len(x0)) if i != j]
                left = total - s[j]
                if left >= sum(lims[i][0] for i in rest):
                    share = x0[rest] / max(x0[rest].sum(), 1e-12) * left
                    s[rest] = np.clip(share, [lims[i][0] for i in rest], [lims[i][1] for i in rest])
                    starts.append(s)
        if objective == "quantile":  # the quantile surface is rough; also start from the mean optimum
            r = optimize.minimize(f_mean, x0, method="SLSQP",
                                  bounds=lims, constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
            starts.append(r.x)
            # and from the best point on the straight path from the current mix to
            # the mean optimum (shrinkage), where the quantile objective is smooth
            path = [x0 + t * (r.x - x0) for t in np.linspace(0, 1, 21)]
            starts.append(min(path, key=f))
        best = None
        for s0 in starts:
            r = optimize.minimize(f, s0, method="SLSQP", bounds=lims, constraints=cons,
                                  options={"maxiter": 500, "ftol": 1e-10})
            ok = abs(r.x.sum() - total) < 1e-6 * max(total, 1.0)
            if ok and (best is None or r.fun < best.fun):
                best = r
        return x0 if best is None else best.x

    def optimize_budget(self, total: float, bounds: dict[str, tuple[float, float]] | None = None,
                        max_change: float | None = 0.5, objective: str = "quantile",
                        quantile: float = 0.10, support: str = "flag",
                        support_quantile: float = 0.95) -> pd.DataFrame:
        """Allocate a per-period budget across channels.

        Uses steady-state response curves (constant spend each period) from the
        posterior. ``bounds`` maps channels to (min, max) spend per period; the
        default allows 0 to 3 times each channel's historical average.

        Defaults changed in 0.4.0. Across 120 simulated brands with known
        returns (``benchmarks/recovery.py``), plans that maximize expected
        contribution leaned hardest on the channels the model overrated: they
        lost money for a third of standard brands and for most brands with
        unmodeled launches. Maximizing the 10th percentile of the gain with each
        channel held within 50% of its current spend balanced the two best: it
        lost money far less often than uncapped rules and captured more of the
        available gain than a 25% cap in most settings. For the 0.3 behavior pass ``objective="mean",
        max_change=None``.

        max_change : limit each channel to within this fraction of its current
            spend (default 0.5, plus or minus 50%); None removes the limit.
        objective : "quantile" (default) maximizes the ``quantile`` (default
            10th percentile) of the gain over the current mix, which favors plans
            that hold up if the model is wrong; "mean" maximizes expected
            contribution.
        support : "flag" marks channels whose planned spend exceeds the
            ``support_quantile`` of their historical sustained spend (rolling
            means over the carryover window, since a plan is a sustained level);
            "limit" also caps them there; "ignore" does neither.

        The result carries ``attrs["summary"]`` with the expected gain over the
        current mix, its 90% interval and the probability the plan beats it.
        """
        self._require_fit()
        if objective not in ("mean", "quantile"):
            raise ValueError('objective must be "mean" or "quantile"')
        if support not in ("flag", "limit", "ignore"):
            raise ValueError('support must be "flag", "limit" or "ignore"')
        current, lims, observed_hi = self._plan_limits(total, bounds, max_change, support, support_quantile)
        x_opt = self._solve_plan(total, current, lims, objective, quantile)

        per_channel = lambda x: self._curve(np.asarray(x) / self.spend_scale).mean(axis=0) * self.y_scale
        out = pd.DataFrame({
            "current_spend": current,
            "optimal_spend": x_opt,
            "current_contribution": per_channel(current),
            "optimal_contribution": per_channel(x_opt),
        }, index=pd.Index(self.media, name="channel")).assign(
            change_in_spend=lambda d: d.optimal_spend - d.current_spend,
            outside_support=x_opt > observed_hi + 1e-9)
        ev = self.evaluate_plan(x_opt, current)
        out.attrs["summary"] = {k: v for k, v in ev.items() if k != "draws"} | {
            "objective": objective, "max_change": max_change, "support": support}
        return out

    # ------------------------------------------------------- checking the model
    def _clone(self, data: pd.DataFrame, seed: int | None = None, **overrides) -> "MMM":
        """A new, unfitted model with this model's settings and family on other data."""
        kw = dict(self._config) | {"delayed": list(self.delayed), "s_curve": list(self.s_curve),
                                   "seed": self.seed if seed is None else seed} | overrides
        return MMM(data, **kw)

    def backtest(self, holdout: int = 13, n_starts: int = 2) -> pd.DataFrame:
        """Refit without the last ``holdout`` periods and forecast them.

        The forecast uses the spend and controls that actually ran, so it tests
        the model's response curves and baseline, not a spend forecast. Returns
        one row per held-out period (actual, predicted mean, 90% predictive
        interval). ``attrs["summary"]`` has the mean absolute percentage error,
        the share of periods inside the interval, and the error of a naive
        forecast (the same period a year earlier, or the training mean when there
        is less than a year of history) to compare against.

        A model can forecast well and still credit the wrong channels, so this is
        a necessary check, not a sufficient one; ``self_check()`` covers the rest.
        """
        self._require_fit()
        h, n = int(holdout), len(self.data)
        if not 1 <= h <= n - max(26, 2 * self._base_x.shape[1]):
            raise ValueError("holdout leaves too few periods to fit")
        train, test = self.data.iloc[: n - h], self.data.iloc[n - h:]
        m = self._clone(train).fit(n_starts=n_starts)
        d = m._draws
        full_spend = self.spend_raw / m.spend_scale
        eff = m._effects(d, W=m._windows(full_spend))                   # (n, draws, k)
        media = eff.sum(axis=2)
        mn, bn = m._marginal_parts(m.y[:, None] - media[: n - h])
        xh = m._design(test, n - h)
        mean = xh @ mn + media[n - h:]                                   # (h, draws)
        lev = 1 + np.einsum("ij,jk,ik->i", xh, m._vn, xh)
        scale = np.sqrt(bn / m._an * lev[:, None])
        rng = np.random.default_rng(self.seed + 1)
        pred = (mean + scale * rng.standard_t(2 * m._an, size=mean.shape)) * m.y_scale
        actual = test[self.target].to_numpy(dtype=float)
        out = pd.DataFrame({self.date_col: test[self.date_col].to_numpy(), "actual": actual,
                            "predicted": pred.mean(axis=1),
                            "p05": np.quantile(pred, 0.05, axis=1), "p95": np.quantile(pred, 0.95, axis=1)})
        lag = int(round(self.period))
        naive = self.y_raw[n - h - lag: n - lag] if n - h - lag >= 0 else np.full(h, train[self.target].mean())
        mape = lambda p: float(np.mean(np.abs(p - actual) / np.abs(actual)))
        out.attrs["summary"] = {"holdout": h, "mape": mape(out.predicted.to_numpy()),
                                "naive_mape": mape(naive),
                                "coverage_90": float(np.mean((actual >= out.p05) & (actual <= out.p95)))}
        return out

    def self_check(self, n_worlds: int = 12, confounded: dict[str, float] | None = None,
                   n_starts: int = 2, seed: int | None = None) -> pd.DataFrame:
        """Grade this exact setup on simulated copies of the brand with known returns.

        Each world keeps your spend, controls and calendar, draws true channel
        parameters from the fitted posterior, and generates sales from them with
        fresh noise. The same model settings are then refit to each world and
        graded against its answer key: does the 90% ROAS range hold the truth
        about nine times in ten, and does the default budget plan gain or lose
        money against the true curves?

        This answers "could this model, with this data, recover returns like the
        ones it reports?" It cannot tell you the model's form is right for the
        real world. To see what an unmeasured driver would do, pass
        ``confounded={"paid_search": 0.5}``: sales in every world also get demand
        that moves with that channel's spend (0.5 standard deviations of sales per
        standard deviation of spend) and is not caused by it.

        Returns one row per world and channel; ``attrs["summary"]`` has coverage,
        mean absolute error and bias by channel, the share of worlds where the
        plan lost money and the share of the best possible gain it captured.
        """
        self._require_fit()
        rng = np.random.default_rng(self.seed + 7 if seed is None else seed)
        d, k = self._draws, len(self.media)
        pick = rng.choice(len(d["beta"]), n_worlds, replace=False)
        truth = {key: v[pick] for key, v in d.items()}
        eff = self._effects(truth)                                      # (T, worlds, k)
        mn, bn = self._marginal_parts(self.y[:, None] - eff.sum(axis=2))
        sigma = np.sqrt(bn / (self._an - 1))
        y_sim = (self._base_x @ mn + eff.sum(axis=2) + rng.normal(size=(len(self.y), n_worlds)) * sigma) * self.y_scale
        for c, strength in (confounded or {}).items():
            x = self.spend_raw[:, self.media.index(c)]
            y_sim += strength * self.y_raw.std() * ((x - x.mean()) / (x.std() or 1.0))[:, None]
        true_roas = eff.sum(axis=0) * self.y_scale / self._spend_tot     # (worlds, k)
        hist = self.spend_raw.mean(axis=0)
        total = float(hist.sum())
        rows, plans = [], []
        for w in range(n_worlds):
            df = self.data.copy()
            df[self.target] = y_sim[:, w]
            m = self._clone(df, seed=self.seed + 100 + w).fit(n_starts=n_starts)
            roi = m.get_roi()
            tw = {key: v[w: w + 1] for key, v in truth.items()}

            def true_steady(x, tw=tw):
                xs = (np.asarray(x, dtype=float) / self.spend_scale) ** tw["shape"][0]
                return float(np.sum(tw["beta"][0] * xs / (xs + tw["half_sat"][0] ** tw["shape"][0])) * self.y_scale)
            x = m.optimize_budget(total)["optimal_spend"].to_numpy()
            lims = [(0.0, 3.0 * hv) for hv in hist]
            best = max((optimize.minimize(lambda v: -true_steady(v), s0, method="SLSQP", bounds=lims,
                                          constraints=[{"type": "eq", "fun": lambda v: v.sum() - total}],
                                          options={"maxiter": 300, "ftol": 1e-10})
                        for s0 in (hist, np.full(k, total / k))), key=lambda r: -r.fun)
            gain, best_gain = true_steady(x) - true_steady(hist), max(-best.fun - true_steady(hist), 0.0)
            plans.append((gain, best_gain))
            for j, c in enumerate(self.media):
                rows.append({"world": w, "channel": c, "true_roas": true_roas[w, j],
                             "roas_mean": roi.loc[c, "roas_mean"], "roas_p05": roi.loc[c, "roas_p05"],
                             "roas_p95": roi.loc[c, "roas_p95"], "plan_gain": gain, "best_gain": best_gain})
        out = pd.DataFrame(rows)
        out["covered"] = (out.true_roas >= out.roas_p05) & (out.true_roas <= out.roas_p95)
        out["error"] = out.roas_mean - out.true_roas
        g = out.groupby("channel", sort=False)
        gains = np.array(plans)
        out.attrs["summary"] = {
            "worlds": n_worlds, "confounded": dict(confounded or {}),
            "coverage_90": float(out.covered.mean()),
            "coverage_by_channel": g.covered.mean().to_dict(),
            "abs_error_by_channel": g.error.apply(lambda e: float(np.abs(e).mean())).to_dict(),
            "bias_by_channel": g.error.mean().to_dict(),
            "plan_lost_money": float(np.mean(gains[:, 0] < 0)),
            "plan_capture": float(gains[:, 0].sum() / max(gains[:, 1].sum(), 1e-12)),
        }
        return out

    def value_of_lift_test(self, sd: float | dict[str, float] = 0.25, total: float | None = None,
                           n_sim: int = 100, seed: int | None = None, **plan_kwargs) -> pd.DataFrame:
        """Rank channels by what a lift test on each would be worth to the budget plan.

        For each channel, simulate ``n_sim`` test results the posterior says are
        plausible (the channel's ROAS measured with standard error ``sd``), update
        the posterior on each by reweighting its draws, re-plan, and score the new
        plan against the world that produced the result. The expected improvement
        over today's plan is the expected value of running that test, in target
        units per period. No refitting is needed, so this takes seconds.

        Columns: the 90% ROAS range today and its expected width after the test,
        ``expected_gain`` per period, and ``plan_shift``, the average spend the
        test result would move. Sorted with the most valuable test first. The
        values assume the model's form is right; treat them as a ranking.

        Plans here maximize expected sales within the usual limits (each channel
        within 50% of current spend), because the value of information is an
        expectation: under that rule a test can never be worth less than nothing,
        and estimates below zero are simulation noise. ``plan_kwargs`` change the
        limits or the rule (``max_change``, ``bounds``, ``objective``, ``quantile``).
        """
        self._require_fit()
        rng = np.random.default_rng(self.seed + 13 if seed is None else seed)
        roas = self.roas_draws()["roas"]
        n = len(roas)
        hist = self.spend_raw.mean(axis=0)
        total = float(hist.sum()) if total is None else float(total)
        objective = plan_kwargs.get("objective", "mean")
        quantile = plan_kwargs.get("quantile", 0.10)
        current, lims, _ = self._plan_limits(total, plan_kwargs.get("bounds"), plan_kwargs.get("max_change", 0.5),
                                             "ignore", 0.95)
        x_now = self._solve_plan(total, current, lims, objective, quantile)
        truth_idx = rng.choice(n, n_sim)
        noise = rng.normal(size=n_sim)
        steady_all = lambda x: self._steady(x)                           # per draw
        now_val = steady_all(x_now)
        roi = self.get_roi()
        rows = []
        for j, c in enumerate(self.media):
            s = float(sd[c]) if isinstance(sd, dict) else float(sd)
            gains, widths, shifts = [], [], []
            for i, e in zip(truth_idx, noise):
                measured = roas[i, j] + s * e
                lw = -0.5 * ((roas[:, j] - measured) / s) ** 2
                w = np.exp(lw - lw.max()); w /= w.sum()
                x_new = self._solve_plan(total, current, lims, objective, quantile, weights=w)
                gains.append(steady_all(x_new)[i] - now_val[i])
                widths.append(_weighted_quantile(roas[:, j], w, 0.95) - _weighted_quantile(roas[:, j], w, 0.05))
                shifts.append(np.abs(x_new - x_now).sum() / 2)
            rows.append({"channel": c, "roas_p05": roi.loc[c, "roas_p05"], "roas_p95": roi.loc[c, "roas_p95"],
                         "width_now": roi.loc[c, "roas_p95"] - roi.loc[c, "roas_p05"],
                         "width_after": float(np.mean(widths)), "expected_gain": float(np.mean(gains)),
                         "gain_se": float(np.std(gains) / np.sqrt(n_sim)), "plan_shift": float(np.mean(shifts))})
        return pd.DataFrame(rows).set_index("channel").sort_values("expected_gain", ascending=False)


def _weighted_quantile(v: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][min(np.searchsorted(cw, q * cw[-1]), len(v) - 1)])


def _pareto_k(log_w: np.ndarray) -> float:
    """Pareto-smoothed importance sampling tail shape k-hat (Vehtari et al.).

    Fits a generalized Pareto distribution to the largest importance ratios.
    k-hat below 0.5 is good, 0.5 to 0.7 usable, above 0.7 unreliable.
    """
    lw = np.sort(np.asarray(log_w, dtype=float))
    lw = lw[np.isfinite(lw)]
    n = len(lw)
    m = int(min(0.2 * n, 3 * np.sqrt(n)))
    if m < 5:
        return float("nan")
    w = np.exp(lw - lw[-1])
    tail, u = w[-m:], w[-m - 1]
    try:
        k, _, _ = stats.genpareto.fit(tail - u, floc=0)
    except Exception:
        return float("nan")
    return float(k)


# Closed-form log densities. scipy.stats adds ~0.1 ms of argument checking per
# call, which dominated fit time (the posterior is evaluated tens of thousands
# of times); these are the same formulas without the overhead.
_HALF_LOG_2PI = 0.5 * np.log(2 * np.pi)


def _beta_lpdf(x, a, b):
    return (a - 1) * np.log(x) + (b - 1) * np.log1p(-x) - betaln(a, b)


def _lognorm_lpdf(x, s):  # scale 1
    lx = np.log(x)
    return -lx - np.log(s) - _HALF_LOG_2PI - lx * lx / (2 * s * s)


def _halfnorm_lpdf(x, scale):
    return np.log(2.0) - _HALF_LOG_2PI - np.log(scale) - x * x / (2 * scale * scale)
