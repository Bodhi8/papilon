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
"""Geo-level marketing mix model with partial pooling across regions.

Regional data multiplies the evidence: each region's spend moves somewhat
independently, which separates channels that move together nationally. The model:

    sales[g, t] = baseline_g(t) + controls + sum_c beta[g, c] * f_c(adstock(spend[g, c])) + noise_g

Curve shapes (carryover, saturation) are shared by all regions, in each region's own
spend units (spend divided by that region's average), so a small region and a large
one saturate at the same share of their usual spend. Each region has its own effect
size beta[g, c], drawn around a pooled effect mu_c with spread tau_c: regions with
little information borrow from the rest, regions with a lot keep their own estimate.

Given the curve shapes, the model is linear in the baselines and the regional effects,
so those are integrated out exactly (a conjugate normal and inverse-gamma prior per
region, with the pooling prior on the effects). Sampling runs only over the shared
shapes, mu and tau: four or five parameters a channel whatever the number of regions,
which is what keeps Papilon's importance sampler accurate on dozens of regions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln

from .model import MMM, Family, GateResult  # noqa: F401  (GateResult re-exported for persistence)

__all__ = ["GeoMMM"]


class GeoMMM(MMM):
    """Marketing mix model across regions with partial pooling.

    Parameters
    ----------
    data : long DataFrame, one row per region and period.
    target, media, date_col, controls : as :class:`papilon.MMM`.
    geo_col : column naming the region (state, DMA, market).
    carryover, saturation, delayed, s_curve : model family, shared by all regions.
    yearly_seasonality, trend, l_max, period : baseline settings, applied per region
        (each region gets its own baseline coefficients).
    roas_prior_scale : prior scale for the pooled return on spend (default 2.0).
    escalate : "smc" (default) or "none", as :class:`papilon.MMM`.

    Not yet available in geo models: time-varying effects, reach and frequency,
    organic channels and lift tests. Fit a national :class:`papilon.MMM` for those.
    """

    def __init__(self, data: pd.DataFrame, target: str, media: list[str], date_col: str, geo_col: str,
                 controls: list[str] | None = None, yearly_seasonality: int = 2, trend: bool = True,
                 l_max: int = 8, period: float = 52.18, seed: int = 0,
                 carryover: dict | None = None, saturation: dict | None = None,
                 delayed: list[str] | None = None, s_curve: list[str] | None = None,
                 alpha_prior=(2.0, 2.0), roas_prior_scale: float = 2.0, escalate: str = "smc", **unused):
        bad = {k for k, v in unused.items() if v not in (None, [], {}, 0, 3, "smc")}
        if bad:
            raise ValueError(f"not available in geo models yet: {sorted(bad)}")
        self._config = dict(target=target, media=list(media), date_col=date_col, geo_col=geo_col,
                            controls=list(controls or []), yearly_seasonality=yearly_seasonality, trend=trend,
                            l_max=l_max, period=period, alpha_prior=alpha_prior,
                            roas_prior_scale=roas_prior_scale, escalate=escalate)
        self.target, self.media, self.date_col, self.geo_col = target, list(media), date_col, geo_col
        self.channels, self.organic, self.controls = list(media), [], list(controls or [])
        self.l_max, self.period, self.seed, self.escalate = int(l_max), float(period), seed, escalate
        self.rng = np.random.default_rng(seed)
        self.n_tv, self.reach_frequency, self._rf_stats, self._lift, self._roi_idx = 3, {}, {}, [], []
        self._lags = np.arange(self.l_max)
        self._fam = Family(self.channels, dict(carryover or {}) | {c: "delayed" for c in (delayed or [])},
                           dict(saturation or {}) | {c: "s_curve" for c in (s_curve or [])})
        self.alpha_prior = ({c: tuple(alpha_prior.get(c, (2.0, 2.0))) for c in self.channels}
                            if isinstance(alpha_prior, dict) else {c: tuple(alpha_prior) for c in self.channels})
        missing = [c for c in [target, date_col, geo_col] + self.media + self.controls if c not in data.columns]
        if missing:
            raise ValueError(f"columns not in data: {missing}")
        df = data.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values([geo_col, date_col]).reset_index(drop=True)
        self.data = df
        self.geos = list(dict.fromkeys(df[geo_col]))
        counts = df.groupby(geo_col)[date_col].nunique()
        if counts.nunique() != 1 or df.groupby(geo_col)[date_col].apply(lambda s: s.duplicated().any()).any():
            raise ValueError("every region needs the same periods, once each")
        self.dates = pd.DatetimeIndex(sorted(df[date_col].unique()))
        T, G, K = len(self.dates), len(self.geos), len(self.media)
        self._T_fit = T
        Y = np.stack([df.loc[df[geo_col] == g, target].to_numpy(dtype=float) for g in self.geos])        # (G, T)
        S = np.stack([df.loc[df[geo_col] == g, self.media].to_numpy(dtype=float) for g in self.geos])   # (G, T, K)
        if np.any(S < 0) or not np.all(np.isfinite(S)) or not np.all(np.isfinite(Y)):
            raise ValueError("spend must be non-negative and target and spend complete")
        self._Y_raw, self._S_raw = Y, S
        self._y_g = Y.mean(axis=1)                                   # (G,)
        self._x_g = S.mean(axis=1); self._x_g[self._x_g == 0] = 1.0  # (G, K)
        self._Yn = Y / self._y_g[:, None]
        self._Wg = [np.stack([self._lagged(S[g, :, c] / self._x_g[g, c]) for c in range(K)]) for g in range(G)]
        # national series, used by the planner
        self.spend_raw = S.sum(axis=0)
        self.spend_scale = self.spend_raw.mean(axis=0); self.spend_scale[self.spend_scale == 0] = 1.0
        self.y_raw, self.y_scale = Y.sum(axis=0), 1.0
        self._spend_tot = self.spend_raw.sum(axis=0)
        self._share = S.sum(axis=1) / np.maximum(S.sum(axis=(0, 1)), 1e-12)          # (G, K) share of channel spend
        # pooled effect prior: ROAS ~ beta * 0.5 * y_g / x_g, so beta scale = roas * x_g / (0.5 * y_g), pooled
        ratio = np.median(self._x_g / self._y_g[:, None], axis=0)
        self._b_scale = {c: float(roas_prior_scale) * ratio[j] / 0.5 for j, c in enumerate(self.media)}
        self._trend, self._harmonics, self._drift = bool(trend), int(yearly_seasonality), 0
        self._ctrl_stats = []
        for c in self.controls:
            v = df[c].to_numpy(dtype=float)
            self._ctrl_stats.append((float(v.mean()), float(v.std() or 1.0)))
        self._X = [self._design(df.loc[df[geo_col] == g].reset_index(drop=True), 0) for g in self.geos]
        p = self._X[0].shape[1]
        sds = [100.0] + ([10.0] if trend else []) + [10.0] * (2 * yearly_seasonality) + [20.0] * len(self.controls)
        self._sd_b2 = np.array(sds) ** 2
        self._a0, self._an = 2.0, 2.0 + T / 2
        self._b0 = 0.1 * np.var(self._Yn, axis=1)                                    # (G,)
        self._XtX = [X.T @ X for X in self._X]
        self._Xty = [X.T @ y for X, y in zip(self._X, self._Yn)]
        self._yty = (self._Yn ** 2).sum(axis=1)
        # plug-in noise scale per region (from a rough fit), which sets the pooling prior in sigma units
        self._s_hat = np.empty(G)
        for g in range(G):
            F = np.stack([w @ (0.5 ** self._lags / (0.5 ** self._lags).sum()) for w in self._Wg[g]], axis=1)
            Z = np.column_stack([self._X[g], F / (F + 1.0)])
            coef, *_ = np.linalg.lstsq(Z, self._Yn[g], rcond=None)
            self._s_hat[g] = max(float(np.std(self._Yn[g] - Z @ coef)), 0.02)
        self._p = p
        self.gate_results, self.family_probabilities, self._draws = {}, {}, None

    # ---------------------------------------------------------------- model
    def _layout(self, fam):
        return [names + ["zt"] for names in MMM._layout(self, fam)]

    def _features(self, P, fam, g):
        """Saturated carried-over spend of region g, shape (draws, periods, channels)."""
        rows = np.arange(self._T_fit)
        return np.stack([self._saturate(self._adstock(P, j, fam.carry[c], ("w", self._Wg[g][j]), rows), P, j, fam.sat[c])
                         for j, c in enumerate(self.media)], axis=-1).transpose(1, 0, 2)

    def _geo_posterior(self, P, fam, g):
        """Per draw: A (precision), rhs, the effect-prior variance, and pieces of the evidence for region g."""
        F = self._features(P, fam, g)                                             # (n, T, K)
        n, K, p = F.shape[0], F.shape[2], self._p
        mu = P["beta"]                                                            # (n, K), pooled effect
        v = (P["tau"] * mu / self._s_hat[g]) ** 2 + 1e-10                          # (n, K)
        X = self._X[g]
        A = np.zeros((n, p + K, p + K))
        A[:, :p, :p] = self._XtX[g] + np.diag(1 / self._sd_b2)
        XtF = np.einsum("tp,ntk->npk", X, F)
        A[:, :p, p:] = XtF; A[:, p:, :p] = XtF.transpose(0, 2, 1)
        A[:, p:, p:] = np.einsum("ntk,ntl->nkl", F, F) + np.eye(K)[None] / v[:, None, :]
        rhs = np.concatenate([np.broadcast_to(self._Xty[g], (n, p)),
                              np.einsum("ntk,t->nk", F, self._Yn[g]) + mu / v], axis=1)
        return A, rhs, v, mu

    def _log_post_batch(self, Z, fam, shaped=None):
        fam = self._as_family(fam, shaped)
        P, lp = self._unpack_batch(np.atleast_2d(Z), fam)
        total = np.array(lp, dtype=float)
        with np.errstate(all="ignore"):
            for g in range(len(self.geos)):
                A, rhs, v, mu = self._geo_posterior(P, fam, g)
                try:
                    mn = np.linalg.solve(A, rhs[..., None])[..., 0]
                    sign, logdet = np.linalg.slogdet(A)
                except np.linalg.LinAlgError:
                    return np.full(len(total), -np.inf)
                quad = np.sum(mu * mu / v, axis=1)
                bn = self._b0[g] + 0.5 * (self._yty[g] + quad - np.sum(mn * rhs, axis=1))
                logdet_v0 = np.sum(np.log(self._sd_b2)) + np.sum(np.log(v), axis=1)
                T = self._T_fit
                total = total + (-T / 2 * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * logdet_v0
                                 + self._a0 * np.log(self._b0[g]) - self._an * np.log(np.maximum(bn, 1e-300))
                                 + gammaln(self._an) - gammaln(self._a0))
                total = np.where((sign > 0) & (bn > 0), total, -np.inf)
        return np.where(np.isfinite(total), total, -np.inf)

    def _fast_ok(self, fam):
        return False

    def _store_draws(self, fam, z, w, n_keep=1000):
        idx = self.rng.choice(len(w), n_keep, p=w)
        P, _ = self._unpack_batch(z[idx], fam)
        G, K, p = len(self.geos), len(self.media), self._p
        betas = np.empty((n_keep, G, K))
        for g in range(G):
            A, rhs, v, mu = self._geo_posterior(P, fam, g)
            mn = np.linalg.solve(A, rhs[..., None])[..., 0]
            bn = self._b0[g] + 0.5 * (self._yty[g] + np.sum(mu * mu / v, axis=1) - np.sum(mn * rhs, axis=1))
            cov = np.linalg.inv(A)[:, p:, p:] * (bn / self._an)[:, None, None]
            L = np.linalg.cholesky(cov + 1e-12 * np.eye(K))
            t = self.rng.standard_normal((n_keep, K)) * np.sqrt(2 * self._an / self.rng.chisquare(2 * self._an, n_keep))[:, None]
            betas[:, g, :] = mn[:, p:] + np.einsum("nkl,nl->nk", L, t)
        P["geo_beta"] = betas
        self._draws, self._fam = P, fam
        self._roi_cache = self._roas_cache = None

    # -------------------------------------------------------------- results
    def _geo_contrib(self, scale: float = 1.0) -> np.ndarray:
        """Contribution per draw, region, period and channel in target units, shape (n, G, T, K)."""
        d, fam = self._draws, self._fam
        out = []
        for g in range(len(self.geos)):
            F = self._features(d, fam, g) if scale == 1.0 else np.stack(
                [self._saturate(self._adstock(d, j, fam.carry[c], ("w", self._Wg[g][j] * scale),
                                              np.arange(self._T_fit)), d, j, fam.sat[c])
                 for j, c in enumerate(self.media)], axis=-1).transpose(1, 0, 2)
            out.append(F * d["geo_beta"][:, g, None, :] * self._y_g[g])
        return np.stack(out, axis=1)

    def roas_draws(self) -> dict:
        self._require_fit()
        if getattr(self, "_roas_cache", None) is None:
            c = self._geo_contrib()
            m = c.sum(axis=(1, 2))
            bumped = self._geo_contrib(1.01).sum(axis=(1, 2))
            by_geo = c.sum(axis=2) / np.maximum(self._S_raw.sum(axis=1), 1e-12)[None]
            self._roas_cache = {"roas": m / self._spend_tot, "mroas": (bumped - m) / (0.01 * self._spend_tot),
                                "share": m / self.y_raw.sum(), "by_geo": by_geo}
        return self._roas_cache

    def get_roi(self, by_geo: bool = False) -> pd.DataFrame:
        """National return on spend per channel (all regions), or one row per region and channel."""
        self._require_fit()
        r = self.roas_draws()
        if not by_geo:
            return MMM.get_roi(self)
        rows = []
        for g, geo in enumerate(self.geos):
            for j, c in enumerate(self.media):
                v = r["by_geo"][:, g, j]
                rows.append({self.geo_col: geo, "channel": c, "spend": self._S_raw[g, :, j].sum(),
                             "roas_mean": v.mean(), "roas_p05": np.quantile(v, 0.05), "roas_p95": np.quantile(v, 0.95)})
        return pd.DataFrame(rows)

    def pooling(self) -> pd.DataFrame:
        """Per channel: pooled effect and spread across regions (tau, as a share of the pooled effect)."""
        self._require_fit()
        d = self._draws
        return pd.DataFrame({"tau_mean": d["tau"].mean(0), "tau_p05": np.quantile(d["tau"], 0.05, axis=0),
                             "tau_p95": np.quantile(d["tau"], 0.95, axis=0),
                             "negative_effect_share": (d["geo_beta"] < 0).mean(axis=(0, 1))},
                            index=pd.Index(self.media, name="channel"))

    def contributions(self) -> pd.DataFrame:
        """Posterior-mean contribution per region, period and channel (long format)."""
        self._require_fit()
        c = self._geo_contrib().mean(axis=0)
        rows = []
        for g, geo in enumerate(self.geos):
            part = pd.DataFrame(c[g], columns=self.media)
            part.insert(0, self.date_col, self.dates); part.insert(1, self.geo_col, geo)
            rows.append(part)
        return pd.concat(rows, ignore_index=True)

    def _curve(self, xs, P=None, fam=None):
        """National steady-state contribution per draw and channel at scaled national spend xs,
        with each channel's spend split across regions in its historical shares."""
        P = self._draws if P is None else P
        fam = self._fam if fam is None else fam
        n, K = len(P["beta"]), len(self.media)
        xs = np.broadcast_to(np.asarray(xs, dtype=float), (n, K))
        out = np.zeros((n, K))
        for g in range(len(self.geos)):
            for j, c in enumerate(self.media):
                xg = xs[:, j] * self.spend_scale[j] * self._share[g, j] / self._x_g[g, j]
                out[:, j] += P["geo_beta"][:, g, j] * self._saturate(xg[None, :], P, j, fam.sat[c])[0] * self._y_g[g]
        return out

    def parameters(self) -> pd.DataFrame:
        self._require_fit()
        d, fam = self._draws, self._fam
        return pd.DataFrame({"carryover": [fam.carry[c] for c in self.media],
                             "saturation": [fam.sat[c] for c in self.media],
                             "alpha": np.nanmean(d["alpha"], axis=0), "half_saturation_share_of_usual_spend":
                             d["half_sat"].mean(0), "pooled_effect": d["beta"].mean(0), "tau": d["tau"].mean(0)},
                            index=pd.Index(self.media, name="channel"))

    def _not_yet(self, *a, **k):
        raise NotImplementedError("not available for geo models yet; fit a national papilon.MMM on summed data")

    backtest = self_check = value_of_lift_test = forecast = scenarios = monitor = decompose = _not_yet
    response_curve = posterior_curves = future_frame = _not_yet
