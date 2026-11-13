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

With ``regional_curves=True`` (1.1) each region also has its own carryover and
saturation parameters, drawn around the pooled value with a spread the data sets.
Regional effects and baselines are still integrated out exactly, so the sampled
parameters are the curve shapes alone: two or three per channel and region. That is
hundreds to thousands of parameters, which is the job of the NUTS sampler on the
optional JAX backend (``pip install "papilon[jax]"``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln

from .model import (MMM, Family, GateResult, _HALF_LOG_2PI, _check_sampler,  # noqa: F401  (GateResult re-exported)
                    _halfnorm_lpdf)

__all__ = ["GeoMMM"]

_REGIONAL = ("za", "zu", "zwk", "zwl", "zh", "zs")   # curve parameters that vary by region under regional_curves
_REGIONAL_SD = 0.5                                   # half-normal scale of their spread, on the unconstrained scale
_REGIONAL_LABEL = {"za": "carryover", "zu": "peak", "zwk": "weibull_shape", "zwl": "weibull_scale",
                   "zh": "half_saturation", "zs": "shape"}
_DENSE_CELLS = 2e7                                   # above this many draw-region-period-channel cells, work by region


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
    lift_tests : regional experiments, as a DataFrame or list of dicts with the region
        (under the name of ``geo_col``, or ``geo``), ``channel``, ``start``, ``end``,
        ``lift`` (incremental outcome in that region over the test plus ``readout``
        periods), ``se``, and ``factor`` or ``spend_change``, as :class:`papilon.MMM`.
        Each test calibrates its own region's effect, and the other regions through
        pooling. A test measured as one number over several regions is not supported:
        enter it per region, or fit a national model.
    sampler : "auto" (default), "gate" or "nuts", as :class:`papilon.MMM`.
    regional_curves : give every region its own carryover and saturation parameters,
        partially pooled toward the shared ones (default False: shapes are shared).
        The sampled parameters then grow with the number of regions, so "auto" fits
        with NUTS when the JAX backend is installed.

    Notes
    -----
    Geo models do not yet have time-varying effects, reach and frequency or organic
    channels. Fit a national :class:`papilon.MMM` for those.
    """

    def __init__(self, data: pd.DataFrame, target: str, media: list[str], date_col: str, geo_col: str,
                 controls: list[str] | None = None, yearly_seasonality: int = 2, trend: bool = True,
                 l_max: int = 8, period: float = 52.18, seed: int = 0,
                 carryover: dict | None = None, saturation: dict | None = None,
                 delayed: list[str] | None = None, s_curve: list[str] | None = None,
                 alpha_prior=(2.0, 2.0), roas_prior_scale: float = 2.0, escalate: str = "smc",
                 lift_tests=None, sampler: str = "auto", regional_curves: bool = False, **unused):
        bad = {k for k, v in unused.items() if v not in (None, [], {}, 0, 3, "smc")}
        if bad:
            raise ValueError(f"not available in geo models yet: {sorted(bad)}")
        self._config = dict(target=target, media=list(media), date_col=date_col, geo_col=geo_col,
                            controls=list(controls or []), yearly_seasonality=yearly_seasonality, trend=trend,
                            l_max=l_max, period=period, alpha_prior=alpha_prior,
                            roas_prior_scale=roas_prior_scale, escalate=escalate, sampler=sampler,
                            regional_curves=bool(regional_curves),
                            lift_tests=None if lift_tests is None else pd.DataFrame(lift_tests).to_dict("records"))
        self.sampler, self.regional_curves = _check_sampler(sampler), bool(regional_curves)
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
        self._geo_lift = self._parse_geo_lift_tests(lift_tests)
        self.gate_results, self.family_probabilities, self._draws = {}, {}, None

    def _parse_geo_lift_tests(self, lift_tests) -> list:
        """Per region, the lift tests that ran there, in that region's model units."""
        out = [[] for _ in self.geos]
        if lift_tests is None:
            return out
        tests = pd.DataFrame(lift_tests)
        if tests.empty:
            return out
        key = self.geo_col if self.geo_col in tests.columns else "geo"
        need = {key, "channel", "start", "end", "lift", "se"}
        if not need <= set(tests.columns):
            raise ValueError(f"geo lift_tests needs columns {sorted(need)} plus factor or spend_change")
        for _, t in tests.iterrows():
            c, geo = t["channel"], t[key]
            if c not in self.media:
                raise ValueError(f"lift test channel {c!r} is not in media")
            if geo not in self.geos:
                raise ValueError(f"lift test region {geo!r} is not in the data")
            g, j = self.geos.index(geo), self.media.index(c)
            inside = np.where((self.dates >= pd.Timestamp(t["start"])) & (self.dates <= pd.Timestamp(t["end"])))[0]
            if len(inside) == 0:
                raise ValueError(f"lift test on {c} in {geo} from {t['start']} to {t['end']} has no periods in the data")
            r0, r1 = int(inside[0]), int(inside[-1]) + 1
            readout = int(t["readout"]) if "readout" in t and pd.notna(t.get("readout")) else self.l_max - 1
            spend = self._S_raw[g, :, j]
            if "factor" in t and pd.notna(t.get("factor")):
                factor = float(t["factor"])
            elif "spend_change" in t and pd.notna(t.get("spend_change")):
                base = spend[r0:r1].sum()
                if base <= 0:
                    raise ValueError(f"lift test on {c} in {geo}: no spend in the test window to scale")
                factor = 1 + float(t["spend_change"]) / base
            else:
                raise ValueError("each lift test needs factor or spend_change")
            if factor < 0:
                raise ValueError("a lift test cannot reduce spend below zero")
            if not float(t["se"]) > 0:
                raise ValueError("a lift test needs a positive standard error")
            cf = spend.copy()
            cf[r0:r1] *= factor
            se = float(t["se"]) / self._y_g[g]
            # The region's noise scale is integrated out, so the test enters as one more observation of that
            # region, weighted so that at the plug-in noise scale it has the stated standard error.
            out[g].append({"j": j, "rows": np.arange(r0, min(self._T_fit, r1 + readout)),
                           "W_cf": self._lagged(cf / self._x_g[g, j]), "lift": float(t["lift"]) / self._y_g[g],
                           "w": float(self._s_hat[g] / se), "channel": c, "geo": geo, "factor": factor})
        return out

    # ---------------------------------------------------------------- model
    def _layout(self, fam):
        return [names + ["zt"] for names in MMM._layout(self, fam)]

    def _regional_index(self, fam) -> np.ndarray:
        """Positions, in the shared block, of the curve parameters that vary by region."""
        flat = [nm for names in self._layout(fam) for nm in names]
        return np.array([i for i, nm in enumerate(flat) if nm in _REGIONAL], dtype=int)

    def _dim(self, fam, shaped=None) -> int:
        d0 = MMM._dim(self, fam, shaped)
        if not self.regional_curves:
            return d0
        return d0 + len(self._regional_index(self._as_family(fam, shaped))) * (1 + len(self.geos))

    def _init_point(self, fam):
        z0 = MMM._init_point(self, fam)
        if not self.regional_curves:
            return z0
        nr = len(self._regional_index(fam))
        return np.concatenate([z0, np.full(nr, np.log(0.15)), np.zeros(nr * len(self.geos))])

    def _unpack_batch(self, Z, fam, shaped=None):
        """As :meth:`MMM._unpack_batch`. With regional curves the vector is the shared block, then the log
        spread of each regional parameter, then one standard-normal offset per region and parameter."""
        fam = self._as_family(fam, shaped)
        Z = np.atleast_2d(Z)
        if not self.regional_curves:
            return MMM._unpack_batch(self, Z, fam)
        d0, ridx, G = MMM._dim(self, fam), self._regional_index(fam), len(self.geos)
        nr = len(ridx)
        P, lp = MMM._unpack_batch(self, Z[:, :d0], fam)
        log_sd = Z[:, d0:d0 + nr]
        eps = Z[:, d0 + nr:].reshape(len(Z), G, nr)
        with np.errstate(all="ignore"):
            sd = np.exp(log_sd)
            lp = lp + np.sum(_halfnorm_lpdf(sd, _REGIONAL_SD) + log_sd, axis=1) \
                + np.sum(-0.5 * eps * eps - _HALF_LOG_2PI, axis=(1, 2))
            zgeo = np.repeat(Z[:, None, :d0], G, axis=1)
            zgeo[:, :, ridx] += sd[:, None, :] * eps
        P["zgeo"], P["regional_sd"] = zgeo, sd
        return P, lp

    def _region(self, P, fam, g):
        """The parameters of region g: the shared ones, or its own under regional curves."""
        if "zgeo" not in P:
            return P
        Pg, _ = MMM._unpack_batch(self, P["zgeo"][:, g, :], fam)
        return Pg

    def _features(self, P, fam, g, scale: float = 1.0, W=None, rows=None):
        """Saturated carried-over spend of region g, shape (draws, periods, channels)."""
        rows = np.arange(self._T_fit) if rows is None else rows
        W = self._Wg[g] if W is None else W
        Pg = self._region(P, fam, g)
        return np.stack([self._saturate(self._adstock(Pg, j, fam.carry[c], ("w", W[j] if scale == 1.0 else W[j] * scale),
                                                      rows), Pg, j, fam.sat[c])
                         for j, c in enumerate(self.media)], axis=-1).transpose(1, 0, 2)

    def _geo_posterior(self, P, fam, g):
        """Per draw: A (precision), rhs, the effect-prior variance, the pooled effect, y'y, and
        (number of lift tests, their constant) for region g."""
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
        yty, const = self._yty[g], 0.0
        if self._geo_lift[g]:
            Pg = self._region(P, fam, g)
            for t in self._geo_lift[g]:
                j, c, r = t["j"], self.media[t["j"]], t["rows"]
                d = (self._saturate(self._adstock(Pg, j, fam.carry[c], ("w", t["W_cf"]), r), Pg, j, fam.sat[c])
                     - self._saturate(self._adstock(Pg, j, fam.carry[c], ("w", self._Wg[g][j]), r), Pg, j, fam.sat[c])
                     ).sum(axis=0) * t["w"]                                       # (n,)
                A[:, p + j, p + j] += d * d
                rhs[:, p + j] += d * (t["w"] * t["lift"])
                yty = yty + (t["w"] * t["lift"]) ** 2
                const += float(np.log(t["w"]))
        return A, rhs, v, mu, yty, (len(self._geo_lift[g]), const)

    def _log_post_batch(self, Z, fam, shaped=None):
        fam = self._as_family(fam, shaped)
        P, lp = self._unpack_batch(np.atleast_2d(Z), fam)
        total = np.array(lp, dtype=float)
        with np.errstate(all="ignore"):
            for g in range(len(self.geos)):
                A, rhs, v, mu, yty, (m, const) = self._geo_posterior(P, fam, g)
                try:
                    mn = np.linalg.solve(A, rhs[..., None])[..., 0]
                    sign, logdet = np.linalg.slogdet(A)
                except np.linalg.LinAlgError:
                    return np.full(len(total), -np.inf)
                quad = np.sum(mu * mu / v, axis=1)
                bn = self._b0[g] + 0.5 * (yty + quad - np.sum(mn * rhs, axis=1))
                logdet_v0 = np.sum(np.log(self._sd_b2)) + np.sum(np.log(v), axis=1)
                T, an = self._T_fit + m, self._an + 0.5 * m
                total = total + (-T / 2 * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * logdet_v0
                                 + self._a0 * np.log(self._b0[g]) - an * np.log(np.maximum(bn, 1e-300))
                                 + gammaln(an) - gammaln(self._a0))
                if m:
                    total = total + const
                total = np.where((sign > 0) & (bn > 0), total, -np.inf)
        return np.where(np.isfinite(total), total, -np.inf)

    def _fast_ok(self, fam):
        return False

    def _store_draws(self, fam, z, w, n_keep=1000, thin=False):
        idx = self._pick(w, n_keep, thin)
        n_keep = len(idx)
        P, _ = self._unpack_batch(z[idx], fam)
        G, K, p = len(self.geos), len(self.media), self._p
        betas = np.empty((n_keep, G, K))
        for g in range(G):
            A, rhs, v, mu, yty, (m, _) = self._geo_posterior(P, fam, g)
            an = self._an + 0.5 * m
            mn = np.linalg.solve(A, rhs[..., None])[..., 0]
            bn = self._b0[g] + 0.5 * (yty + np.sum(mu * mu / v, axis=1) - np.sum(mn * rhs, axis=1))
            cov = np.linalg.inv(A)[:, p:, p:] * (bn / an)[:, None, None]
            L = np.linalg.cholesky(cov + 1e-12 * np.eye(K))
            t = self.rng.standard_normal((n_keep, K)) * np.sqrt(2 * an / self.rng.chisquare(2 * an, n_keep))[:, None]
            betas[:, g, :] = mn[:, p:] + np.einsum("nkl,nl->nk", L, t)
        P["geo_beta"] = betas
        self._draws, self._fam = P, fam
        self._roi_cache = self._roas_cache = None

    # -------------------------------------------------------------- results
    def _region_contrib(self, g: int, scale: float = 1.0) -> np.ndarray:
        """Contribution of region g per draw, period and channel in target units, shape (n, T, K)."""
        d = self._draws
        return self._features(d, self._fam, g, scale) * d["geo_beta"][:, g, None, :] * self._y_g[g]

    def _geo_contrib(self, scale: float = 1.0) -> np.ndarray:
        """Contribution per draw, region, period and channel in target units, shape (n, G, T, K)."""
        return np.stack([self._region_contrib(g, scale) for g in range(len(self.geos))], axis=1)

    def _dense(self) -> bool:
        """Whether the full (draws, regions, periods, channels) array is small enough to hold at once."""
        return len(self._draws["beta"]) * len(self.geos) * self._T_fit * len(self.media) <= _DENSE_CELLS

    def roas_draws(self) -> dict:
        self._require_fit()
        if getattr(self, "_roas_cache", None) is None:
            if self._dense():
                c = self._geo_contrib()
                m = c.sum(axis=(1, 2))
                cb = self._geo_contrib(1.01)
                bumped = cb.sum(axis=(1, 2))
                spend_g = np.maximum(self._S_raw.sum(axis=1), 1e-12)[None]
                by_geo, m_by_geo = c.sum(axis=2) / spend_g, (cb.sum(axis=2) - c.sum(axis=2)) / (0.01 * spend_g)
            else:                                   # hundreds of regions: one region at a time
                n, G, K = len(self._draws["beta"]), len(self.geos), len(self.media)
                m, bumped = np.zeros((n, K)), np.zeros((n, K))
                by_geo, m_by_geo = np.empty((n, G, K)), np.empty((n, G, K))
                for g in range(G):
                    cg, cbg = self._region_contrib(g).sum(axis=1), self._region_contrib(g, 1.01).sum(axis=1)
                    m += cg
                    bumped += cbg
                    spend_g = np.maximum(self._S_raw[g].sum(axis=0), 1e-12)
                    by_geo[:, g, :], m_by_geo[:, g, :] = cg / spend_g, (cbg - cg) / (0.01 * spend_g)
            self._roas_cache = {"roas": m / self._spend_tot, "mroas": (bumped - m) / (0.01 * self._spend_tot),
                                "share": m / self.y_raw.sum(), "by_geo": by_geo, "mroas_by_geo": m_by_geo}
        return self._roas_cache

    def get_roi(self, by_geo: bool = False) -> pd.DataFrame:
        """National return on spend per channel (all regions), or one row per region and channel
        (with the marginal return of 1% more spend in that region since 1.1)."""
        self._require_fit()
        r = self.roas_draws()
        if not by_geo:
            return MMM.get_roi(self)
        rows = []
        for g, geo in enumerate(self.geos):
            for j, c in enumerate(self.media):
                v, mv = r["by_geo"][:, g, j], r["mroas_by_geo"][:, g, j]
                rows.append({self.geo_col: geo, "channel": c, "spend": self._S_raw[g, :, j].sum(),
                             "roas_mean": v.mean(), "roas_p05": np.quantile(v, 0.05), "roas_p95": np.quantile(v, 0.95),
                             "mroas_mean": mv.mean(), "mroas_p05": np.quantile(mv, 0.05),
                             "mroas_p95": np.quantile(mv, 0.95)})
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
        c = (self._geo_contrib().mean(axis=0) if self._dense()
             else np.stack([self._region_contrib(g).mean(axis=0) for g in range(len(self.geos))]))
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
            Pg = self._region(P, fam, g)
            for j, c in enumerate(self.media):
                xg = xs[:, j] * self.spend_scale[j] * self._share[g, j] / self._x_g[g, j]
                out[:, j] += P["geo_beta"][:, g, j] * self._saturate(xg[None, :], Pg, j, fam.sat[c])[0] * self._y_g[g]
        return out

    def parameters(self, by_geo: bool = False) -> pd.DataFrame:
        """Posterior means of the pooled parameters per channel, or, with ``by_geo``, of each
        region's own curve parameters and effect (one row per region and channel)."""
        self._require_fit()
        d, fam = self._draws, self._fam
        if by_geo:
            rows = []
            for g, geo in enumerate(self.geos):
                Pg = self._region(d, fam, g)
                for j, c in enumerate(self.media):
                    rows.append({self.geo_col: geo, "channel": c, "alpha": np.nanmean(Pg["alpha"][:, j]),
                                 "peak_lag": np.nanmean(Pg["peak"][:, j]) if fam.carry[c] == "delayed" else np.nan,
                                 "half_saturation_share_of_usual_spend": Pg["half_sat"][:, j].mean(),
                                 "shape": Pg["shape"][:, j].mean(), "effect": d["geo_beta"][:, g, j].mean()})
            return pd.DataFrame(rows)
        return pd.DataFrame({"carryover": [fam.carry[c] for c in self.media],
                             "saturation": [fam.sat[c] for c in self.media],
                             "alpha": np.nanmean(d["alpha"], axis=0), "half_saturation_share_of_usual_spend":
                             d["half_sat"].mean(0), "pooled_effect": d["beta"].mean(0), "tau": d["tau"].mean(0)},
                            index=pd.Index(self.media, name="channel"))

    def regional_spread(self) -> pd.DataFrame:
        """With regional curves: how far regions' curve parameters spread around the pooled value.

        One row per channel and parameter, with the posterior mean and 90% range of the
        spread on the parameter's unconstrained scale (logit for carryover retention, log
        for the rest). A spread near zero means regions share that curve.
        """
        self._require_fit()
        if "regional_sd" not in self._draws:
            raise ValueError("curves are shared across regions; fit with regional_curves=True")
        sd, rows, i = self._draws["regional_sd"], [], 0
        for c, names in zip(self.media, self._layout(self._fam)):
            for nm in names:
                if nm in _REGIONAL:
                    rows.append({"channel": c, "parameter": _REGIONAL_LABEL[nm], "spread_mean": sd[:, i].mean(),
                                 "spread_p05": np.quantile(sd[:, i], 0.05), "spread_p95": np.quantile(sd[:, i], 0.95)})
                    i += 1
        return pd.DataFrame(rows)

    def compare_models(self, candidates, *args, **kwargs) -> pd.DataFrame:
        if self.regional_curves:
            raise ValueError("compare families with shared curves (regional_curves=False), which the gate can "
                             "weigh by Bayes factor, then fit the winner with regional curves")
        return MMM.compare_models(self, candidates, *args, **kwargs)

    # ---------------------------------------------------------- forecasting
    def future_frame(self, periods, spend=None, controls=None, start=None, *a, **k) -> pd.DataFrame:
        """A long frame of ``periods`` future periods for every region, for :meth:`forecast`.

        ``spend`` maps channels to national spend per period (a number, or a sequence of
        length ``periods`` for flighting), split across regions in each channel's
        historical shares; channels left out continue at each region's historical mean.
        ``controls`` maps a control to a value or sequence used in every region; controls
        left out are held at each region's mean.
        """
        _no_extra(a, k)
        step = pd.Series(self.dates).diff().median()
        dates = pd.date_range(self.dates[-1] + step if start is None else pd.Timestamp(start), periods=periods, freq=step)
        spend, controls = dict(spend or {}), dict(controls or {})
        unknown = (set(spend) - set(self.media)) | (set(controls) - set(self.controls))
        if unknown:
            raise ValueError(f"not channels or controls of this model: {sorted(unknown)}")
        parts = []
        for g, geo in enumerate(self.geos):
            hist = self.data.loc[self.data[self.geo_col] == geo]
            part = pd.DataFrame({self.date_col: dates, self.geo_col: geo})
            for j, c in enumerate(self.media):
                v = np.asarray(spend[c], dtype=float) * self._share[g, j] if c in spend else self._S_raw[g, :, j].mean()
                part[c] = np.broadcast_to(v, (periods,)).copy()
            for c in self.controls:
                v = controls.get(c, hist[c].mean())
                part[c] = np.broadcast_to(np.asarray(v, dtype=float), (periods,)).copy()
            parts.append(part)
        return pd.concat(parts, ignore_index=True)

    def _predict(self, future: pd.DataFrame, seed: int | None = None) -> dict:
        """Posterior predictive draws per region for periods that follow the fitted window.

        Carryover from the fitted window is included. Given the curve parameters, each
        region's baseline and effects are integrated out, so every period's predictive
        distribution is a Student t, as in :meth:`MMM._predict`.
        """
        self._require_fit()
        missing = [c for c in [self.geo_col, self.date_col] + self.media if c not in future.columns]
        if missing:
            raise ValueError(f"future periods need columns {missing}")
        fut = future.copy()
        fut[self.date_col] = pd.to_datetime(fut[self.date_col])
        parts = [fut.loc[fut[self.geo_col] == geo].sort_values(self.date_col).reset_index(drop=True) for geo in self.geos]
        h = len(parts[0])
        if h == 0 or any(len(q) != h for q in parts):
            raise ValueError("future periods must cover every region, with the same periods in each")
        d, fam = self._draws, self._fam
        n, G, K, p, T = len(d["beta"]), len(self.geos), len(self.media), self._p, self._T_fit
        rng = np.random.default_rng(self.seed + 1 if seed is None else seed)
        rows = np.arange(T, T + h)
        draws, eff = np.empty((G, h, n)), np.empty((G, h, n, K))
        for g in range(G):
            both = np.vstack([self._S_raw[g], parts[g][self.media].to_numpy(dtype=float)])
            if np.any(both < 0) or not np.all(np.isfinite(both)):
                raise ValueError("future spend must be non-negative and complete")
            W = [self._lagged(both[:, j] / self._x_g[g, j]) for j in range(K)]
            F = self._features(d, fam, g, W=W, rows=rows)                          # (n, h, K)
            A, rhs, v, mu, yty, (m, _) = self._geo_posterior(d, fam, g)
            an = self._an + 0.5 * m
            mn = np.linalg.solve(A, rhs[..., None])[..., 0]
            bn = self._b0[g] + 0.5 * (yty + np.sum(mu * mu / v, axis=1) - np.sum(mn * rhs, axis=1))
            x = np.concatenate([np.broadcast_to(self._design(parts[g], T)[None], (n, h, p)), F], axis=2)
            mean = np.einsum("nhi,ni->hn", x, mn)
            lev = 1 + np.einsum("nhi,nij,nhj->hn", x, np.linalg.inv(A), x)
            scale = np.sqrt(np.maximum(bn, 0.0) / an * lev)
            draws[g] = (mean + scale * rng.standard_t(2 * an, size=mean.shape)) * self._y_g[g]
            eff[g] = (F * d["geo_beta"][:, g, None, :]).transpose(1, 0, 2) * self._y_g[g]
        return {"draws": draws, "effects": eff, "dates": parts[0][self.date_col].to_numpy()}

    def forecast(self, future, *a, **k) -> pd.DataFrame:
        """Forecast the target for future periods in every region under a spend plan.

        ``future`` is a long frame with the region, date, each channel's spend and the
        controls (see :meth:`future_frame`). Returns one row per region and period with
        the predicted mean, the 5th and 95th percentiles and each channel's contribution.
        ``attrs["national"]`` holds the same summary for the sum over regions, and
        ``attrs["total_draws"]`` the posterior draws of the total over regions and periods.
        """
        _no_extra(a, k)
        p = self._predict(future)
        parts = []
        for g, geo in enumerate(self.geos):
            part = pd.DataFrame({self.date_col: p["dates"], self.geo_col: geo, "predicted": p["draws"][g].mean(axis=1),
                                 "p05": np.quantile(p["draws"][g], 0.05, axis=1),
                                 "p95": np.quantile(p["draws"][g], 0.95, axis=1)})
            for j, c in enumerate(self.media):
                part[c] = p["effects"][g][:, :, j].mean(axis=1)
            parts.append(part)
        out = pd.concat(parts, ignore_index=True)
        nat = p["draws"].sum(axis=0)                                               # (h, n)
        out.attrs["national"] = pd.DataFrame({self.date_col: p["dates"], "predicted": nat.mean(axis=1),
                                              "p05": np.quantile(nat, 0.05, axis=1),
                                              "p95": np.quantile(nat, 0.95, axis=1)})
        out.attrs["total_draws"] = nat.sum(axis=0)
        return out

    def scenarios(self, plans, *a, **k) -> pd.DataFrame:
        """Compare future spend plans across all regions; see :meth:`papilon.MMM.scenarios`."""
        _no_extra(a, k)
        return MMM.scenarios(self, plans)

    def _not_yet(self, *a, **k):
        raise NotImplementedError("not available for geo models yet; fit a national papilon.MMM on summed data")

    backtest = self_check = value_of_lift_test = monitor = decompose = _not_yet
    response_curve = posterior_curves = _not_yet


def _no_extra(a, k) -> None:
    if a or k:
        raise TypeError(f"unexpected arguments: {list(a) + sorted(k)}")
