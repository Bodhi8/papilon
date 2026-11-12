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


CARRYOVER_KINDS = ("geometric", "delayed", "weibull")
SATURATION_KINDS = ("hill", "s_curve", "logistic")
_TV_SD = 0.25                                        # prior sd of time-varying effect coefficients
_WK_MU, _WK_SD = float(np.log(1.5)), 0.5             # Weibull carryover shape
_WL_MU, _WL_SD = float(np.log(2.0)), 0.6             # Weibull carryover scale, in periods
_RFS_MU, _RFS_SD = float(np.log(2.0)), 0.3           # frequency-response shape for reach and frequency channels


class Family:
    """The functional form of each channel: carryover, saturation and whether its effect drifts.

    ``carryover`` maps channels to "geometric" (default), "delayed" or "weibull";
    ``saturation`` maps channels to "hill" (default, concave), "s_curve" or
    "logistic"; ``time_varying`` lists channels whose effect may change over the
    window (creative wear-out, platform changes).
    """

    def __init__(self, channels, carryover=None, saturation=None, time_varying=None):
        carryover, saturation = dict(carryover or {}), dict(saturation or {})
        for what, given, kinds in (("carryover", carryover, CARRYOVER_KINDS), ("saturation", saturation, SATURATION_KINDS)):
            unknown = set(given) - set(channels)
            if unknown:
                raise ValueError(f"{what} channels not in the model: {sorted(unknown)}")
            bad = {c: k for c, k in given.items() if k not in kinds}
            if bad:
                raise ValueError(f"{what} kinds must be one of {kinds}; got {bad}")
        unknown = set(time_varying or []) - set(channels)
        if unknown:
            raise ValueError(f"time_varying channels not in the model: {sorted(unknown)}")
        self.carry = {c: carryover.get(c, "geometric") for c in channels}
        self.sat = {c: saturation.get(c, "hill") for c in channels}
        self.tv = [c for c in channels if c in set(time_varying or [])]

    @classmethod
    def from_spec(cls, channels, spec, base=None):
        """Build from ``{"delayed": [...], "weibull": [...], "s_curve": [...], "logistic": [...],
        "time_varying": [...], "carryover": {...}, "saturation": {...}}``; ``base`` fills what is left out."""
        carry = dict(base.carry) if base is not None and "keep" in spec else {}
        sat = dict(base.sat) if base is not None and "keep" in spec else {}
        tv = list(base.tv) if base is not None and "keep" in spec else []
        carry.update({c: "delayed" for c in spec.get("delayed", [])})
        carry.update({c: "weibull" for c in spec.get("weibull", [])})
        carry.update(spec.get("carryover", {}))
        sat.update({c: "s_curve" for c in spec.get("s_curve", [])})
        sat.update({c: "logistic" for c in spec.get("logistic", [])})
        sat.update(spec.get("saturation", {}))
        tv = list(dict.fromkeys(tv + list(spec.get("time_varying", []))))
        return cls(channels, carry, sat, tv)

    def of(self, kind):
        return [c for c, k in self.carry.items() if k == kind] or [c for c, k in self.sat.items() if k == kind]

    @property
    def delayed(self):
        return [c for c, k in self.carry.items() if k == "delayed"]

    @property
    def s_curve(self):
        return [c for c, k in self.sat.items() if k == "s_curve"]

    def name(self):
        parts = []
        for kind in ("delayed", "weibull"):
            cs = [c for c, k in self.carry.items() if k == kind]
            if cs:
                parts.append(f"{kind}:" + ",".join(cs))
        for kind in ("s_curve", "logistic"):
            cs = [c for c, k in self.sat.items() if k == kind]
            if cs:
                parts.append(f"{kind}:" + ",".join(cs))
        if self.tv:
            parts.append("time_varying:" + ",".join(self.tv))
        return "+".join(parts) or "all_geometric"

    def simple(self):
        return (all(k in ("geometric", "delayed") for k in self.carry.values())
                and all(k in ("hill", "s_curve") for k in self.sat.values()) and not self.tv)

    def to_dict(self):
        return {"carryover": dict(self.carry), "saturation": dict(self.sat), "time_varying": list(self.tv)}


@dataclass
class GateResult:
    """Outcome of the inference gate for one model family."""

    family: str
    delayed: list[str]
    log_evidence: float
    ess: float
    verdict: str
    history: list[tuple[int, float, str]] = field(default_factory=list)
    pareto_k: float = float("nan")  # PSIS tail shape; above 0.7 the importance weights are unreliable
    s_curve: list[str] = field(default_factory=list)
    carryover: dict = field(default_factory=dict)
    saturation: dict = field(default_factory=dict)
    time_varying: list[str] = field(default_factory=list)
    method: str = "importance"      # "importance" or "smc" (the fallback when importance sampling fails)
    modes: int = 1                  # distinct posterior modes found and used in the proposal

    @property
    def passed(self) -> bool:
        """True when the posterior draws can be relied on: the gate accepted, or the fallback sampler ran."""
        return self.verdict in ("ACCEPT", "SMC")


class MMM:
    """Marketing mix model fit by Papilon's gated importance sampling.

    Parameters
    ----------
    data : DataFrame with one row per period.
    target : column with the outcome (for example revenue).
    media : columns with spend per paid channel, in the same currency as ``target``
        if you want returns on ad spend.
    date_col : column with the period dates.
    controls : optional columns for non-media drivers (price, promotions, distribution).
        Their contributions are reported separately by :meth:`decompose`.
    yearly_seasonality : number of Fourier harmonics for yearly seasonality.
    trend : include a linear trend.
    baseline_drift : number of slow cosine terms added to the baseline, so demand can
        drift in ways a straight trend and fixed seasonality cannot (default 0).
    l_max : length of the carryover kernel, in periods.
    carryover : dict mapping channels to "geometric" (default), "delayed" (peaks after
        the period of spend) or "weibull" (a flexible lag shape with two parameters).
    saturation : dict mapping channels to "hill" (default, concave from the first
        dollar), "s_curve" (Hill with a fitted shape, prior median 1.5) or "logistic".
    delayed, s_curve : lists, shorthand for ``carryover`` and ``saturation`` (kept from 0.5).
    time_varying : channels whose effect may change across the window, for creative
        wear-out or platform changes. The effect is multiplied by
        ``exp(sum_k g_k * cos(pi * k * (t + 0.5) / T))`` with ``tv_basis`` terms and a
        prior that shrinks toward no change. Plans use the effect as of the last period.
    organic : columns for unpaid activity with carryover and saturation (email sends,
        organic social impressions). Modeled like media, reported in contributions, left
        out of returns and budget plans.
    reach_frequency : dict mapping a paid channel in ``media`` to its (reach column,
        frequency column). The channel's input becomes reach times a Hill curve in
        frequency, so the model can report the frequency that buys the most response per
        impression. Spend still defines its return.
    lift_tests : experiments to calibrate against, as a DataFrame or list of dicts with
        ``channel``, ``start``, ``end`` (dates, inclusive), ``lift`` (incremental
        outcome measured over the test plus ``readout`` periods, in target units),
        ``se`` (its standard error), and either ``factor`` (the test multiplied the
        channel's spend by this, 0 for a holdout) or ``spend_change`` (total change in
        spend over the test). Optional ``readout`` (periods after the test included in
        the measurement, default ``l_max - 1``). The model's predicted lift for the same
        change in the same periods enters the likelihood, so the test calibrates the
        curve at the spend level and season where it ran.
    alpha_prior : Beta(a, b) prior on carryover retention, one pair or a dict by channel.
    roi_priors : dict mapping a channel to (mean, sd) of its return on spend over the
        fitted window, from an experiment summarized as a return.
    roas_prior_scale : scale of the half-normal prior on each channel's return on spend
        at its typical spend level (default 2.0, prior median about 1.35), one number or a
        dict by channel. ``None`` uses a scale of 0.5 on the effect in units of mean sales.
    escalate : what to do when importance sampling fails its gate: "smc" (default) runs a
        tempered sequential Monte Carlo sampler from the same proposal; "none" keeps the
        failed draws and reports the verdict.
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
        carryover: dict[str, str] | None = None,
        saturation: dict[str, str] | None = None,
        time_varying: list[str] | None = None,
        tv_basis: int = 2,
        baseline_drift: int = 0,
        organic: list[str] | None = None,
        reach_frequency: dict[str, tuple[str, str]] | None = None,
        lift_tests=None,
        escalate: str = "smc",
    ) -> None:
        self._config = dict(target=target, media=list(media), date_col=date_col,
                            controls=list(controls or []), yearly_seasonality=yearly_seasonality,
                            trend=trend, l_max=l_max, period=period, alpha_prior=alpha_prior,
                            roi_priors=dict(roi_priors or {}), roas_prior_scale=roas_prior_scale,
                            tv_basis=tv_basis, baseline_drift=baseline_drift, organic=list(organic or []),
                            reach_frequency=dict(reach_frequency or {}), escalate=escalate,
                            lift_tests=None if lift_tests is None else pd.DataFrame(lift_tests).to_dict("records"))
        if escalate not in ("smc", "none"):
            raise ValueError('escalate must be "smc" or "none"')
        df = data.sort_values(date_col).reset_index(drop=True)
        self.data, self.target, self.media = df, target, list(media)
        self.organic = list(organic or [])
        self.channels = self.media + self.organic
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("a column appears twice among media and organic")
        self.date_col, self.controls = date_col, list(controls or [])
        self.l_max, self.period = int(l_max), float(period)
        self.n_tv, self.escalate = int(tv_basis), escalate
        self.reach_frequency = dict(reach_frequency or {})
        unknown = set(self.reach_frequency) - set(self.media)
        if unknown:
            raise ValueError(f"reach_frequency channels must be paid media: {sorted(unknown)}")
        carry = dict(carryover or {}) | {c: "delayed" for c in (delayed or [])}
        sat = dict(saturation or {}) | {c: "s_curve" for c in (s_curve or [])}
        self._fam = Family(self.channels, carry, sat, time_varying)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        if isinstance(alpha_prior, dict):
            self.alpha_prior = {c: tuple(alpha_prior.get(c, (2.0, 2.0))) for c in self.channels}
        else:
            self.alpha_prior = {c: tuple(alpha_prior) for c in self.channels}
        self.roi_priors = dict(roi_priors or {})
        unknown = set(self.roi_priors) - set(self.media)
        if unknown:
            raise ValueError(f"roi_priors channels not in media: {sorted(unknown)}")
        missing = [c for c in [target, date_col] + self.channels + self.controls
                   + [x for pair in self.reach_frequency.values() for x in pair] if c not in df.columns]
        if missing:
            raise ValueError(f"columns not in data: {missing}")

        self.y_raw = df[target].to_numpy(dtype=float)
        self.spend_raw = df[self.media].to_numpy(dtype=float)
        if np.any(self.spend_raw < 0) or np.any(df[self.organic].to_numpy(dtype=float) < 0):
            raise ValueError("media spend and organic activity must be non-negative")
        if not np.all(np.isfinite(self.y_raw)) or not np.all(np.isfinite(self.spend_raw)):
            raise ValueError("target and media spend must not contain missing values")
        self.y_scale = float(np.mean(self.y_raw))
        self.spend_scale = self.spend_raw.mean(axis=0)
        self.spend_scale[self.spend_scale == 0] = 1.0
        org = df[self.organic].to_numpy(dtype=float) if self.organic else np.zeros((len(df), 0))
        self.organic_scale = org.mean(axis=0) if self.organic else np.zeros(0)
        if self.organic:
            self.organic_scale[self.organic_scale == 0] = 1.0
        self._rf_stats = {}
        for c, (rcol, fcol) in self.reach_frequency.items():
            r, f = df[rcol].to_numpy(dtype=float), df[fcol].to_numpy(dtype=float)
            if np.any(r < 0) or np.any(f < 0):
                raise ValueError(f"reach and frequency for {c} must be non-negative")
            self._rf_stats[c] = (float(r.mean() or 1.0), float(np.median(f[f > 0]) if np.any(f > 0) else 1.0))
        self.y = self.y_raw / self.y_scale
        self.spend = self.spend_raw / self.spend_scale
        self._spend_tot = self.spend_raw.sum(axis=0)
        self._T_fit = len(df)
        self._src = self._sources(df)
        self._W = np.stack([s[1] if s[0] == "w" else np.zeros((len(df), self.l_max)) for s in self._src])
        # Prior scale for each channel's effect b (in units of mean sales). Contribution at
        # typical spend is about b * y_scale * 0.5 (half-saturation prior median 1, kernel
        # normalized to sum to 1), so ROAS ~ b * y_scale * 0.5 / spend_scale.
        rs = ({c: roas_prior_scale.get(c, 2.0) for c in self.media} if isinstance(roas_prior_scale, dict)
              else {c: roas_prior_scale for c in self.media})
        self._b_scale = {c: (0.5 if rs.get(c) is None else
                             float(rs[c]) * self.spend_scale[j] / (0.5 * self.y_scale))
                         for j, c in enumerate(self.media)} | {c: 0.5 for c in self.organic}
        self._roi_idx = [(self.media.index(c), float(m), float(sd)) for c, (m, sd) in
                         (roi_priors or {}).items() if c in self.media]

        self._trend, self._harmonics, self._drift = bool(trend), int(yearly_seasonality), int(baseline_drift)
        self._ctrl_stats = []
        for c in self.controls:
            v = df[c].to_numpy(dtype=float)
            if not np.all(np.isfinite(v)):
                raise ValueError(f"control {c} contains missing values")
            self._ctrl_stats.append((float(v.mean()), float(v.std() or 1.0)))
        n = len(df)
        self._base_x = self._design(df, 0)
        sds = ([100.0] + ([10.0] if trend else []) + [10.0] * (2 * yearly_seasonality)
               + [10.0] * self._drift + [20.0] * len(self.controls))
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
        self._lift = self._parse_lift_tests(lift_tests)

        self.gate_results: dict[str, GateResult] = {}
        self.family_probabilities: dict[str, float] = {}
        self._draws: dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------- data in
    @property
    def delayed(self) -> list[str]:
        return self._fam.delayed

    @property
    def s_curve(self) -> list[str]:
        return self._fam.s_curve

    @property
    def family(self) -> Family:
        return self._fam

    def _check_channels(self, names: list[str], what: str = "delayed") -> None:
        unknown = set(names) - set(self.channels)
        if unknown:
            raise ValueError(f"{what} channels not in media: {sorted(unknown)}")

    _check_delayed = _check_channels

    def _lagged(self, x: np.ndarray) -> np.ndarray:
        """(periods,) -> (periods, l_max) with column l holding x[t - l]."""
        T = len(x)
        out = np.zeros((T, self.l_max))
        for l in range(min(self.l_max, T)):
            out[l:, l] = x[: T - l]
        return out

    def _sources(self, df: pd.DataFrame) -> list:
        """Per-channel inputs, scaled with this model's statistics: ("w", lagged input) or
        ("rf", lagged reach, lagged frequency, reach, frequency)."""
        out = []
        for j, c in enumerate(self.channels):
            if c in self.reach_frequency:
                rcol, fcol = self.reach_frequency[c]
                r = df[rcol].to_numpy(dtype=float) / self._rf_stats[c][0]
                f = df[fcol].to_numpy(dtype=float)
                out.append(("rf", self._lagged(r), self._lagged(f), r, f))
            else:
                scale = self.spend_scale[j] if j < len(self.media) else self.organic_scale[j - len(self.media)]
                out.append(("w", self._lagged(df[c].to_numpy(dtype=float) / scale)))
        return out

    def _windows(self, spend: np.ndarray) -> np.ndarray:
        """Lagged spend, shape (channels, periods, l_max): W[j, t, l] = spend[t - l, j]."""
        return np.stack([self._lagged(spend[:, j]) for j in range(spend.shape[1])])

    def _tv_rows(self, rows: np.ndarray) -> np.ndarray:
        """Time-varying basis at row indices; rows past the fitted window hold the last value."""
        idx = np.minimum(np.asarray(rows), self._T_fit - 1)
        k = np.arange(1, self.n_tv + 1)
        return np.cos(np.pi * k[None, :] * (idx[:, None] + 0.5) / self._T_fit)

    def _design(self, df: pd.DataFrame, offset: int) -> np.ndarray:
        """Baseline design matrix for rows of ``df`` that start ``offset`` periods into the fit window."""
        n = len(df)
        idx = offset + np.arange(n)
        t = idx / self.period
        cols = [np.ones(n)]
        if self._trend:
            cols.append(t)
        for k in range(1, self._harmonics + 1):
            cols += [np.sin(2 * np.pi * k * t), np.cos(2 * np.pi * k * t)]
        held = np.minimum(idx, getattr(self, "_T_fit", n) - 1)
        for k in range(1, self._drift + 1):
            cols.append(np.cos(np.pi * k * (held + 0.5) / getattr(self, "_T_fit", n)))
        for c, (mu, sd) in zip(self.controls, self._ctrl_stats):
            v = df[c].to_numpy(dtype=float) if c in df.columns else np.full(n, mu)
            cols.append((v - mu) / sd)
        return np.column_stack(cols)

    def _parse_lift_tests(self, lift_tests) -> list:
        if lift_tests is None:
            return []
        tests = pd.DataFrame(lift_tests)
        if tests.empty:
            return []
        need = {"channel", "start", "end", "lift", "se"}
        if not need <= set(tests.columns):
            raise ValueError(f"lift_tests needs columns {sorted(need)} plus factor or spend_change")
        dates = pd.to_datetime(self.data[self.date_col])
        out = []
        for _, t in tests.iterrows():
            c = t["channel"]
            if c not in self.media:
                raise ValueError(f"lift test channel {c!r} is not in media")
            j = self.media.index(c)
            inside = np.where((dates >= pd.Timestamp(t["start"])) & (dates <= pd.Timestamp(t["end"])))[0]
            if len(inside) == 0:
                raise ValueError(f"lift test on {c} from {t['start']} to {t['end']} has no periods in the data")
            r0, r1 = int(inside[0]), int(inside[-1]) + 1
            readout = int(t["readout"]) if "readout" in t and pd.notna(t.get("readout")) else self.l_max - 1
            rows = np.arange(r0, min(self._T_fit, r1 + readout))
            col = self.reach_frequency[c][0] if c in self.reach_frequency else c
            window = self.data[col].to_numpy(dtype=float)[r0:r1]
            if "factor" in t and pd.notna(t.get("factor")):
                factor = float(t["factor"])
            elif "spend_change" in t and pd.notna(t.get("spend_change")):
                base = self.data[c].to_numpy(dtype=float)[r0:r1].sum()
                if base <= 0:
                    raise ValueError(f"lift test on {c}: no spend in the test window to scale")
                factor = 1 + float(t["spend_change"]) / base
            else:
                raise ValueError("each lift test needs factor or spend_change")
            if factor < 0:
                raise ValueError("a lift test cannot reduce spend below zero")
            cf = self.data.copy()
            v = cf[col].to_numpy(dtype=float).copy()
            v[r0:r1] = window * factor
            cf[col] = v
            src_cf = self._sources(cf)[j]
            out.append({"j": j, "rows": rows, "src": self._src[j], "src_cf": src_cf,
                        "lift": float(t["lift"]) / self.y_scale, "se": float(t["se"]) / self.y_scale,
                        "channel": c, "factor": factor})
        return out

    # ------------------------------------------------------------------ model
    def _layout(self, fam: Family) -> list[list[str]]:
        lay = []
        for c in self.channels:
            k = fam.carry[c]
            names = ["za"] if k == "geometric" else ["za", "zu"] if k == "delayed" else ["zwk", "zwl"]
            names += ["zh", "zb"]
            if fam.sat[c] == "s_curve":
                names.append("zs")
            if c in self.reach_frequency:
                names += ["zrk", "zrs"]
            if c in fam.tv:
                names += ["zg"] * self.n_tv
            lay.append(names)
        return lay

    def _dim(self, fam, shaped=None) -> int:
        if not isinstance(fam, Family):          # 0.5 signature: _dim(delayed, s_curve)
            fam = Family(self.channels, {c: "delayed" for c in fam}, {c: "s_curve" for c in (shaped or [])})
        return sum(len(x) for x in self._layout(fam))

    def _as_family(self, fam, shaped=None) -> Family:
        if isinstance(fam, Family):
            return fam
        return Family(self.channels, {c: "delayed" for c in (fam or [])}, {c: "s_curve" for c in (shaped or [])})

    def _init_point(self, fam: Family) -> np.ndarray:
        z = []
        for c, names in zip(self.channels, self._layout(fam)):
            for nm in names:
                z.append({"za": 0.0, "zu": 0.0, "zwk": _WK_MU, "zwl": _WL_MU, "zh": 0.0,
                          "zb": np.log(0.4 * self._b_scale[c]), "zs": _SHAPE_MU, "zg": 0.0, "zrs": _RFS_MU,
                          "zt": np.log(0.3),
                          "zrk": np.log(self._rf_stats[c][1]) if c in self._rf_stats else 0.0}[nm])
        return np.array(z)

    def _unpack_batch(self, Z: np.ndarray, fam, shaped=None):
        """Unconstrained draws (n, d) -> parameter arrays (n, channels), plus log prior with Jacobian."""
        fam = self._as_family(fam, shaped)
        Z = np.atleast_2d(Z)
        n, k = len(Z), len(self.channels)
        nan = lambda: np.full((n, k), np.nan)
        P = {"alpha": nan(), "peak": nan(), "wk": nan(), "wl": nan(), "half_sat": np.empty((n, k)),
             "beta": np.empty((n, k)), "shape": np.ones((n, k)), "rf_k": nan(), "rf_s": nan(),
             "tv": np.zeros((n, k, self.n_tv))}
        lp = np.zeros(n)
        i, max_peak = 0, self.l_max - 2
        norm = lambda z, mu, sd: -np.log(sd) - _HALF_LOG_2PI - (z - mu) ** 2 / (2 * sd * sd)
        with np.errstate(all="ignore"):
            for j, (c, names) in enumerate(zip(self.channels, self._layout(fam))):
                g = 0
                for nm in names:
                    z = Z[:, i]; i += 1
                    if nm == "za":
                        a = expit(z); P["alpha"][:, j] = a
                        pa, pb = self.alpha_prior[c]
                        lp += _beta_lpdf(a, pa, pb) + np.log(a) + np.log1p(-a)
                    elif nm == "zu":
                        u = expit(z); P["peak"][:, j] = max_peak * u
                        lp += _beta_lpdf(u, 2.0, 2.0) + np.log(u) + np.log1p(-u)
                    elif nm == "zwk":
                        P["wk"][:, j] = np.exp(z); lp += norm(z, _WK_MU, _WK_SD)
                    elif nm == "zwl":
                        P["wl"][:, j] = np.exp(z); lp += norm(z, _WL_MU, _WL_SD)
                    elif nm == "zh":
                        hs = np.exp(z); P["half_sat"][:, j] = hs
                        lp += _lognorm_lpdf(hs, 0.5) + z
                    elif nm == "zb":
                        b = np.exp(z); P["beta"][:, j] = b
                        lp += _halfnorm_lpdf(b, self._b_scale[c]) + z
                    elif nm == "zs":
                        P["shape"][:, j] = np.exp(z); lp += norm(z, _SHAPE_MU, _SHAPE_SD)
                    elif nm == "zrk":
                        P["rf_k"][:, j] = np.exp(z); lp += norm(z, np.log(self._rf_stats[c][1]), 0.5)
                    elif nm == "zrs":
                        P["rf_s"][:, j] = np.exp(z); lp += norm(z, _RFS_MU, _RFS_SD)
                    elif nm == "zg":
                        P["tv"][:, j, g] = z; g += 1; lp += norm(z, 0.0, _TV_SD)
                    elif nm == "zt":                        # geo models: spread of effects across geos
                        P.setdefault("tau", np.full((n, k), np.nan))[:, j] = np.exp(z)
                        lp += norm(z, np.log(0.3), 0.5)
        return P, lp

    def _kernel(self, P: dict, j: int, kind: str) -> np.ndarray:
        """Carryover weights per draw, shape (draws, l_max), each row summing to one."""
        lags = self._lags[None, :]
        with np.errstate(all="ignore"):
            if kind == "weibull":
                k, lam = P["wk"][:, j][:, None], P["wl"][:, j][:, None]
                edges = np.arange(self.l_max + 1)[None, :]
                kern = np.diff(1.0 - np.exp(-(edges / lam) ** k), axis=1)
            else:                             # geometric, or delayed where a peak is set
                a, pk = P["alpha"][:, j][:, None], P["peak"][:, j][:, None]
                kern = np.where(np.isnan(pk), a ** lags, a ** ((lags - np.nan_to_num(pk)) ** 2))
            kern = np.nan_to_num(kern, nan=0.0) + 1e-12
        return kern / kern.sum(axis=1, keepdims=True)

    def _adstock(self, P, j, kind, src, rows, scale=1.0):
        """Carried-over input for channel j at ``rows``, shape (rows, draws)."""
        kern = self._kernel(P, j, kind)
        if src[0] == "rf":
            _, R, F, r, f = src
            kap, s = P["rf_k"][:, j], P["rf_s"][:, j]
            with np.errstate(all="ignore"):
                fs = F[rows][None] ** s[:, None, None]
                G = fs / (fs + kap[:, None, None] ** s[:, None, None])
                g_all = f[None] ** s[:, None] / (f[None] ** s[:, None] + kap[:, None] ** s[:, None])
                norm = np.mean(r[None] * g_all, axis=1)          # mean input per draw = 1 after scaling
                ad = np.einsum("ntl,nl->tn", R[rows][None] * G, kern) / np.maximum(norm, 1e-12)[None, :]
        else:
            ad = src[1][rows] @ kern.T
        return ad * scale

    def _saturate(self, ad, P, j, kind):
        hs = P["half_sat"][:, j]
        with np.errstate(all="ignore"):
            if kind == "logistic":
                e = np.exp(-ad / hs)
                return (1 - e) / (1 + e)
            if kind == "hill":
                return ad / (ad + hs)
            s = P["shape"][:, j]
            a_s = ad ** s
            return a_s / (a_s + hs ** s)

    def _chan_eff(self, P, fam, j, src, rows, scale=1.0):
        """Contribution of channel j at ``rows`` in model units, shape (rows, draws)."""
        c = self.channels[j]
        out = P["beta"][:, j] * self._saturate(self._adstock(P, j, fam.carry[c], src, rows, scale), P, j, fam.sat[c])
        if c in fam.tv:
            out = out * np.exp(self._tv_rows(rows) @ P["tv"][:, j, :].T)
        return out

    def _effects(self, P: dict, W=None, scale: float = 1.0, fam: Family | None = None, src=None) -> np.ndarray:
        """Channel contributions in model units, shape (periods, draws, channels).

        ``scale`` multiplies paid-media input in every period (adstock is linear in input).
        ``src`` gives inputs from :meth:`_sources` (default: the fitted data); ``W`` is the
        0.5 form, lagged spend of shape (channels, periods, l_max).
        """
        fam = self._fam if fam is None else fam
        if src is None:
            src = self._src if W is None else [("w", W[j]) for j in range(W.shape[0])] + self._src[W.shape[0]:]
        T = src[0][1].shape[0]
        rows = np.arange(T)
        out = np.empty((T, len(P["beta"]), len(self.channels)))
        for j in range(len(self.channels)):
            out[:, :, j] = self._chan_eff(P, fam, j, src[j], rows, scale if j < len(self.media) else 1.0)
        return out

    def _marginal_parts(self, resid: np.ndarray):
        """Posterior mean of the baseline coefficients and the noise scale term, per draw."""
        mn = self._proj @ resid                                        # (p, n)
        bn = self._b0 + 0.5 * (np.sum(resid * resid, axis=0) - np.sum(mn * (self._vn_inv @ mn), axis=0))
        return mn, bn

    def _calibration_lp(self, P, fam, eff):
        lp = 0.0
        with np.errstate(all="ignore"):
            for j, m, sd in self._roi_idx:
                roas = eff[:, :, j].sum(axis=0) * self.y_scale / self._spend_tot[j]
                lp = lp + (-0.5 * ((roas - m) / sd) ** 2 - np.log(sd) - _HALF_LOG_2PI)
            for t in self._lift:
                pred = (self._chan_eff(P, fam, t["j"], t["src_cf"], t["rows"])
                        - self._chan_eff(P, fam, t["j"], t["src"], t["rows"])).sum(axis=0)
                lp = lp + (-0.5 * ((pred - t["lift"]) / t["se"]) ** 2 - np.log(t["se"]) - _HALF_LOG_2PI)
        return lp

    def _log_post_batch(self, Z: np.ndarray, fam, shaped=None) -> np.ndarray:
        fam = self._as_family(fam, shaped)
        P, lp = self._unpack_batch(Z, fam)
        eff = self._effects(P, fam=fam)
        with np.errstate(all="ignore"):
            lp = lp + self._calibration_lp(P, fam, eff)
            _, bn = self._marginal_parts(self.y[:, None] - eff.sum(axis=2))
            out = self._lm_const - self._an * np.log(bn) + lp
        return np.where(np.isfinite(out), out, -np.inf)

    def _fast_ok(self, fam: Family) -> bool:
        return fam.simple() and not self._lift and not self.reach_frequency

    def _log_post(self, z: np.ndarray, fam, shaped=None) -> float:
        """One draw. Simple families take a fast path; the rest use the batch code."""
        fam = self._as_family(fam, shaped)
        if not self._fast_ok(fam):
            return float(self._log_post_batch(z[None, :], fam)[0])
        i, lp, max_peak, lags = 0, 0.0, self.l_max - 2, self._lags
        media = np.zeros(len(self.y))
        roas = {}
        with np.errstate(all="ignore"):
            for j, c in enumerate(self.channels):
                za, zh, zb = z[i], None, None
                i += 1
                a = expit(za)
                pa, pb = self.alpha_prior[c]
                lp += _beta_lpdf(a, pa, pb) + np.log(a) + np.log1p(-a)
                if fam.carry[c] == "delayed":
                    u = expit(z[i]); i += 1
                    lp += _beta_lpdf(u, 2.0, 2.0) + np.log(u) + np.log1p(-u)
                    kern = a ** ((lags - max_peak * u) ** 2)
                else:
                    kern = a ** lags
                zh, zb = z[i], z[i + 1]; i += 2
                hs, b = np.exp(zh), np.exp(zb)
                lp += _lognorm_lpdf(hs, 0.5) + zh + _halfnorm_lpdf(b, self._b_scale[c]) + zb
                ad = self._src[j][1] @ (kern / kern.sum())
                if fam.sat[c] == "s_curve":
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
    def _hessian_cov(self, x, fam):
        d, h = len(x), 1e-3
        pts, where = [], []
        for a in range(d):
            for b in range(a, d):
                ea = np.zeros(d); ea[a] = h
                eb = np.zeros(d); eb[b] = h
                pts += [x + ea + eb, x + ea - eb, x - ea + eb, x - ea - eb]
                where.append((a, b))
        v = -self._log_post_batch(np.array(pts), fam).reshape(-1, 4)
        hess = np.zeros((d, d))
        for (a, b), (pp_, pm, mp, mm) in zip(where, v):
            hess[a, b] = hess[b, a] = (pp_ - pm - mp + mm) / (4 * h * h)
        hess = np.nan_to_num(hess, nan=0.0, posinf=0.0, neginf=0.0)
        vals, vecs = np.linalg.eigh(0.5 * (hess + hess.T))
        return vecs @ np.diag(1 / np.maximum(vals, 0.5)) @ vecs.T

    def _modes(self, fam: Family, n_starts: int) -> list:
        """Posterior modes from several starts, each with a Laplace covariance and its log posterior.

        Distinct modes within 10 log units of the best are all kept, so a skewed or
        multimodal posterior gets a mixture proposal instead of a single Gaussian.
        """
        f = lambda z: -self._log_post(z, fam)
        z_init = self._init_point(fam)
        found = []
        for _ in range(n_starts):
            z0 = z_init + self.rng.normal(0, 0.3, len(z_init))
            r = optimize.minimize(f, z0, method="Nelder-Mead",
                                  options={"maxiter": 20000, "maxfev": 20000, "xatol": 1e-4, "fatol": 1e-6})
            r = optimize.minimize(f, r.x, method="BFGS")
            if np.isfinite(r.fun):
                found.append((r.x, float(-r.fun)))
        if not found:
            raise RuntimeError("the posterior could not be evaluated at any starting point; check the data")
        found.sort(key=lambda m: -m[1])
        best = found[0][1]
        modes = []
        for x, lpv in found:
            if lpv < best - 10:
                continue
            if any(np.max(np.abs(x - m[0])) < 0.5 for m in modes):
                continue
            modes.append((x, self._hessian_cov(x, fam), lpv))
        return modes

    def _laplace(self, fam, shaped=None, n_starts=3):
        """The best mode and its Laplace covariance (kept for compatibility)."""
        x, cov, _ = self._modes(self._as_family(fam, shaped), n_starts)[0]
        return x, cov

    @staticmethod
    def _mixture_weights(modes):
        lw = np.array([lpv + 0.5 * np.linalg.slogdet(cov)[1] for _, cov, lpv in modes])
        w = np.exp(lw - lw.max())
        return w / w.sum()

    def _mixture(self, comps):
        """comps: list of (weight, mean, cov). Returns sampler and log density for a mixture of t(5)."""
        dists = [(w, stats.multivariate_t(loc=m, shape=c, df=5, allow_singular=True)) for w, m, c in comps]

        def rvs(n):
            counts = self.rng.multinomial(n, [w for w, _ in dists])
            parts = [np.atleast_2d(d.rvs(k, random_state=self.rng)) for (w, d), k in zip(dists, counts) if k > 0]
            z = np.vstack(parts)
            return z[self.rng.permutation(len(z))]

        def logpdf(z):
            return logsumexp(np.stack([np.log(w) + np.atleast_1d(d.logpdf(z)) for w, d in dists]), axis=0)
        return rvs, logpdf

    def _importance(self, fam, comps, n_draws):
        rvs, logq = self._mixture(comps)
        z = rvs(n_draws)
        log_w = self._log_post_batch(z, fam) - logq(z)
        log_ev = float(logsumexp(log_w) - np.log(n_draws))
        w = np.exp(log_w - log_w.max()); w /= w.sum()
        self._last_k = _pareto_k(log_w)
        return z, w, log_ev, float(1 / np.sum(w ** 2) / n_draws)

    def _refine(self, comps, z, w):
        """Moment-match each mixture component to the weighted draws it is responsible for."""
        if len(comps) == 1:
            m = np.average(z, axis=0, weights=w)
            return [(1.0, m, np.cov(z.T, aweights=w) + 1e-6 * np.eye(z.shape[1]))]
        dens = np.stack([np.log(cw) + stats.multivariate_t(loc=m, shape=c, df=5, allow_singular=True).logpdf(z)
                         for cw, m, c in comps])
        resp = np.exp(dens - logsumexp(dens, axis=0))
        out = []
        for k in range(len(comps)):
            wk = w * resp[k]
            if wk.sum() < 1e-8 or (wk > 0).sum() < z.shape[1] + 2:
                out.append(comps[k]); continue
            m = np.average(z, axis=0, weights=wk)
            out.append((wk.sum(), m, np.cov(z.T, aweights=wk) + 1e-6 * np.eye(z.shape[1])))
        tot = sum(c[0] for c in out)
        return [(c[0] / tot, c[1], c[2]) for c in out]

    def _smc(self, fam, comps, n_particles=2000, n_moves=5, target_ess=0.5):
        """Tempered sequential Monte Carlo from the proposal q to the posterior p.

        Intermediate targets are q^(1-b) p^b for b rising from 0 to 1, chosen so each
        step keeps half the particles' effective size; particles are resampled and moved
        with random-walk Metropolis under each target. Handles skewed and multimodal
        posteriors that defeat importance sampling, at a few times the cost. Returns the
        particles, equal weights and an estimate of the log evidence.
        """
        rvs, logq = self._mixture(comps)
        z = rvs(n_particles)
        lq, lp = logq(z), self._log_post_batch(z, fam)
        b, log_ev, N, scale, steps = 0.0, 0.0, len(z), 0.5, 0
        while b < 1.0 and steps < 200:
            steps += 1
            delta = np.where(np.isfinite(lp), lp - lq, -np.inf)

            def ess(nb):
                lw = (nb - b) * delta
                w = np.exp(lw - np.max(lw))
                return w.sum() ** 2 / (w ** 2).sum() / N
            if ess(1.0) >= target_ess:
                nb = 1.0
            else:
                lo, hi = b, 1.0
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if ess(mid) >= target_ess else (lo, mid)
                nb = max(lo, b + 1e-6)
            lw = (nb - b) * delta
            log_ev += float(logsumexp(lw) - np.log(N))
            w = np.exp(lw - lw.max()); w /= w.sum()
            pos = (self.rng.random() + np.arange(N)) / N
            idx = np.minimum(np.searchsorted(np.cumsum(w), pos), N - 1)
            z, lq, lp, b = z[idx], lq[idx], lp[idx], nb
            cov = np.cov(z.T) + 1e-8 * np.eye(z.shape[1])
            chol = np.linalg.cholesky(cov)
            for _ in range(n_moves):
                prop = z + scale * (self.rng.standard_normal(z.shape) @ chol.T)
                lqp, lpp = logq(prop), self._log_post_batch(prop, fam)
                cur = (1 - b) * lq + b * lp
                new = (1 - b) * lqp + b * lpp
                acc = np.log(self.rng.random(N)) < np.where(np.isfinite(new), new - cur, -np.inf)
                z[acc], lq[acc], lp[acc] = prop[acc], lqp[acc], lpp[acc]
                rate = acc.mean()
                scale *= 1.3 if rate > 0.4 else 0.7 if rate < 0.15 else 1.0
        return z, np.full(N, 1.0 / N), log_ev

    def _gate(self, name, fam, n_draws, n_starts, accept, refine, max_refine, shaped=None):
        fam = self._as_family(fam, shaped)
        modes = self._modes(fam, n_starts)
        weights = self._mixture_weights(modes)
        comps = [(w, x, 1.5 * cov) for w, (x, cov, _) in zip(weights, modes)]
        history = []
        for step in range(max_refine + 1):
            z, w, log_ev, ess = self._importance(fam, comps, n_draws)
            verdict = "ACCEPT" if ess > accept else "REFINE" if ess > refine else "ESCALATE"
            history.append((step, round(ess, 3), verdict))
            if verdict == "ACCEPT":
                break
            comps = self._refine(comps, z, w)
        method, k_hat = "importance", float(getattr(self, "_last_k", np.nan))
        if verdict != "ACCEPT" and self.escalate == "smc":
            z, w, log_ev = self._smc(fam, comps, n_particles=max(n_draws // 2, 1000))
            verdict, ess, method, k_hat = "SMC", 1.0, "smc", float("nan")
            history.append((len(history), 1.0, "SMC"))
        res = GateResult(name, fam.delayed, log_ev, ess, verdict, history, pareto_k=k_hat, s_curve=fam.s_curve,
                         carryover=dict(fam.carry), saturation=dict(fam.sat), time_varying=list(fam.tv),
                         method=method, modes=len(modes))
        return res, z, w, fam

    def _store_draws(self, fam, z, w, n_keep=2000):
        idx = self.rng.choice(len(w), n_keep, p=w)
        self._draws, _ = self._unpack_batch(z[idx], fam)
        self._fam = fam
        self._roi_cache = self._roas_cache = None

    def _family_name(self) -> str:
        return self._fam.name()

    def fit(self, n_draws: int = 4000, n_starts: int = 3, accept: float = 0.10,
            refine: float = 0.01, max_refine: int = 3) -> "MMM":
        """Fit the model with the family set in the constructor."""
        name = self._family_name()
        res, z, w, fam = self._gate(name, self._fam, n_draws, n_starts, accept, refine, max_refine)
        self.gate_results = {name: res}
        self.family_probabilities = {name: 1.0}
        self._store_draws(fam, z, w)
        return self

    def compare_models(self, candidates: dict[str, dict], n_draws: int = 4000,
                       n_starts: int = 3, accept: float = 0.10, refine: float = 0.01,
                       max_refine: int = 3) -> pd.DataFrame:
        """Fit each model family, compare them by marginal likelihood, keep the winner.

        ``candidates`` maps a family name to a spec with any of the keys ``delayed``,
        ``weibull``, ``s_curve``, ``logistic``, ``time_varying`` (lists of channels),
        ``carryover`` and ``saturation`` (dicts), for example::

            {"concave": {"delayed": ["online_video"]},
             "video_s_curve": {"delayed": ["online_video"], "s_curve": ["online_video"]},
             "video_weibull": {"weibull": ["online_video"]},
             "social_wears_out": {"delayed": ["online_video"], "time_varying": ["paid_social"]}}

        Channels a spec leaves out are geometric, concave and constant. Families get
        equal prior odds. A wrong curve can fit history and still extrapolate badly,
        which is what a reallocation asks of it; this lets the data vote on the form
        before a plan is built on it.
        """
        fitted = {}
        for name, spec in candidates.items():
            fam = Family.from_spec(self.channels, spec)
            fitted[name] = self._gate(name, fam, n_draws, n_starts, accept, refine, max_refine)
        names = list(fitted)
        ev = np.array([fitted[k][0].log_evidence for k in names])
        probs = np.exp(ev - logsumexp(ev))
        self.gate_results = {k: fitted[k][0] for k in names}
        self.family_probabilities = dict(zip(names, map(float, probs)))
        win = names[int(np.argmax(probs))]
        res, z, w, fam = fitted[win]
        self._store_draws(fam, z, w)
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
        Saturation and time-varying settings from the constructor are kept in every
        family; use :meth:`compare_models` to compare those too.
        """
        base = self._fam
        specs = {}
        for k, v in candidates.items():
            carry = {c: ("weibull" if base.carry[c] == "weibull" else "geometric") for c in self.channels}
            carry.update({c: "delayed" for c in v})
            specs[k] = {"carryover": carry, "saturation": dict(base.sat), "time_varying": list(base.tv)}
        return self.compare_models(specs, n_draws, n_starts, accept, refine, max_refine)

    # ---------------------------------------------------------------- results
    def _require_fit(self) -> None:
        if self._draws is None:
            raise RuntimeError("call fit() or compare_carryover() first")

    def _mult_now(self, P, j, fam=None):
        """Effect multiplier of channel j in the last fitted period, per draw (1 unless time-varying)."""
        fam = self._fam if fam is None else fam
        if self.channels[j] not in fam.tv:
            return np.ones(len(P["beta"]))
        return np.exp(P["tv"][:, j, :] @ self._tv_rows(np.array([self._T_fit - 1]))[0])

    def parameters(self) -> pd.DataFrame:
        """Posterior means of each channel's response parameters (half-saturation in raw input units)."""
        self._require_fit()
        d, fam = self._draws, self._fam
        scale = np.concatenate([self.spend_scale, self.organic_scale])
        rows = []
        for j, c in enumerate(self.channels):
            mult = self._mult_now(d, j)
            row = {"channel": c, "paid": j < len(self.media), "carryover": fam.carry[c], "saturation": fam.sat[c],
                   "alpha": np.nanmean(d["alpha"][:, j]) if fam.carry[c] != "weibull" else np.nan,
                   "peak_lag": np.nanmean(d["peak"][:, j]) if fam.carry[c] == "delayed" else np.nan,
                   "weibull_shape": np.nanmean(d["wk"][:, j]) if fam.carry[c] == "weibull" else np.nan,
                   "weibull_scale": np.nanmean(d["wl"][:, j]) if fam.carry[c] == "weibull" else np.nan,
                   "half_saturation": (d["half_sat"][:, j].mean() * scale[j]
                                       if c not in self.reach_frequency else d["half_sat"][:, j].mean()),
                   "shape": d["shape"][:, j].mean(),
                   "max_weekly_effect": float((d["beta"][:, j] * mult).mean() * self.y_scale),
                   "effect_change": float(mult.mean() - 1) if c in fam.tv else 0.0}
            if c in self.reach_frequency:
                k, s = d["rf_k"][:, j], d["rf_s"][:, j]
                row["optimal_frequency"] = float(np.median(k * np.maximum(s - 1, 1e-6) ** (1 / s)))
            rows.append(row)
        out = pd.DataFrame(rows).set_index("channel")
        if self.media == self.channels and not self.reach_frequency and fam.simple():
            out = out[["alpha", "half_saturation", "max_weekly_effect", "peak_lag", "shape"]]
        return out

    def roas_draws(self) -> dict:
        """Posterior draws, shape (draws, channels), of ROAS, marginal ROAS and share of sales (paid media)."""
        self._require_fit()
        if getattr(self, "_roas_cache", None) is None:
            km = len(self.media)
            m = self._effects(self._draws)[:, :, :km].sum(axis=0) * self.y_scale
            bumped = self._effects(self._draws, scale=1.01)[:, :, :km].sum(axis=0) * self.y_scale
            self._roas_cache = {"roas": m / self._spend_tot, "mroas": (bumped - m) / (0.01 * self._spend_tot),
                                "share": m / self.y_raw.sum()}
        return self._roas_cache

    def get_roi(self) -> pd.DataFrame:
        """Return on spend per paid channel: posterior mean, 90% interval and marginal return.

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
        """Posterior-mean contribution of each channel (paid and organic) per period, in target units."""
        self._require_fit()
        out = pd.DataFrame(self._effects(self._draws).mean(axis=1) * self.y_scale, columns=self.channels)
        out.insert(0, self.date_col, self.data[self.date_col].to_numpy())
        out["baseline_and_controls"] = self.y_raw - out[self.channels].sum(axis=1)
        return out

    def decompose(self) -> pd.DataFrame:
        """Fitted sales split into baseline, each control, and each channel, per period (posterior means).

        ``baseline`` is the intercept, trend, seasonality and drift; each control column
        is its own effect; ``residual`` is what the model does not explain.
        """
        self._require_fit()
        eff = self._effects(self._draws)
        mn, _ = self._marginal_parts(self.y[:, None] - eff.sum(axis=2))
        coef = mn.mean(axis=1)
        X = self._base_x
        nb = X.shape[1] - len(self.controls)
        out = pd.DataFrame({self.date_col: self.data[self.date_col].to_numpy(),
                            "actual": self.y_raw, "baseline": X[:, :nb] @ coef[:nb] * self.y_scale})
        for i, c in enumerate(self.controls):
            out[c] = X[:, nb + i] * coef[nb + i] * self.y_scale
        for j, c in enumerate(self.channels):
            out[c] = eff[:, :, j].mean(axis=1) * self.y_scale
        parts = ["baseline"] + self.controls + self.channels
        out["fitted"] = out[parts].sum(axis=1)
        out["residual"] = out["actual"] - out["fitted"]
        return out

    def _curve(self, xs: np.ndarray, P: dict | None = None, fam: Family | None = None) -> np.ndarray:
        """Steady-state contribution, model units, per draw and paid channel, at scaled spend xs."""
        P = self._draws if P is None else P
        fam = self._fam if fam is None else fam
        km, n = len(self.media), len(P["beta"])
        xs = np.broadcast_to(np.asarray(xs, dtype=float), (n, km))
        out = np.empty((n, km))
        for j, c in enumerate(self.media):
            sat = self._saturate(xs[:, j][None, :], P, j, fam.sat[c])[0]
            out[:, j] = P["beta"][:, j] * sat * self._mult_now(P, j, fam)
        return out

    def response_curve(self, channel: str, spend_levels: np.ndarray) -> pd.DataFrame:
        """Steady-state contribution per period at constant spend levels, with a 90% interval."""
        self._require_fit()
        j = self.media.index(channel)
        lv = np.asarray(spend_levels, dtype=float) / self.spend_scale[j]
        ad = np.repeat(lv[:, None], len(self._draws["beta"]), axis=1)
        sat = self._saturate(ad, self._draws, j, self._fam.sat[channel])
        r = (self._draws["beta"][:, j] * self._mult_now(self._draws, j) * sat).T * self.y_scale
        return pd.DataFrame({"spend": spend_levels, "contribution_mean": r.mean(0),
                             "contribution_p05": np.quantile(r, 0.05, axis=0),
                             "contribution_p95": np.quantile(r, 0.95, axis=0)})

    def posterior_curves(self) -> dict:
        """Posterior draws of each paid channel's response parameters, in raw units.

        Returns a dict with ``media`` and arrays of shape (draws, channels): ``beta``
        (maximum contribution per period as of the last fitted period, target units),
        ``half_sat`` (spend per period at the curve's midpoint, or the logistic scale),
        ``alpha``, ``peak`` (NaN unless delayed), ``shape`` (Hill shape), plus
        ``kernel`` (draws, channels, l_max) with each channel's carryover weights and
        ``saturation`` (the curve kind per channel). Steady-state contribution at
        constant spend x is ``beta * f(x)`` with f the channel's saturation curve. This
        is the hand-off format used by ``pycausalsim.futures.media_plan_scm``.
        """
        self._require_fit()
        d, km = self._draws, len(self.media)
        mult = np.stack([self._mult_now(d, j) for j in range(km)], axis=1)
        kernel = np.stack([self._kernel(d, j, self._fam.carry[c]) for j, c in enumerate(self.media)], axis=1)
        return {"media": list(self.media),
                "beta": d["beta"][:, :km] * mult * self.y_scale,
                "half_sat": d["half_sat"][:, :km] * self.spend_scale,
                "alpha": d["alpha"][:, :km].copy(), "peak": d["peak"][:, :km].copy(),
                "shape": d["shape"][:, :km].copy(), "kernel": kernel,
                "saturation": [self._fam.sat[c] for c in self.media],
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
        concave = all(k == "hill" for k in self._fam.sat.values())
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
                        support_quantile: float = 0.95, bagging: int = 0) -> pd.DataFrame:
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

        bagging : if positive, solve the plan on this many random halves of the
            posterior draws and average them. When two channels have nearly the same
            marginal return, a single solve can swing between them on noise; the
            average moves less and only where the halves agree.

        The result carries ``attrs["summary"]`` with the expected gain over the
        current mix, its 90% interval, the probability the plan beats it, and
        ``indistinguishable``: channel pairs the posterior cannot rank at the current
        mix (see :meth:`indistinguishable_channels` and :meth:`plan_stability`).
        """
        self._require_fit()
        if objective not in ("mean", "quantile"):
            raise ValueError('objective must be "mean" or "quantile"')
        if support not in ("flag", "limit", "ignore"):
            raise ValueError('support must be "flag", "limit" or "ignore"')
        current, lims, observed_hi = self._plan_limits(total, bounds, max_change, support, support_quantile)
        if bagging:
            rng = np.random.default_rng(self.seed + 31)
            n = len(self._draws["beta"])
            plans = []
            for _ in range(int(bagging)):
                w = np.zeros(n); w[rng.choice(n, n // 2, replace=False)] = 1.0; w /= w.sum()
                plans.append(self._solve_plan(total, current, lims, objective, quantile, weights=w))
            x_opt = np.mean(plans, axis=0)        # the average of feasible plans is feasible (limits are a box)
        else:
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
            "objective": objective, "max_change": max_change, "support": support, "bagging": int(bagging),
            "indistinguishable": [tuple(r) for r in self.indistinguishable_channels(current)[["channel_a", "channel_b"]].to_numpy()]}
        return out

    def _marginal_at(self, x: np.ndarray, rel: float = 0.01) -> np.ndarray:
        """Marginal return per draw and paid channel at spend vector x (target units per unit of spend)."""
        xs = np.asarray(x, dtype=float) / self.spend_scale
        base = self._curve(xs)
        out = np.empty_like(base)
        for j in range(len(self.media)):
            h = max(rel * xs[j], 1e-6)
            up = xs.copy(); up[j] += h
            out[:, j] = (self._curve(up)[:, j] - base[:, j]) / (h * self.spend_scale[j])
        return out * self.y_scale

    def indistinguishable_channels(self, spend=None, band: float = 0.15) -> pd.DataFrame:
        """Pairs of channels whose marginal returns the posterior cannot rank.

        For each pair, the probability that moving a dollar from the second channel to
        the first pays off at ``spend`` (default: the historical mix). Pairs within
        ``band`` of 0.5 are listed: a planner moving money between them is acting on
        noise, which is how nearly identical response curves produce unstable plans.
        """
        self._require_fit()
        x = self.spend_raw.mean(axis=0) if spend is None else self._as_vector(spend)
        m = self._marginal_at(x)
        rows = []
        for a in range(len(self.media)):
            for b in range(a + 1, len(self.media)):
                p = float(np.mean(m[:, a] > m[:, b]))
                if abs(p - 0.5) <= band:
                    rows.append({"channel_a": self.media[a], "channel_b": self.media[b], "prob_a_higher": p,
                                 "marginal_a": float(m[:, a].mean()), "marginal_b": float(m[:, b].mean())})
        return pd.DataFrame(rows, columns=["channel_a", "channel_b", "prob_a_higher", "marginal_a", "marginal_b"])

    def plan_stability(self, total: float, n_splits: int = 10, seed: int | None = None, **plan_kwargs) -> pd.DataFrame:
        """How much the recommended plan depends on which posterior draws it was built from.

        Re-solves the plan on ``n_splits`` random halves of the posterior draws and
        reports, per channel, the spread of planned spend and the share of halves that
        move the channel in the same direction as the full plan. Low agreement means
        the plan's move on that channel is not supported by the evidence.
        """
        self._require_fit()
        objective = plan_kwargs.get("objective", "quantile"); quantile = plan_kwargs.get("quantile", 0.10)
        current, lims, _ = self._plan_limits(total, plan_kwargs.get("bounds"), plan_kwargs.get("max_change", 0.5),
                                             plan_kwargs.get("support", "flag"), 0.95)
        full = self._solve_plan(total, current, lims, objective, quantile)
        rng = np.random.default_rng(self.seed + 29 if seed is None else seed)
        n = len(self._draws["beta"])
        plans = []
        for _ in range(n_splits):
            w = np.zeros(n); w[rng.choice(n, n // 2, replace=False)] = 1.0; w /= w.sum()
            plans.append(self._solve_plan(total, current, lims, objective, quantile, weights=w))
        plans = np.array(plans)
        move = np.sign(np.round(full - current, 9))
        agree = np.mean(np.sign(np.round(plans - current, 9)) == move, axis=0)
        return pd.DataFrame({"current_spend": current, "planned_spend": full,
                             "half_p10": np.quantile(plans, 0.10, axis=0), "half_p90": np.quantile(plans, 0.90, axis=0),
                             "direction_agreement": agree, "stable": agree >= 0.8},
                            index=pd.Index(self.media, name="channel"))

    # ------------------------------------------------------- checking the model
    def _clone(self, data: pd.DataFrame, seed: int | None = None, **overrides) -> "MMM":
        """A new, unfitted model with this model's settings and family on other data."""
        kw = dict(self._config) | {"carryover": dict(self._fam.carry), "saturation": dict(self._fam.sat),
                                   "time_varying": list(self._fam.tv),
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
        pred = m._predict(test, seed=self.seed + 1)["draws"]
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
                return float(self._curve(np.asarray(x, dtype=float) / self.spend_scale, P=tw).sum() * self.y_scale)
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
            "plan_lost_money": float(np.mean(gains[:, 0] < -1e-6 * np.maximum(gains[:, 1], 1.0))),
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


    # ------------------------------------------------------ forward planning
    def _predict(self, future: pd.DataFrame, seed: int | None = None, n: int | None = None) -> dict:
        """Posterior predictive draws for periods that follow the fitted window.

        ``future`` holds the date, each paid and organic channel (and reach and
        frequency columns) and controls for the new periods. Carryover from the fitted
        window is included. Controls missing from ``future`` are held at their mean.
        """
        self._require_fit()
        need = list(self.channels) + [x for pair in self.reach_frequency.values() for x in pair]
        missing = [c for c in need if c not in future.columns]
        if missing:
            raise ValueError(f"future periods need columns {missing}")
        T = len(self.data)
        both = pd.concat([self.data[need + [self.date_col]], future[need + [self.date_col]]], ignore_index=True)
        d = self._draws if n is None else {k: v[:n] for k, v in self._draws.items()}
        eff = self._effects(d, src=self._sources(both))[T:]                  # (h, draws, channels)
        fit_eff = self._effects(d)
        mn, bn = self._marginal_parts(self.y[:, None] - fit_eff.sum(axis=2))
        xh = self._design(future.reset_index(drop=True), T)
        mean = xh @ mn + eff.sum(axis=2)
        lev = 1 + np.einsum("ij,jk,ik->i", xh, self._vn, xh)
        scale = np.sqrt(bn / self._an * lev[:, None])
        rng = np.random.default_rng(self.seed + 1 if seed is None else seed)
        draws = (mean + scale * rng.standard_t(2 * self._an, size=mean.shape)) * self.y_scale
        return {"draws": draws, "mean": mean * self.y_scale, "effects": eff * self.y_scale}

    def future_frame(self, periods: int, spend: dict | None = None, controls: dict | None = None,
                     start=None) -> pd.DataFrame:
        """A frame of ``periods`` future periods for :meth:`forecast`.

        ``spend`` maps channels to a constant spend per period or a sequence of length
        ``periods`` (flighting); channels left out continue at their historical mean.
        ``controls`` works the same way. Reach and frequency columns scale with spend
        at the historical frequency.
        """
        last = pd.to_datetime(self.data[self.date_col]).iloc[-1]
        step = pd.to_datetime(self.data[self.date_col]).diff().median()
        dates = pd.date_range(last + step if start is None else pd.Timestamp(start), periods=periods, freq=step)
        out = pd.DataFrame({self.date_col: dates})
        spend, controls = dict(spend or {}), dict(controls or {})
        for c in self.channels:
            v = spend.get(c, self.data[c].mean())
            out[c] = np.broadcast_to(np.asarray(v, dtype=float), (periods,)).copy()
        for c, (rcol, fcol) in self.reach_frequency.items():
            ratio = out[c] / max(self.data[c].mean(), 1e-12)
            out[rcol] = self.data[rcol].mean() * ratio
            out[fcol] = self.data[fcol].mean()
        for c in self.controls:
            v = controls.get(c, self.data[c].mean())
            out[c] = np.broadcast_to(np.asarray(v, dtype=float), (periods,)).copy()
        return out

    def forecast(self, future: pd.DataFrame) -> pd.DataFrame:
        """Forecast the target for future periods under a spend plan, with a 90% interval.

        Unlike :meth:`optimize_budget`, which compares steady states, this plays the
        plan forward period by period: carryover from history fades out, changes ramp
        in, and flighted plans are handled as written. Returns one row per period with
        the predicted mean, 5th and 95th percentiles and each channel's contribution.
        """
        p = self._predict(future.reset_index(drop=True))
        out = pd.DataFrame({self.date_col: future[self.date_col].to_numpy(),
                            "predicted": p["draws"].mean(axis=1),
                            "p05": np.quantile(p["draws"], 0.05, axis=1),
                            "p95": np.quantile(p["draws"], 0.95, axis=1)})
        for j, c in enumerate(self.channels):
            out[c] = p["effects"][:, :, j].mean(axis=1)
        out.attrs["total_draws"] = p["draws"].sum(axis=0)
        return out

    def scenarios(self, plans: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Compare future spend plans: total outcome, spend, and probability each beats the first.

        ``plans`` maps a name to a frame of future periods (see :meth:`future_frame`).
        All plans are evaluated on the same posterior draws, so differences are paired.
        """
        rows, first = [], None
        for name, f in plans.items():
            fc = self.forecast(f)
            tot = fc.attrs["total_draws"]
            spend = float(f[self.media].to_numpy(dtype=float).sum())
            if first is None:
                first = tot
            rows.append({"plan": name, "spend": spend, "outcome_mean": float(tot.mean()),
                         "outcome_p05": float(np.quantile(tot, 0.05)), "outcome_p95": float(np.quantile(tot, 0.95)),
                         "gain_vs_first": float((tot - first).mean()),
                         "prob_beats_first": float(np.mean(tot > first)) if tot is not first else np.nan})
        return pd.DataFrame(rows).set_index("plan")

    def monitor(self, new_data: pd.DataFrame, coverage_alert: float = 0.7, z_alert: float = 1.0) -> dict:
        """Check periods that arrived after the fit against what the model predicted.

        ``new_data`` has the same columns as the fitted data, including the actual
        target. Reports how many periods fell inside the 90% interval, the mean
        standardized error, spend outside the range the model has seen, and an
        ``alerts`` list. Any alert means the model should be refit, and possibly
        rethought, before its plans are used again.
        """
        p = self._predict(new_data.reset_index(drop=True))
        actual = new_data[self.target].to_numpy(dtype=float)
        lo, hi = np.quantile(p["draws"], 0.05, axis=1), np.quantile(p["draws"], 0.95, axis=1)
        sd = p["draws"].std(axis=1)
        z = (actual - p["draws"].mean(axis=1)) / np.maximum(sd, 1e-12)
        cover = float(np.mean((actual >= lo) & (actual <= hi)))
        seen_hi = self.data[self.media].max()
        outside = [c for c in self.media if new_data[c].max() > 1.25 * seen_hi[c]]
        alerts = []
        if cover < coverage_alert:
            alerts.append(f"only {cover:.0%} of new periods fell inside the 90% interval")
        if abs(z.mean()) > z_alert:
            alerts.append(f"actuals run {'above' if z.mean() > 0 else 'below'} the forecast by {abs(z.mean()):.1f} "
                          "standard deviations on average")
        for c in outside:
            alerts.append(f"{c} spend went more than 25% above anything in the fitted data")
        return {"periods": len(actual), "coverage_90": cover, "mean_z": float(z.mean()),
                "outside_support": outside, "alerts": alerts, "refit_recommended": bool(alerts),
                "detail": pd.DataFrame({self.date_col: new_data[self.date_col].to_numpy(), "actual": actual,
                                        "predicted": p["draws"].mean(axis=1), "p05": lo, "p95": hi, "z": z})}

    # ---------------------------------------------------------- persistence
    def save(self, path) -> None:
        """Save the fitted model (data, settings and posterior draws) to one file. See :mod:`papilon.persist`."""
        from ..persist import save_model
        save_model(self, path)

    @classmethod
    def load(cls, path) -> "MMM":
        """Load a model saved with :meth:`save`, without refitting."""
        from ..persist import load_model
        return load_model(path)

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
