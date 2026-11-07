"""Bayesian marketing mix model with gated importance sampling (Papilon RFC 001).

Inference follows the path in ``papilon-amortized-inference-design.md``:

1. a proposal distribution over the media parameters (here a Laplace fit at the
   posterior mode, standing in for the amortized network described in the RFC),
2. importance sampling against the exact posterior,
3. an effective-sample-size gate that accepts the proposal, refines it from the
   weighted draws, or escalates,
4. marginal-likelihood estimates from the same weights, so carryover families
   (geometric vs delayed adstock) can be compared by Bayes factor.

Baseline terms (intercept, trend, seasonality, controls) and the noise variance
are integrated out in closed form with a conjugate prior, so sampling runs only
over the nonlinear media parameters. Channel effects are constrained positive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import betaln, expit, gammaln, logsumexp

from .transforms import delayed_adstock, geometric_adstock

__all__ = ["MMM", "GateResult"]


@dataclass
class GateResult:
    """Outcome of the importance-sampling gate for one carryover family."""

    family: str
    delayed: list[str]
    log_evidence: float
    ess: float
    verdict: str
    history: list[tuple[int, float, str]] = field(default_factory=list)
    pareto_k: float = float("nan")  # PSIS tail shape; above 0.7 the importance weights are unreliable


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
    alpha_prior : Beta(a, b) prior on carryover retention, either one pair for
        every channel or a dict of pairs by channel. Default (2, 2).
    roi_priors : optional dict mapping a channel to (mean, sd) of its return on
        spend over the fitted window, typically from a lift test or geo
        experiment. Added to the posterior as a normal prior on that ROAS.
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
    ) -> None:
        df = data.sort_values(date_col).reset_index(drop=True)
        self.data, self.target, self.media = df, target, list(media)
        self.date_col, self.controls = date_col, list(controls or [])
        self.l_max, self.period = int(l_max), float(period)
        self.delayed = list(delayed or [])
        self._check_delayed(self.delayed)
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
        self._roi_idx = [(self.media.index(c), float(m), float(sd)) for c, (m, sd) in
                         (roi_priors or {}).items() if c in self.media]

        n = len(df)
        t = np.arange(n) / self.period
        cols, sds = [np.ones(n)], [100.0]
        if trend:
            cols.append(t); sds.append(10.0)
        for k in range(1, yearly_seasonality + 1):
            cols += [np.sin(2 * np.pi * k * t), np.cos(2 * np.pi * k * t)]; sds += [10.0, 10.0]
        for c in self.controls:
            v = df[c].to_numpy(dtype=float)
            sd = v.std() or 1.0
            cols.append((v - v.mean()) / sd); sds.append(20.0)
        self._base_x = np.column_stack(cols)
        self._prior_sd = np.array(sds)  # baseline coefficients, in units of sigma
        v0_inv = np.diag(1 / self._prior_sd ** 2)
        self._vn_inv = v0_inv + self._base_x.T @ self._base_x
        self._vn = np.linalg.inv(self._vn_inv)
        self._logdet_ratio = np.linalg.slogdet(self._vn)[1] - np.sum(np.log(self._prior_sd ** 2))
        self._a0 = 2.0
        self._b0 = 0.1 * float(np.var(self.y))  # weakly informative noise prior
        self._an = self._a0 + n / 2

        self.gate_results: dict[str, GateResult] = {}
        self.family_probabilities: dict[str, float] = {}
        self._draws: dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------------ model
    def _check_delayed(self, delayed: list[str]) -> None:
        unknown = set(delayed) - set(self.media)
        if unknown:
            raise ValueError(f"delayed channels not in media: {sorted(unknown)}")

    def _dim(self, delayed: list[str]) -> int:
        return 3 * len(self.media) + len(delayed)

    def _unpack(self, z: np.ndarray, delayed: list[str]):
        """Unconstrained vector -> parameters per channel, plus log prior with Jacobian."""
        i, params, lp = 0, {}, 0.0
        max_peak = self.l_max - 2
        for c in self.media:
            za, zh, zb = z[i : i + 3]
            i += 3
            a, hs, b = expit(za), np.exp(zh), np.exp(zb)
            pa, pb = self.alpha_prior[c]
            lp += (_beta_lpdf(a, pa, pb) + np.log(a) + np.log1p(-a)
                   + _lognorm_lpdf(hs, 0.5) + zh
                   + _halfnorm_lpdf(b, 0.5) + zb)
            peak = None
            if c in delayed:
                u = expit(z[i]); i += 1
                peak = max_peak * u
                lp += _beta_lpdf(u, 2.0, 2.0) + np.log(u) + np.log1p(-u)
            params[c] = (a, hs, b, peak)
        return params, lp

    def _channel(self, x: np.ndarray, a: float, hs: float, b: float, peak) -> np.ndarray:
        ad = (delayed_adstock(x, a, peak, self.l_max) if peak is not None
              else geometric_adstock(x, a, self.l_max))
        return b * ad / (ad + hs)

    def _media(self, params, spend: np.ndarray | None = None) -> np.ndarray:
        spend = self.spend if spend is None else spend
        return np.column_stack([self._channel(spend[:, j], *params[c]) for j, c in enumerate(self.media)])

    def _log_marginal(self, resid: np.ndarray) -> float:
        mn = self._vn @ self._base_x.T @ resid
        bn = self._b0 + 0.5 * (resid @ resid - mn @ self._vn_inv @ mn)
        n = len(resid)
        return (-n / 2 * np.log(2 * np.pi) + 0.5 * self._logdet_ratio + self._a0 * np.log(self._b0)
                - self._an * np.log(bn) + gammaln(self._an) - gammaln(self._a0))

    def _log_post(self, z: np.ndarray, delayed: list[str]) -> float:
        params, lp = self._unpack(z, delayed)
        if not np.isfinite(lp):
            return -np.inf
        media = self._media(params)
        for j, m, sd in self._roi_idx:
            roas = media[:, j].sum() * self.y_scale / self._spend_tot[j]
            lp += -0.5 * ((roas - m) / sd) ** 2 - np.log(sd) - _HALF_LOG_2PI
        return self._log_marginal(self.y - media.sum(axis=1)) + lp

    # -------------------------------------------------------------- inference
    def _laplace(self, delayed: list[str], n_starts: int):
        f = lambda z: -self._log_post(z, delayed)
        z_init = np.concatenate([[0.0, 0.0, np.log(0.2)] for _ in self.media] + [[0.0]] * len(delayed))
        best = None
        for _ in range(n_starts):
            z0 = z_init + self.rng.normal(0, 0.3, len(z_init))
            r = optimize.minimize(f, z0, method="Nelder-Mead",
                                  options={"maxiter": 20000, "maxfev": 20000, "xatol": 1e-4, "fatol": 1e-6})
            r = optimize.minimize(f, r.x, method="BFGS")
            if best is None or r.fun < best.fun:
                best = r
        d, h = len(best.x), 1e-3
        hess = np.zeros((d, d))
        for a in range(d):
            for b in range(a, d):
                ea = np.zeros(d); ea[a] = h
                eb = np.zeros(d); eb[b] = h
                hess[a, b] = hess[b, a] = (f(best.x + ea + eb) - f(best.x + ea - eb)
                                           - f(best.x - ea + eb) + f(best.x - ea - eb)) / (4 * h * h)
        vals, vecs = np.linalg.eigh(0.5 * (hess + hess.T))
        return best.x, vecs @ np.diag(1 / np.maximum(vals, 0.5)) @ vecs.T

    def _importance(self, delayed, mean, cov, n_draws):
        q = stats.multivariate_t(loc=mean, shape=cov, df=5, seed=self.rng)
        z = q.rvs(n_draws)
        log_w = np.array([self._log_post(zi, delayed) for zi in z]) - q.logpdf(z)
        log_ev = float(logsumexp(log_w) - np.log(n_draws))
        w = np.exp(log_w - log_w.max()); w /= w.sum()
        self._last_k = _pareto_k(log_w)
        return z, w, log_ev, float(1 / np.sum(w ** 2) / n_draws)

    def _gate(self, name, delayed, n_draws, n_starts, accept, refine, max_refine):
        mean, cov = self._laplace(delayed, n_starts)
        history = []
        for step in range(max_refine + 1):
            z, w, log_ev, ess = self._importance(delayed, mean, 1.5 * cov, n_draws)
            verdict = "ACCEPT" if ess > accept else "REFINE" if ess > refine else "ESCALATE"
            history.append((step, round(ess, 3), verdict))
            if verdict == "ACCEPT":
                break
            mean = np.average(z, axis=0, weights=w)  # refine: moment-match the weighted draws
            cov = np.cov(z.T, aweights=w) + 1e-6 * np.eye(len(mean))
        return GateResult(name, list(delayed), log_ev, ess, verdict, history,
                          pareto_k=float(getattr(self, "_last_k", np.nan))), z, w

    def _store_draws(self, delayed, z, w, n_keep=2000):
        idx = self.rng.choice(len(w), n_keep, p=w)
        draws = [self._unpack(zi, delayed)[0] for zi in z[idx]]
        self._draws = {
            "alpha": np.array([[d[c][0] for c in self.media] for d in draws]),
            "half_sat": np.array([[d[c][1] for c in self.media] for d in draws]),
            "beta": np.array([[d[c][2] for c in self.media] for d in draws]),
            "peak": np.array([[np.nan if d[c][3] is None else d[c][3] for c in self.media] for d in draws]),
        }
        self.delayed = list(delayed)
        self._roi_cache = None

    def fit(self, n_draws: int = 4000, n_starts: int = 3, accept: float = 0.10,
            refine: float = 0.01, max_refine: int = 3) -> "MMM":
        """Fit the model with the carryover family set by ``delayed``."""
        name = "delayed:" + ",".join(self.delayed) if self.delayed else "all_geometric"
        res, z, w = self._gate(name, self.delayed, n_draws, n_starts, accept, refine, max_refine)
        self.gate_results = {name: res}
        self.family_probabilities = {name: 1.0}
        self._store_draws(self.delayed, z, w)
        return self

    def compare_carryover(self, candidates: dict[str, list[str]], n_draws: int = 4000,
                          n_starts: int = 3, accept: float = 0.10, refine: float = 0.01,
                          max_refine: int = 3) -> pd.DataFrame:
        """Fit each carryover family, compare them by marginal likelihood, keep the winner.

        ``candidates`` maps a family name to the channels that use delayed adstock,
        for example ``{"all_geometric": [], "video_delayed": ["online_video"]}``.
        Families get equal prior odds.
        """
        fitted = {}
        for name, delayed in candidates.items():
            self._check_delayed(delayed)
            fitted[name] = self._gate(name, delayed, n_draws, n_starts, accept, refine, max_refine)
        names = list(fitted)
        ev = np.array([fitted[k][0].log_evidence for k in names])
        probs = np.exp(ev - logsumexp(ev))
        self.gate_results = {k: fitted[k][0] for k in names}
        self.family_probabilities = dict(zip(names, map(float, probs)))
        win = names[int(np.argmax(probs))]
        _, z, w = fitted[win]
        self._store_draws(fitted[win][0].delayed, z, w)
        return pd.DataFrame({
            "log_evidence": ev,
            "log_bayes_factor_vs_best": ev - ev.max(),
            "probability": probs,
            "ess_fraction": [fitted[k][0].ess for k in names],
            "verdict": [fitted[k][0].verdict for k in names],
        }, index=pd.Index(names, name="family"))

    # ---------------------------------------------------------------- results
    def _require_fit(self) -> None:
        if self._draws is None:
            raise RuntimeError("call fit() or compare_carryover() first")

    def _draw_params(self, i: int):
        d = self._draws
        return {c: (d["alpha"][i, j], d["half_sat"][i, j], d["beta"][i, j],
                    None if np.isnan(d["peak"][i, j]) else d["peak"][i, j])
                for j, c in enumerate(self.media)}

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
        }, index=pd.Index(self.media, name="channel"))
        return out

    def get_roi(self) -> pd.DataFrame:
        """Return on spend per channel: posterior mean, 90% interval and marginal return.

        ROAS is the channel's modeled contribution over the fitted window divided
        by its spend. Marginal ROAS is the extra contribution from 1% more spend in
        every period, per extra unit of spend.
        """
        self._require_fit()
        if getattr(self, "_roi_cache", None) is not None:
            return self._roi_cache
        n_draws = len(self._draws["beta"])
        k = len(self.media)
        roas = np.zeros((n_draws, k)); mroas = np.zeros((n_draws, k)); share = np.zeros((n_draws, k))
        spend_tot = self.spend_raw.sum(axis=0)
        for i in range(n_draws):
            p = self._draw_params(i)
            m = self._media(p) * self.y_scale
            for j, c in enumerate(self.media):
                bumped = self._channel(self.spend[:, j] * 1.01, *p[c]) * self.y_scale
                roas[i, j] = m[:, j].sum() / spend_tot[j]
                mroas[i, j] = (bumped.sum() - m[:, j].sum()) / (0.01 * spend_tot[j])
            share[i] = m.sum(axis=0) / self.y_raw.sum()
        self._roi_cache = pd.DataFrame({
            "spend": spend_tot,
            "roas_mean": roas.mean(0),
            "roas_p05": np.quantile(roas, 0.05, axis=0),
            "roas_p95": np.quantile(roas, 0.95, axis=0),
            "mroas_mean": mroas.mean(0),
            "share_of_sales": share.mean(0),
        }, index=pd.Index(self.media, name="channel"))
        return self._roi_cache

    def contributions(self) -> pd.DataFrame:
        """Posterior-mean contribution of each channel per period, in target units."""
        self._require_fit()
        n_draws = len(self._draws["beta"])
        acc = np.zeros_like(self.spend)
        for i in range(n_draws):
            acc += self._media(self._draw_params(i))
        out = pd.DataFrame(acc / n_draws * self.y_scale, columns=self.media)
        out.insert(0, self.date_col, self.data[self.date_col].to_numpy())
        out["baseline_and_controls"] = self.y_raw - out[self.media].sum(axis=1)
        return out

    def response_curve(self, channel: str, spend_levels: np.ndarray) -> pd.DataFrame:
        """Steady-state contribution per period at constant spend levels, with a 90% interval."""
        self._require_fit()
        j = self.media.index(channel)
        x = np.asarray(spend_levels, dtype=float) / self.spend_scale[j]
        b = self._draws["beta"][:, j][:, None]; hs = self._draws["half_sat"][:, j][:, None]
        r = b * x / (x + hs) * self.y_scale
        return pd.DataFrame({"spend": spend_levels, "contribution_mean": r.mean(0),
                             "contribution_p05": np.quantile(r, 0.05, axis=0),
                             "contribution_p95": np.quantile(r, 0.95, axis=0)})

    def posterior_curves(self) -> dict:
        """Posterior draws of each channel's response parameters, in raw units.

        Returns a dict with ``media`` (channel names) and arrays of shape
        (draws, channels): ``beta`` (maximum contribution per period, target
        units), ``half_sat`` (spend per period at half of beta), ``alpha``
        (carryover retention) and ``peak`` (delay of the peak, NaN for
        geometric channels). Steady-state contribution at constant spend x is
        ``beta * x / (x + half_sat)``. This is the hand-off format used by
        simulators such as ``pycausalsim.futures.marketing``.
        """
        self._require_fit()
        d = self._draws
        return {"media": list(self.media),
                "beta": d["beta"] * self.y_scale,
                "half_sat": d["half_sat"] * self.spend_scale,
                "alpha": d["alpha"].copy(), "peak": d["peak"].copy(),
                "l_max": self.l_max,
                "historical_spend": self.spend_raw.mean(axis=0)}

    def _steady(self, x):
        """Steady-state media contribution per period for each posterior draw."""
        xs = np.asarray(x, dtype=float) / self.spend_scale
        b = self._draws["beta"]; hs = self._draws["half_sat"]
        return np.sum(b * xs / (xs + hs), axis=-1) * self.y_scale

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

        base = self._steady(current)
        if objective == "mean":
            f = lambda x: -float(self._steady(x).mean())
        else:
            f = lambda x: -float(np.quantile(self._steady(x) - base, quantile))
        cons = [{"type": "eq", "fun": lambda x: x.sum() - total}]
        x0 = np.clip(current, [lo for lo, _ in lims], [hi for _, hi in lims])
        starts = [x0]
        if objective == "quantile":  # the quantile surface is rough; also start from the mean optimum
            r = optimize.minimize(lambda x: -float(self._steady(x).mean()), x0, method="SLSQP",
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
            if best is None or r.fun < best.fun:
                best = r
        x_opt = best.x

        def per_channel(x):
            xs = np.asarray(x) / self.spend_scale
            return (self._draws["beta"] * xs / (xs + self._draws["half_sat"])).mean(axis=0) * self.y_scale

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
