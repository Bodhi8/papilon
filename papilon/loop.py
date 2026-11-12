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
"""The learning loop: refresh, check, decide and learn, one cycle per planning period.

A marketing mix model that is refit every quarter on data shaped by its own plans
can lock in its mistakes: it overrates a channel, the plan leans on that channel, the
other channels stop varying, and the next fit grows more confident in the same
error. The loop feeds back only new information, never the model's own conclusions:

* **new periods** of data, scored against the forecast the last cycle made for them;
* **experiment results**: each cycle values every possible lift test by its expected
  gain to the budget plan (:meth:`papilon.MMM.value_of_lift_test`) over a payback
  period, nets off the sales the holdout would give up, and recommends the best one only
  if it is expected to pay for itself; results come back as calibration
  (``lift_tests``) on the periods where they ran;
* **designed variation**: a small, budget-neutral pulse moves money back and forth
  between two channels the model cannot tell apart, so the data the plan produces
  can separate them. This is exploration, as in a bandit, applied to the whole mix.

Whether the loop converges on the truth is an empirical question, and
``pycausalsim.futures.simulate_loop`` answers it on brands where the truth is known.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .mmm import MMM

__all__ = ["Loop"]


class Loop:
    """A measurement loop around :class:`papilon.MMM`.

    Parameters
    ----------
    data : history so far, one row per period.
    target, media, date_col, controls : as :class:`papilon.MMM`.
    horizon : periods each cycle plans for (13 for a quarter of weeks).
    budget : spend per period to allocate; default: the mean of the last ``horizon`` periods.
    explore : size of the exploration pulse as a share of the explored channel's planned
        spend (default 0.10; 0 turns exploration off).
    explore_block : periods per pulse; the pulse alternates up and down in blocks of this length.
    recommend_tests : rank lift tests each cycle and recommend the most valuable one.
    test_sd : standard error on ROAS the recommended test is expected to achieve.
    test_share : share of the market a recommended holdout switches off (default 0.5).
    payback_periods : periods over which a better plan is expected to pay (default 52).
        A test is recommended only when its expected gain to the plan over this many
        periods exceeds the sales the holdout is expected to give up. Experiments cost
        money; the loop runs one only when it is expected to pay for itself.
    carryover_candidates : passed to ``compare_carryover`` each refresh; default: fit once.
    model_kwargs, plan_kwargs : passed to :class:`papilon.MMM` and ``optimize_budget``.
    """

    def __init__(self, data: pd.DataFrame, target: str, media: list[str], date_col: str,
                 controls: list[str] | None = None, horizon: int = 13, budget: float | None = None,
                 explore: float = 0.10, explore_block: int = 2, recommend_tests: bool = True,
                 test_sd: float = 0.25, test_weeks: int = 8, test_share: float = 0.5,
                 payback_periods: int = 52, carryover_candidates: dict | None = None,
                 model_kwargs: dict | None = None, plan_kwargs: dict | None = None, seed: int = 0,
                 n_sim: int = 40):
        self.data = data.sort_values(date_col).reset_index(drop=True).copy()
        self.target, self.media, self.date_col = target, list(media), date_col
        self.controls = list(controls or [])
        self.horizon, self.budget, self.explore, self.explore_block = int(horizon), budget, float(explore), int(explore_block)
        self.recommend_tests, self.test_sd, self.test_weeks = recommend_tests, float(test_sd), int(test_weeks)
        self.test_share, self.payback_periods = float(test_share), int(payback_periods)
        self.carryover_candidates = carryover_candidates
        self.model_kwargs, self.plan_kwargs = dict(model_kwargs or {}), dict(plan_kwargs or {})
        self.seed, self.n_sim = seed, n_sim
        self.tests: list[dict] = []
        self.log: list[dict] = []
        self.model: MMM | None = None
        self._pending: pd.DataFrame | None = None

    # ------------------------------------------------------------------ learn
    def record_test(self, channel: str, start, end, lift: float, se: float, factor: float | None = None,
                    spend_change: float | None = None, readout: int | None = None) -> None:
        """Add an experiment result. ``lift`` is incremental outcome (test versus no test) over
        the test and ``readout`` periods; ``factor`` is the change the test made to the channel's
        spend relative to the spend recorded in the data, or give ``spend_change``."""
        t = {"channel": channel, "start": start, "end": end, "lift": float(lift), "se": float(se)}
        if factor is not None:
            t["factor"] = float(factor)
        if spend_change is not None:
            t["spend_change"] = float(spend_change)
        if readout is not None:
            t["readout"] = int(readout)
        self.tests.append(t)

    def _score_pending(self, new: pd.DataFrame) -> dict | None:
        """Score the last model on the new periods, using the spend that actually ran (so the
        score measures the model, not whether the plan was followed), and measure adherence."""
        if self.model is None:
            return None
        mon = self.model.monitor(new)
        d = mon["detail"]
        a = d["actual"].to_numpy(dtype=float)
        out = {"periods": len(d), "coverage_90": mon["coverage_90"],
               "mape": float(np.mean(np.abs(d.predicted - a) / np.abs(a))), "bias": float(np.mean(a - d.predicted)),
               "alerts": mon["alerts"]}
        if self._pending is not None:
            sched = self._pending.merge(new[[self.date_col] + self.media], on=self.date_col, suffixes=("_plan", ""))
            if len(sched):
                planned = sched[[f"{c}_plan" for c in self.media]].to_numpy(dtype=float)
                actual = sched[self.media].to_numpy(dtype=float)
                out["plan_adherence"] = float(1 - np.abs(actual - planned).sum() / max(planned.sum(), 1e-12))
        return out

    # ------------------------------------------------------------------ cycle
    def _exploration(self, m: MMM, plan: pd.DataFrame) -> tuple[list[str], dict]:
        """Pick two channels and a budget-neutral pulse between them."""
        if self.explore <= 0:
            return [], {}
        pairs = m.indistinguishable_channels(plan["optimal_spend"].to_numpy())
        if len(pairs):
            r = pairs.iloc[(pairs.prob_a_higher - 0.5).abs().argmin()]
            a, b = r.channel_a, r.channel_b
        else:
            roi = m.get_roi()
            width = (roi.roas_p95 - roi.roas_p05) * plan["optimal_spend"]
            a = width.idxmax()
            b = plan["optimal_spend"].drop(a).idxmax()
        amp = self.explore * min(plan.loc[a, "optimal_spend"], plan.loc[b, "optimal_spend"])
        sign = np.array([1.0 if (t // self.explore_block) % 2 == 0 else -1.0 for t in range(self.horizon)])
        return [a, b], {a: plan.loc[a, "optimal_spend"] + amp * sign, b: plan.loc[b, "optimal_spend"] - amp * sign}

    def cycle(self, new_data: pd.DataFrame | None = None, tests: list[dict] | None = None) -> dict:
        """Run one cycle. ``new_data``: periods since the last cycle; ``tests``: experiment
        results to record (dicts with the arguments of :meth:`record_test`).

        Returns a dict with ``model``, ``plan`` (steady-state allocation), ``schedule``
        (spend per period for the next ``horizon`` periods, pulse included),
        ``forecast`` for that schedule, ``experiment`` (the recommended lift test, or
        None), ``explore`` (the two channels pulsed), ``forecast_score`` (how the last
        cycle's forecast did on ``new_data``) and ``alerts``.
        """
        k = len(self.log)
        score = None
        alerts = []
        if new_data is not None and len(new_data):
            new_data = new_data.sort_values(self.date_col)
            score = self._score_pending(new_data)
            if score:
                alerts = list(score.pop("alerts"))
            self.data = pd.concat([self.data, new_data], ignore_index=True)
        for t in tests or []:
            self.record_test(**t)
        m = MMM(self.data, self.target, self.media, self.date_col, controls=self.controls,
                lift_tests=self.tests or None, seed=self.seed + k, **self.model_kwargs)
        if self.carryover_candidates:
            m.compare_carryover(self.carryover_candidates)
        else:
            m.fit()
        self.model = m
        gate = m.gate_results[max(m.family_probabilities, key=m.family_probabilities.get)]
        if not gate.passed:
            alerts.append(f"the fit did not pass its quality gate ({gate.verdict})")
        recent = self.data[self.media].tail(self.horizon).sum(axis=1).mean()
        total = float(self.budget if self.budget is not None else recent)
        plan = m.optimize_budget(total, **self.plan_kwargs)
        explored, pulse = self._exploration(m, plan)
        spend = {c: pulse.get(c, plan.loc[c, "optimal_spend"]) for c in self.media}
        schedule = m.future_frame(self.horizon, spend=spend)
        forecast = m.forecast(schedule)
        self._pending = schedule[[self.date_col] + self.media]
        experiment, candidates = None, None
        if self.recommend_tests:
            v = m.value_of_lift_test(sd=self.test_sd, total=total, n_sim=self.n_sim, seed=self.seed + 97 + k)
            x = plan["optimal_spend"].to_numpy()
            xs = x / m.spend_scale
            full = m._curve(xs).mean(axis=0)
            weeks = min(self.test_weeks, self.horizon)
            rows = []
            for j, c in enumerate(self.media):
                cut = xs.copy(); cut[j] *= (1 - self.test_share)
                lost = float((full[j] - m._curve(cut).mean(axis=0)[j]) * m.y_scale * weeks)
                value = float(v.loc[c, "expected_gain"]) * self.payback_periods
                rows.append({"channel": c, "value": value, "cost": lost, "net": value - lost,
                             "significant": v.loc[c, "expected_gain"] > 2 * v.loc[c, "gain_se"]})
            candidates = pd.DataFrame(rows).set_index("channel").sort_values("net", ascending=False)
            top = candidates.index[0]
            if candidates.loc[top, "net"] > 0 and candidates.loc[top, "significant"]:
                start = schedule[self.date_col].iloc[0]
                experiment = {"channel": top, "start": start,
                              "end": schedule[self.date_col].iloc[weeks - 1],
                              "expected_gain_per_period": float(v.loc[top, "expected_gain"]),
                              "expected_value": float(candidates.loc[top, "value"]),
                              "expected_cost": float(candidates.loc[top, "cost"]),
                              "roas_range_now": (float(v.loc[top, "roas_p05"]), float(v.loc[top, "roas_p95"])),
                              "expected_range_width_after": float(v.loc[top, "width_after"]),
                              "design": f"geo holdout: stop {top} in about {self.test_share:.0%} of the market "
                                        f"(matched regions) for {weeks} periods and measure incremental "
                                        f"{self.target} over the test plus {m.l_max - 1} periods of carryover"}
        roi = m.get_roi()
        entry = {"cycle": k, "periods": len(self.data), "family": max(m.family_probabilities, key=m.family_probabilities.get),
                 "gate": gate.verdict, "tests_used": len(self.tests), "plan": plan, "schedule": schedule,
                 "forecast": forecast, "experiment": experiment, "test_candidates": candidates, "explore": explored,
                 "forecast_score": score, "alerts": alerts, "roi": roi, "model": m}
        self.log.append(entry)
        return entry

    def history(self) -> pd.DataFrame:
        """One row per cycle: data size, family, gate, tests used, forecast score, experiment and returns."""
        rows = []
        for e in self.log:
            r = {"cycle": e["cycle"], "periods": e["periods"], "family": e["family"], "gate": e["gate"],
                 "tests_used": e["tests_used"], "experiment": (e["experiment"] or {}).get("channel"),
                 "explore": " / ".join(e["explore"]), "alerts": len(e["alerts"])}
            if e["forecast_score"]:
                r |= {f"last_model_{k}": v for k, v in e["forecast_score"].items() if k != "periods"}
            for c, row in e["roi"].iterrows():
                r[f"roas_{c}"] = row.roas_mean
                r[f"width_{c}"] = row.roas_p95 - row.roas_p05
            rows.append(r)
        return pd.DataFrame(rows)
