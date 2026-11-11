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
"""Grounded access to a fitted model for language models and agents.

Nothing here calls a language model or needs a network. It gives one what it
needs to talk about a marketing mix model without inventing numbers:

``brief(model)``
    every headline fact and every reason for caution as plain data, computed by
    the model, with the warnings decided by fixed rules rather than by prose.

``ToolBox(model)``
    the model's methods as tools with JSON schemas, in the two layouts chat
    model APIs commonly accept, and a dispatcher that returns JSON-ready results.
    Each result carries the caveats that apply to it.

``SYSTEM_PROMPT``
    instructions that keep an assistant to the tool results.

``python -m papilon.mcp_server`` serves the same tools over the Model Context
Protocol (needs the optional ``mcp`` package).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

__all__ = ["brief", "ToolBox", "SYSTEM_PROMPT"]

SYSTEM_PROMPT = (
    "You are helping someone read a Papilon marketing mix model. Every number you state must come "
    "from a tool result in this conversation; if you have not called the tool, call it or say you do "
    "not have the figure. Give returns with their 90% range, not the point estimate alone. Repeat any "
    "item in a result's `caveats` that bears on the question. Do not describe a budget plan as safe "
    "when `prob_better` is under 0.8 or a channel is flagged `outside_support`. Relations between "
    "channels and drivers come from the knowledge graph tool when one is available; relations marked "
    "proposed are unconfirmed and must be described that way. The model estimates returns under its "
    "assumptions; it does not prove causation."
)


def _clean(x):
    """Make pandas and numpy values JSON-ready."""
    if isinstance(x, pd.DataFrame):
        return [_clean(r) for r in x.reset_index().to_dict("records")]
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return [_clean(v) for v in list(x)]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return x


def brief(model, plan: pd.DataFrame | None = None, graph=None) -> dict:
    """Facts and cautions about a fitted model, as plain data.

    ``plan`` is a result of ``optimize_budget()``; ``graph`` a
    :class:`papilon.MarketingGraph`. The ``caveats`` list is produced by fixed
    rules (weak importance weights, ranges too wide to rank channels, plans that
    leave the data or are not clearly better, declared but unobserved drivers),
    so the same model always yields the same warnings.
    """
    model._require_fit()
    roi = model.get_roi()
    caveats = []
    gates = {}
    for name, g in model.gate_results.items():
        gates[name] = {"verdict": g.verdict, "ess_fraction": g.ess, "pareto_k": g.pareto_k,
                       "probability": model.family_probabilities.get(name)}
    win = max(model.family_probabilities, key=model.family_probabilities.get)
    g = model.gate_results[win]
    if g.verdict != "ACCEPT":
        caveats.append(f"The fit did not pass its own quality gate (verdict {g.verdict}); treat every estimate as provisional.")
    if np.isfinite(g.pareto_k) and g.pareto_k > 0.7:
        caveats.append(f"Importance weights are unreliable (Pareto k {g.pareto_k:.2f} > 0.7); intervals may be too narrow.")
    channels = {}
    for c in model.media:
        r = roi.loc[c]
        channels[c] = {"roas": r.roas_mean, "roas_p05": r.roas_p05, "roas_p95": r.roas_p95,
                       "marginal_roas": r.mroas_mean, "share_of_sales": r.share_of_sales, "spend": r.spend,
                       "calibrated_by_lift_test": c in model.roi_priors,
                       "carryover": "delayed" if c in model.delayed else "geometric",
                       "saturation": "s_curve" if c in model.s_curve else "concave"}
        if r.roas_p95 - r.roas_p05 > max(1.0, r.roas_mean) and c not in model.roi_priors:
            caveats.append(f"{c}: the 90% range ({r.roas_p05:.2f} to {r.roas_p95:.2f}) is wider than the estimate; "
                           "the data does not pin this channel down. A lift test would.")
    ranked = roi.sort_values("roas_mean", ascending=False)
    if len(ranked) > 1 and ranked.roas_p05.iloc[0] < ranked.roas_p95.iloc[1]:
        caveats.append(f"The ranges of the top two channels ({ranked.index[0]}, {ranked.index[1]}) overlap; "
                       "their order is not established.")
    out = {"papilon_version": _version(), "target": model.target, "periods": len(model.data),
           "family": win, "fit": gates, "channels": channels}
    if plan is not None:
        s = plan.attrs.get("summary", {})
        out["plan"] = {"summary": s, "channels": plan[["current_spend", "optimal_spend", "outside_support"]]}
        if s.get("prob_better", 1.0) < 0.8:
            caveats.append(f"The plan beats the current mix in only {s['prob_better']:.0%} of posterior draws.")
        for c in plan.index[plan.outside_support]:
            caveats.append(f"{c}: planned spend is above anything the brand has sustained; the curve is extrapolated there.")
    if graph is not None:
        audit = graph.audit(model.data)
        out["graph"] = {"description": graph.describe(), "audit": audit}
        for _, r in audit[audit.severity.isin(["high", "medium"])].iterrows():
            caveats.append(f"{r.subject}: {r.detail}")
    out["caveats"] = caveats
    return _clean(out)


def _version() -> str:
    from . import __version__
    return __version__


_CH = {"type": "string", "description": "Channel name, exactly as in the model's media list."}
_SPEND = {"type": "object", "additionalProperties": {"type": "number"},
          "description": "Spend per period by channel, in the data's currency units."}

_TOOLS = [
    ("brief", "Headline facts about the fitted model with every caution that applies: returns with 90% ranges, "
              "fit quality, and (if a knowledge graph is attached) declared relations and their audit.",
     {"type": "object", "properties": {}}),
    ("get_roi", "Return on ad spend per channel: mean, 90% range, marginal return and share of sales.",
     {"type": "object", "properties": {}}),
    ("response_curve", "Steady-state sales contribution of one channel at given spend levels per period, with a 90% range.",
     {"type": "object", "properties": {"channel": _CH, "spend_levels": {"type": "array", "items": {"type": "number"}}},
      "required": ["channel", "spend_levels"]}),
    ("optimize_budget", "Allocate a per-period budget. Default: maximize the 10th percentile of the gain, each "
                        "channel within 50% of current spend.",
     {"type": "object", "properties": {
         "total": {"type": "number", "description": "Budget per period. Omit to keep the historical total."},
         "max_change": {"type": ["number", "null"], "description": "Max fractional move per channel; null removes the cap."},
         "objective": {"type": "string", "enum": ["quantile", "mean"]}}}),
    ("evaluate_plan", "Posterior change in per-period sales from a spend plan versus the current mix.",
     {"type": "object", "properties": {"spend": _SPEND}, "required": ["spend"]}),
    ("value_of_lift_test", "Rank channels by the expected gain to the budget plan from running a lift test on each.",
     {"type": "object", "properties": {"sd": {"type": "number", "description": "Standard error the test would achieve on ROAS."}}}),
    ("backtest", "Refit without the last N periods and forecast them; reports error against a naive forecast.",
     {"type": "object", "properties": {"holdout": {"type": "integer", "minimum": 1}}}),
    ("what_moves", "From the knowledge graph: everything downstream of a channel or driver, with the relation chain.",
     {"type": "object", "properties": {"node": {"type": "string"}}, "required": ["node"]}),
]


class ToolBox:
    """A fitted model (and optionally its knowledge graph) as callable tools.

    >>> box = ToolBox(model, graph=g)
    >>> box.specs("tool_use")             # pass as the tool list of a chat model API
    >>> box.call("get_roi", {})           # run a tool call the model asked for
    """

    def __init__(self, model, graph=None) -> None:
        model._require_fit()
        self.model, self.graph = model, graph
        self._last_plan = None

    def specs(self, format: str = "tool_use") -> list[dict]:
        """Tool definitions.

        ``format`` is "tool_use" (``name``, ``description``, ``input_schema``),
        "function" (wrapped as ``{"type": "function", "function": {...}}`` with
        ``parameters``) or "mcp" (``inputSchema``).
        """
        tools = [t for t in _TOOLS if t[0] != "what_moves" or self.graph is not None]
        if format == "tool_use":
            return [{"name": n, "description": d, "input_schema": s} for n, d, s in tools]
        if format == "function":
            return [{"type": "function", "function": {"name": n, "description": d, "parameters": s}} for n, d, s in tools]
        if format == "mcp":
            return [{"name": n, "description": d, "inputSchema": s} for n, d, s in tools]
        raise ValueError('format must be "tool_use", "function" or "mcp"')

    def call(self, name: str, arguments: dict | None = None) -> dict:
        """Run one tool. Always returns a dict; errors come back as ``{"error": ...}``
        so a model can read them and correct its call."""
        a = dict(arguments or {})
        try:
            result, caveats = getattr(self, f"_t_{name}")(a)
        except AttributeError:
            return {"error": f"unknown tool {name!r}", "tools": [t[0] for t in _TOOLS]}
        except (KeyError, ValueError, TypeError) as e:
            return {"error": f"{type(e).__name__}: {e}"}
        return _clean({"tool": name, "result": result, "caveats": caveats,
                       "units": "spend and sales per period in the data's units"})

    def call_json(self, name: str, arguments: dict | None = None) -> str:
        return json.dumps(self.call(name, arguments))

    def _channel(self, c):
        if c not in self.model.media:
            raise ValueError(f"unknown channel {c!r}; channels are {self.model.media}")
        return c

    def _t_brief(self, a):
        b = brief(self.model, plan=self._last_plan, graph=self.graph)
        return b, b["caveats"]

    def _t_get_roi(self, a):
        return self.model.get_roi(), [c for c in brief(self.model)["caveats"]]

    def _t_response_curve(self, a):
        c = self._channel(a["channel"])
        levels = np.asarray(a["spend_levels"], dtype=float)
        hi = float(self.model.spend_raw[:, self.model.media.index(c)].max())
        cav = [f"Levels above {hi:.4g} (the most {c} has spent in a period) are extrapolation."] if levels.max() > hi else []
        return self.model.response_curve(c, levels), cav

    def _t_optimize_budget(self, a):
        total = a.get("total") or float(self.model.spend_raw.mean(axis=0).sum())
        kw = {k: a[k] for k in ("max_change", "objective") if k in a}
        plan = self.model.optimize_budget(total, **kw)
        self._last_plan = plan
        s = plan.attrs["summary"]
        cav = [f"{c}: planned spend is above anything the brand has sustained." for c in plan.index[plan.outside_support]]
        if s["prob_better"] < 0.8:
            cav.append(f"The plan beats the current mix in only {s['prob_better']:.0%} of posterior draws.")
        cav.append("Gains are steady-state: cuts cost sales immediately and delayed channels pay back later. "
                   "Play the plan forward with pycausalsim.futures.media_plan_scm before moving money.")
        return {"plan": plan, "summary": s}, cav

    def _t_evaluate_plan(self, a):
        spend = {self._channel(c): v for c, v in a["spend"].items()}
        missing = set(self.model.media) - set(spend)
        if missing:
            raise ValueError(f"spend is missing channels {sorted(missing)}")
        ev = self.model.evaluate_plan(spend, self.model.spend_raw.mean(axis=0))
        ev.pop("draws")
        return ev, ([] if ev["prob_better"] >= 0.8 else ["Not clearly better than the current mix."])

    def _t_value_of_lift_test(self, a):
        return self.model.value_of_lift_test(sd=a.get("sd", 0.25)), [
            "Values assume the model's form is right; use them to rank tests, not as forecasts."]

    def _t_backtest(self, a):
        bt = self.model.backtest(int(a.get("holdout", 13)))
        return {"summary": bt.attrs["summary"], "periods": bt}, [
            "A good forecast does not show channel credit is right; it only rules out a broken baseline."]

    def _t_what_moves(self, a):
        if self.graph is None:
            raise ValueError("no knowledge graph is attached to this ToolBox")
        return {"downstream": self.graph.what_moves(a["node"]), "description": self.graph.describe()}, [
            "Relations are declared by the analyst, not learned from the data."]
