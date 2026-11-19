#   Copyright 2026 Brian Curry / Vector1 Research
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
"""Papilon's offline agent: question to plan of tool calls, results to answer.

Deterministic and keyword-driven. Every number in an answer is formatted from a
tool result or from the question, so the grounding check in :mod:`papilon.chat`
can trace it. Used by :class:`papilon.chat.OfflineAgent`.
"""

from __future__ import annotations

import re

TASKS = {
    "overview": "Give me an overview: what does each channel return, and how sure is the model?",
    "plan_quarter": ("Plan next quarter's budget: find the best mix at today's total, check which moves are "
                     "stable, say which lift test to run first, and forecast the next 13 weeks under the plan."),
    "audit": "Can I trust this model? Check the fit, the cautions and how well it forecasts weeks it has not seen.",
}

HELP = ("I am the built-in offline agent, so I answer a fixed range of questions. Try: which channel returns "
        "the most; plan the budget (or plan it with a bigger or smaller budget); what if I spend more on one "
        "channel; which lift test to run first; forecast the coming weeks; can I trust this model; plan next "
        "quarter. Connect a language model for open-ended questions.")

_GENERIC = {"paid", "online", "spend", "media", "cost", "channel", "the", "and", "ads", "ad"}
_DOWN = ("cut", "reduce", "less", "lower", "decrease", "down", "drop", "shrink", "smaller", "fewer")


def cli_toolbox(argv):
    """Build a :class:`papilon.ai.ToolBox` from command line options (``python -m papilon.chat``)."""
    import argparse
    import json

    import pandas as pd

    from .ai import ToolBox
    from .graph import MarketingGraph
    from .mmm import MMM

    ap = argparse.ArgumentParser(prog="python -m papilon.chat",
                                 description="Chat with a Papilon marketing mix model fitted to a CSV file.")
    ap.add_argument("--data", required=True, help="CSV with one row per period")
    ap.add_argument("--target", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--media", required=True, help="comma-separated spend columns")
    ap.add_argument("--controls", default="")
    ap.add_argument("--delayed", default="")
    ap.add_argument("--graph", default=None, help="JSON from MarketingGraph.to_dict()")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    split = lambda s: [x for x in s.split(",") if x]
    df = pd.read_csv(a.data, parse_dates=[a.date])
    model = MMM(df, a.target, split(a.media), a.date, controls=split(a.controls),
                delayed=split(a.delayed), seed=a.seed).fit()
    graph = MarketingGraph.from_dict(json.load(open(a.graph))) if a.graph else None
    return ToolBox(model, graph=graph)


class Plan:
    def __init__(self, steps, write):
        self.steps, self.write = steps, write


def _nice(c: str) -> str:
    return str(c).replace("_", " ")


def _has(q: str, *words) -> bool:
    return any(w in q for w in words)


def _channel(q: str, media):
    best = None
    for c in media:
        aliases = {c.lower(), _nice(c).lower()} | {p for p in re.split(r"[_\s]+", c.lower())
                                                    if len(p) > 2 and p not in _GENERIC}
        for a in aliases:
            m = re.search(r"\b" + re.escape(a) + r"\b", q)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), c)
    return best[1] if best else None


def _amounts(q: str):
    """(value, is_percent) for numbers in the question, sign from words like "cut"."""
    out = []
    for m in re.finditer(r"([-+]?\d+(?:\.\d+)?)\s*(%|percent)?", q):
        v = float(m.group(1))
        out.append((v, bool(m.group(2))))
    return out


def _pct(q: str):
    for v, p in _amounts(q):
        if p:
            return -abs(v) / 100 if _has(q, *_DOWN) else v / 100
    return None


def _caveats(done, skip=()):
    seen, out = set(), []
    for d in done:
        for c in d["result"].get("caveats", []) if isinstance(d["result"], dict) else []:
            if c not in seen and not any(s in c for s in skip):
                seen.add(c)
                out.append(c)
    return out


def _cav_block(done, skip=()):
    cav = _caveats(done, skip)
    return ("\n\n**Cautions from the model**\n" + "\n".join(f"- {c}" for c in cav)) if cav else ""


def _err(done):
    bad = [d for d in done if isinstance(d["result"], dict) and "error" in d["result"]]
    return "\n".join(f"The `{d['name']}` call failed: {d['result']['error']}" for d in bad)


def _result(done, name):
    for d in done:
        if d["name"] == name and "result" in d["result"]:
            return d["result"]["result"]
    return None


# ------------------------------------------------------------------ writers

def _w_roi(rows, focus=None):
    rows = sorted(rows, key=lambda r: -r["roas_mean"])
    lines = []
    if focus:
        r = next(r for r in rows if r["channel"] == focus)
        rank = [x["channel"] for x in rows].index(focus) + 1
        lines.append(f"**{_nice(focus)}** returns {r['roas_mean']:.2f} in sales per unit of spend "
                     f"(90% range {r['roas_p05']:.2f} to {r['roas_p95']:.2f}). The next unit of spend returns "
                     f"about {r['mroas_mean']:.2f}. It ranks {rank} of {len(rows)} channels by average return.\n")
    lines.append("Return on ad spend by channel, highest first, with the 90% range and the marginal return "
                 "(what the next unit of spend brings):")
    for r in rows:
        lines.append(f"- **{_nice(r['channel'])}**: {r['roas_mean']:.2f} (range {r['roas_p05']:.2f} to "
                     f"{r['roas_p95']:.2f}), marginal {r['mroas_mean']:.2f}, {r['share_of_sales']:.1%} of sales")
    return "\n".join(lines)


def _w_brief(b):
    gate = b["fit"][b["family"]]
    verdict = gate["verdict"]
    lines = [f"The model explains **{b['target']}** over {b['periods']} periods. The chosen carryover family is "
             f"`{b['family']}` and the fit's quality gate says **{verdict}** (effective sample share "
             f"{gate['ess_fraction']:.2f}, Pareto k {gate['pareto_k']:.2f}, "
             + ("in the reliable range)." if gate["pareto_k"] < 0.7 else "above the reliable limit).")]
    lines.append("\nReturns by channel, with 90% ranges:")
    for c, r in sorted(b["channels"].items(), key=lambda kv: -kv[1]["roas"]):
        cal = ", calibrated by a lift test" if r["calibrated_by_lift_test"] else ""
        lines.append(f"- **{_nice(c)}**: {r['roas']:.2f} (range {r['roas_p05']:.2f} to {r['roas_p95']:.2f}){cal}")
    return "\n".join(lines)


def _w_plan(res, stab=None, change=None):
    s, plan = res["summary"], res["plan"]
    new_total = res.get("total", sum(r["optimal_spend"] for r in plan))
    cap = s.get("max_change")
    rule = ("the best 10th percentile of the gain, so it holds up if the model is somewhat wrong"
            if s.get("objective") == "quantile" else "the best average gain")
    if change:
        head = (f"The plan spends {new_total:.1f} per period, {abs(change):.0%} "
                f"{'more' if change > 0 else 'less'} than today. Each line compares today's mix, scaled to that "
                "total, with the plan.")
    else:
        head = f"The plan keeps today's total of {new_total:.1f} per period and moves money between channels."
    lines = [head + f" It is chosen for {rule}" + (f"; no channel moves more than {cap:.0%}." if cap else
                                                    "; no cap on moves.")]
    for r in plan:
        d = r["optimal_spend"] - r["current_spend"]
        verb = "hold" if abs(d) < 0.05 else ("raise" if d > 0 else "cut")
        flag = " (above anything spent before)" if r["outside_support"] else ""
        lines.append(f"- **{_nice(r['channel'])}**: {verb} {r['current_spend']:.1f} to {r['optimal_spend']:.1f}{flag}")
    lines.append(f"\nExpected gain: **{s['expected_lift']:.1f}** in sales per period (90% range "
                 f"{s['lift_p05']:.1f} to {s['lift_p95']:.1f}). The plan beats today's mix in "
                 f"{s['prob_better']:.0%} of posterior draws" +
                 (", which is strong." if s["prob_better"] >= 0.8 else ", which is not clear enough to act on."))
    for a, b in s.get("indistinguishable", []):
        lines.append(f"The model cannot tell **{_nice(a)}** and **{_nice(b)}** apart, so treat moves between "
                     "them as unproven.")
    if stab:
        steady = [_nice(r["channel"]) for r in stab if r["stable"]]
        shaky = [_nice(r["channel"]) for r in stab if not r["stable"]]
        lines.append("\n**Which moves hold up.** I rebuilt the plan on separate halves of the posterior draws. "
                     + (f"Stable: {', '.join(steady)}. " if steady else "")
                     + (f"Not stable: {', '.join(shaky)}; keep those near current spend or test them first."
                        if shaky else "Every move points the same way in both halves."))
    return "\n".join(lines)


def _w_lift(rows, sd):
    rows = sorted(rows, key=lambda r: -r["expected_gain"])
    top = rows[0]
    lines = [f"Test **{_nice(top['channel'])}** first. A lift test that pins its return to within {sd} would be "
             f"worth about {top['expected_gain']:.2f} in sales per period to the budget plan, and would narrow "
             f"its 90% range from a width of {top['width_now']:.2f} to about {top['width_after']:.2f}.",
             "\nAll channels, by value of a test:"]
    for r in rows:
        lines.append(f"- {_nice(r['channel'])}: {r['expected_gain']:.2f} (range width now {r['width_now']:.2f})")
    return "\n".join(lines)


def _w_forecast(rows, under_plan=False):
    a, z = rows[0], rows[-1]
    how = "under the planned spend" if under_plan else "at today's average spend"
    return (f"Forecast for the next {len(rows)} periods {how}, with 90% ranges:\n"
            f"- first period ({str(a['date'])[:10]}): {a['predicted']:.0f} (range {a['p05']:.0f} to {a['p95']:.0f})\n"
            f"- last period ({str(z['date'])[:10]}): {z['predicted']:.0f} (range {z['p05']:.0f} to {z['p95']:.0f})")


def _w_backtest(res):
    s = res["summary"]
    better = s["mape"] < s["naive_mape"]
    return (f"I refit the model without the last {s['holdout']} periods and forecast them. Average error: "
            f"**{s['mape']:.1%}**, against {s['naive_mape']:.1%} for a naive forecast; "
            f"{s['coverage_90']:.0%} of the actual values fell inside the 90% range. "
            + ("The model beats the naive forecast on weeks it has not seen."
               if better else "The model does not beat a naive forecast; be careful with it."))


def _w_curve(rows, channel, current):
    lines = [f"Sales added by **{_nice(channel)}** at steady state (today it spends {current:.1f} per period):"]
    for r in rows:
        lines.append(f"- at {r['spend']:.1f}: {r['contribution_mean']:.1f} (90% range {r['contribution_p05']:.1f} "
                     f"to {r['contribution_p95']:.1f})")
    if len(rows) >= 3:
        gains = [rows[i + 1]["contribution_mean"] - rows[i]["contribution_mean"] for i in range(len(rows) - 1)]
        steps = [rows[i + 1]["spend"] - rows[i]["spend"] for i in range(len(rows) - 1)]
        per = [g / s for g, s in zip(gains, steps) if s > 0]
        if len(per) >= 2 and per[-1] < per[0]:
            lines.append("Each further step adds less than the one before: the channel is saturating.")
    return "\n".join(lines)


# --------------------------------------------------------------------- plans

def make_plan(question: str, box) -> Plan:
    q = question.lower()
    m = box.model
    media = list(m.media)
    current = {c: float(v) for c, v in zip(media, m.spend_raw.mean(axis=0))}
    total_now = sum(current.values())
    ch = _channel(q, media)
    pct = _pct(q)

    def total():
        if pct is not None and _has(q, "budget", "total", "spend", "more", "less", "plan", "mix"):
            return round(total_now * (1 + pct), 4)
        m_ = re.search(r"(?:budget|total)\s+(?:of|to|at|is|=)?\s*\$?(\d[\d,]*(?:\.\d+)?)(?!\s*(?:%|percent|week|month|period))", q)
        return float(m_.group(1).replace(",", "")) if m_ else None

    budget_args = {"total": total()} if total() else {}
    if _has(q, "no cap", "uncapped", "unconstrained", "without a cap", "any amount"):
        budget_args["max_change"] = None
    plan_total = {"total": budget_args["total"]} if "total" in budget_args else {}

    def fail_or(done, text):
        return _err(done) or text

    # agent task: plan the quarter
    if _has(q, "quarter", "full plan") or (_has(q, "plan") and _has(q, "test") and _has(q, "forecast")):
        def fc_args(done):
            res = _result(done, "optimize_budget")
            spend = {r["channel"]: r["optimal_spend"] for r in res["plan"]} if res else {}
            return {"periods": 13, "spend": spend}

        def write(done):
            if _err(done):
                return _err(done)
            b, opt = _result(done, "brief"), _result(done, "optimize_budget")
            stab, lift = _result(done, "plan_stability"), _result(done, "value_of_lift_test")
            fc = _result(done, "forecast")
            return ("**1. Where the model stands**\n" + _w_brief(b) +
                    "\n\n**2. The budget plan**\n" + _w_plan(opt, stab) +
                    "\n\n**3. What to test first**\n" + _w_lift(lift, 0.25) +
                    "\n\n**4. The next 13 weeks under the plan**\n" + _w_forecast(fc, under_plan=True) +
                    _cav_block(done))
        return Plan([("brief", {}), ("optimize_budget", budget_args), ("plan_stability", plan_total),
                     ("value_of_lift_test", {"sd": 0.25}), ("forecast", fc_args)], write)

    # agent task: audit
    if _has(q, "trust", "reliab", "accura", "how good", "validat", "backtest", "believe", "check the fit",
            "check the model", "sanity"):
        def write(done):
            if _err(done):
                return _err(done)
            b, bt = _result(done, "brief"), _result(done, "backtest")
            gate_ok = b["fit"][b["family"]]["verdict"] == "ACCEPT"
            beats = bt["summary"]["mape"] < bt["summary"]["naive_mape"]
            verdict = ("**Verdict:** the fit passed its quality gate and forecasts unseen weeks better than a "
                       "naive forecast. That supports the baseline. It does not prove each channel's credit is "
                       "right; the ranges and the cautions below say where the data is thin, and a lift test "
                       "is how to settle those." if gate_ok and beats else
                       "**Verdict:** treat this model as provisional; see the fit quality and cautions below.")
            return verdict + "\n\n" + _w_brief(b) + "\n\n" + _w_backtest(bt) + _cav_block(done)
        return Plan([("brief", {}), ("backtest", {"holdout": 13})], write)

    # which lift test
    if _has(q, "lift test", "experiment", "test first", "which test", "should i test", "geo test",
            "incrementality", "what to test", "run a test"):
        return Plan([("value_of_lift_test", {"sd": 0.25})],
                    lambda done: fail_or(done, _w_lift(_result(done, "value_of_lift_test"), 0.25) +
                                         _cav_block(done)))

    # forecast
    if _has(q, "forecast", "predict", "projection", "next week", "next month", "coming week"):
        n = next((int(v) for v, p in _amounts(q) if not p and 1 <= v <= 104), 13)
        if _has(q, "month") and not _has(q, "week"):
            n = min(104, n * 4 if n <= 24 else n)
        return Plan([("forecast", {"periods": n})],
                    lambda done: fail_or(done, _w_forecast(_result(done, "forecast")) + _cav_block(done)))

    # one channel at other spend levels
    budget_words = _has(q, "budget", "allocat", "optimi", "mix", "plan")
    if ch and not budget_words and _has(q, "what if", "if i", "if we", "spend", "double", "halve", "half",
                                        "saturat", "curve", "diminish", "more", "less", "cut", "increase",
                                        "raise", "invest"):
        cur = current[ch]
        nums = [v for v, p in _amounts(q) if not p and v > 0]
        if _has(q, "double"):
            levels = [cur, 2 * cur]
        elif _has(q, "halve", "half"):
            levels = [0.5 * cur, cur]
        elif pct is not None:
            levels = sorted([cur, cur * (1 + pct)])
        elif nums:
            levels = sorted({cur, nums[0]})
        else:
            levels = [0.5 * cur, cur, 1.5 * cur, 2 * cur]
        levels = [round(x, 4) for x in levels]
        return Plan([("response_curve", {"channel": ch, "spend_levels": levels})],
                    lambda done: fail_or(done, _w_curve(_result(done, "response_curve"), ch, cur) +
                                         _cav_block(done)))

    # budget plan
    if budget_words or _has(q, "reallocat", "shift money", "where should", "move money", "spend more",
                            "spend less"):
        def write(done):
            return fail_or(done, _w_plan(_result(done, "optimize_budget"), _result(done, "plan_stability"),
                                         pct if "total" in budget_args and pct is not None else None) +
                           _cav_block(done))
        return Plan([("optimize_budget", budget_args), ("plan_stability", plan_total)], write)

    known = _has(q, "overview", "summary", "summar", "brief", "tell me about", "how sure", "caveat",
                 "caution", "what does each")

    # returns
    if not known and (ch or _has(q, "roi", "roas", "return", "best", "worst", "perform", "working", "effective", "channel",
                  "which", "pay")):
        return Plan([("get_roi", {})],
                    lambda done: fail_or(done, _w_roi(_result(done, "get_roi"), ch) +
                                         _cav_block(done, skip=("planned spend",))))

    # knowledge graph
    if box.graph is not None and _has(q, "what moves", "downstream", "affect", "graph", "relation"):
        node = ch or media[0]
        return Plan([("what_moves", {"node": node})],
                    lambda done: fail_or(done, f"Downstream of **{_nice(node)}** in the knowledge graph:\n"
                                         + "\n".join(f"- {r}" for r in _result(done, "what_moves")["downstream"])
                                         + _cav_block(done)))

    # overview, or anything the agent does not recognise
    def write(done):
        text = fail_or(done, _w_brief(_result(done, "brief")) + _cav_block(done))
        return text if known else HELP + "\n\nHere is the overview in the meantime.\n\n" + text
    return Plan([("brief", {})], write)
