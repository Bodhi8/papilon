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
"""Charts and a one-file HTML report for a fitted model.

Every chart function returns a matplotlib Figure. ``report`` writes a self-contained
HTML page (charts embedded, no external files or scripts) with the fit's quality,
returns with their ranges, response curves, the decomposition of sales, a budget plan
and its stability, the knowledge graph audit if one is given, and every caution
:func:`papilon.ai.brief` raises, so the page can be sent to people who will never
run Python.
"""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["plot_fit", "plot_decomposition", "plot_roas", "plot_response_curves", "plot_plan", "plot_loop", "report"]

INK, MUTED, ACCENT, SOFT, WARN = "#1f2a33", "#6b7780", "#5b3f9e", "#b9a8de", "#d1603d"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    return plt


def plot_fit(model):
    """Actual against fitted outcome, with the share of variance explained."""
    plt = _plt()
    d = model.decompose()
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(d[model.date_col], d["actual"], color=MUTED, lw=1.2, label="actual")
    ax.plot(d[model.date_col], d["fitted"], color=ACCENT, lw=1.6, label="fitted")
    r2 = 1 - d["residual"].var() / d["actual"].var()
    ax.set_title(f"Fit: {r2:.0%} of the variation explained", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def plot_decomposition(model):
    """Outcome split into baseline, controls and channels over time (stacked)."""
    plt = _plt()
    d = model.decompose()
    parts = ["baseline"] + model.controls + model.channels
    fig, ax = plt.subplots(figsize=(9, 3.6))
    cmap = plt.get_cmap("Purples")
    ax.stackplot(d[model.date_col], *[d[p].clip(lower=0) for p in parts], labels=parts,
                 colors=[cmap(0.25 + 0.7 * i / max(len(parts) - 1, 1)) for i in range(len(parts))], alpha=0.9)
    ax.plot(d[model.date_col], d["actual"], color=INK, lw=0.8, label="actual")
    ax.set_title("What drove the outcome", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    fig.tight_layout()
    return fig


def plot_roas(model):
    """Return on spend per channel: posterior mean and 90% range."""
    plt = _plt()
    roi = model.get_roi().sort_values("roas_mean")
    fig, ax = plt.subplots(figsize=(6.5, 0.5 + 0.55 * len(roi)))
    y = np.arange(len(roi))
    ax.hlines(y, roi.roas_p05, roi.roas_p95, color=SOFT, lw=6)
    ax.plot(roi.roas_mean, y, "o", color=ACCENT)
    ax.axvline(1.0, color=MUTED, lw=0.8, ls="--")
    ax.set_yticks(y, roi.index)
    ax.set_xlabel("return on ad spend (mean and 90% range)")
    ax.set_title("Returns by channel", loc="left", color=INK)
    fig.tight_layout()
    return fig


def plot_response_curves(model, max_mult: float = 2.5):
    """Steady-state response per channel up to ``max_mult`` times its average spend, with 90% bands."""
    plt = _plt()
    k = len(model.media)
    cols = min(k, 3)
    fig, axes = plt.subplots(int(np.ceil(k / cols)), cols, figsize=(3.2 * cols, 2.6 * np.ceil(k / cols)), squeeze=False)
    for ax, c in zip(axes.flat, model.media):
        avg = float(model.spend_raw[:, model.media.index(c)].mean())
        lv = np.linspace(0, max_mult * avg, 40)
        rc = model.response_curve(c, lv) if type(model).__name__ == "MMM" else None
        if rc is None:
            j = model.media.index(c)
            xs = np.zeros((len(lv), len(model.media)))
            vals = np.stack([model._curve(np.eye(len(model.media))[j] * v / model.spend_scale[j])[:, j] for v in lv], 1)
            rc = pd.DataFrame({"spend": lv, "contribution_mean": vals.mean(0),
                               "contribution_p05": np.quantile(vals, 0.05, 0), "contribution_p95": np.quantile(vals, 0.95, 0)})
        ax.fill_between(rc.spend, rc.contribution_p05, rc.contribution_p95, color=SOFT, alpha=0.6, lw=0)
        ax.plot(rc.spend, rc.contribution_mean, color=ACCENT)
        ax.axvline(avg, color=MUTED, lw=0.8, ls="--")
        ax.set_title(c, fontsize=10, loc="left")
    for ax in list(axes.flat)[k:]:
        ax.axis("off")
    fig.suptitle("Response curves (dashed: average spend)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    return fig


def plot_plan(plan: pd.DataFrame):
    """Current against recommended spend per channel."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 0.5 + 0.55 * len(plan)))
    y = np.arange(len(plan))
    ax.barh(y + 0.18, plan.current_spend, height=0.34, color=SOFT, label="current")
    ax.barh(y - 0.18, plan.optimal_spend, height=0.34, color=ACCENT, label="recommended")
    ax.set_yticks(y, plan.index)
    ax.legend(frameon=False)
    s = plan.attrs.get("summary", {})
    ax.set_title(f"Budget plan: beats the current mix in {s.get('prob_better', float('nan')):.0%} of draws",
                 loc="left", color=INK)
    fig.tight_layout()
    return fig


def plot_loop(compare: pd.DataFrame):
    """Error and captured gain by cycle for each policy, from ``pycausalsim.futures.compare_loops``."""
    plt = _plt()
    s = compare.attrs["summary"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    colors = {"static": MUTED, "refresh": SOFT, "loop": ACCENT}
    for p, g in s.groupby("policy"):
        a1.plot(g.cycle, g.abs_error, marker="o", color=colors.get(p, INK), label=p)
        a2.plot(g.cycle, g.capture, marker="o", color=colors.get(p, INK), label=p)
    a1.set_title("Error in estimated return", loc="left"); a2.set_title("Share of best gain captured", loc="left")
    a1.set_xlabel("cycle"); a2.set_xlabel("cycle"); a1.legend(frameon=False)
    fig.tight_layout()
    return fig


def _img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return f'<img alt="" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _table(df: pd.DataFrame, digits: int = 2) -> str:
    return df.round(digits).to_html(border=0, classes="t", na_rep="")


def report(model, path, plan: pd.DataFrame | None = None, graph=None, title: str | None = None) -> Path:
    """Write a self-contained HTML report for ``model`` to ``path`` and return the path."""
    from . import __version__
    from .ai import brief
    model._require_fit()
    is_geo = type(model).__name__ == "GeoMMM"
    if plan is None:
        plan = model.optimize_budget(float(model.spend_raw.mean(axis=0).sum()))
    b = brief(model, plan=plan, graph=graph) if not is_geo else {"caveats": []}
    best = max(model.family_probabilities, key=model.family_probabilities.get)
    g = model.gate_results[best]
    sections = []
    sections.append(f"<h2>Fit quality</h2><p>Model family <b>{html.escape(best)}</b>; quality gate "
                    f"<b>{g.verdict}</b> ({g.method}, effective sample {g.ess:.0%}, {g.modes} mode"
                    f"{'s' if g.modes > 1 else ''}).</p>")
    if len(model.gate_results) > 1:
        fam = pd.DataFrame({k: {"log evidence": v.log_evidence, "probability": model.family_probabilities[k],
                                "verdict": v.verdict} for k, v in model.gate_results.items()}).T
        sections.append(_table(fam, 3))
    if b["caveats"]:
        sections.append("<h2>Cautions</h2><ul>" + "".join(f"<li>{html.escape(c)}</li>" for c in b["caveats"]) + "</ul>")
    if not is_geo:
        sections.append("<h2>Fit</h2>" + _img(plot_fit(model)))
        sections.append("<h2>Decomposition</h2>" + _img(plot_decomposition(model)))
    sections.append("<h2>Returns</h2>" + _img(plot_roas(model)) + _table(model.get_roi()))
    sections.append("<h2>Response curves</h2>" + _img(plot_response_curves(model)))
    sections.append("<h2>Budget plan</h2>" + _img(plot_plan(plan)) + _table(plan.drop(columns=[], errors="ignore")))
    try:
        st = model.plan_stability(float(plan.optimal_spend.sum()), n_splits=6)
        sections.append("<h3>Plan stability</h3>" + _table(st))
    except Exception:  # noqa: BLE001  stability is informative, never required for the report
        pass
    if graph is not None:
        sections.append("<h2>Knowledge graph</h2><pre>" + html.escape(graph.describe()) + "</pre>"
                        + _table(graph.audit(model.data if not is_geo else None), 0))
    head = title or f"{model.target}: marketing mix model report"
    css = ("body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#1f2a33;max-width:980px;margin:32px auto;"
           "padding:0 16px;line-height:1.5}h1{font-weight:600}h2{margin-top:2em;border-bottom:1px solid #e3e0ea;"
           "padding-bottom:4px}img{max-width:100%}table.t{border-collapse:collapse;font-size:13px;margin:8px 0}"
           "table.t td,table.t th{padding:4px 10px;border-bottom:1px solid #eee;text-align:right}"
           "table.t th:first-child,table.t td:first-child{text-align:left}pre{white-space:pre-wrap;font-size:13px}"
           ".meta{color:#6b7780;font-size:13px}")
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>{html.escape(head)}</title><style>{css}</style></head>"
            f"<body><h1>{html.escape(head)}</h1><p class='meta'>Papilon {__version__} · {len(model.data)} rows · "
            f"{len(model.media)} paid channels</p>{''.join(sections)}</body></html>")
    path = Path(path)
    path.write_text(page, encoding="utf-8")
    return path
