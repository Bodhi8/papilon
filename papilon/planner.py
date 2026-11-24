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
"""A no-code scenario planner for a fitted model, as one HTML file.

:func:`scenario_planner` writes a page that anyone can open in a browser, with no
Python, server or internet connection. It carries a sample of the model's posterior
response curves and plans budgets in the page itself:

* the best mix for a fixed budget, either protecting the downside (Papilon's
  default: the 10th percentile of the gain over the starting mix) or maximizing the
  expected outcome;
* a flexible budget: the most spend whose total ROI stays above a target, or each
  channel's spend up to where its marginal ROI falls to a target;
* a mix built by hand, with live results;
* channel limits, locked channels, a choice of starting period, saved scenarios
  compared side by side, CSV export and a link that reopens a scenario.

Every result carries its 90% interval, the chance it beats the starting mix, and a
flag when a channel would be pushed past the spend it has sustained before. The
model results (fit, contributions, returns with their ranges, cautions) sit on a
second tab.

From a terminal, without writing code::

    python -m papilon.planner --data sales.csv --target sales --date week \\
        --media search,social,video --controls price -o planner.html
    python -m papilon.planner --model fitted.papilon -o planner.html

:func:`planner_data` returns the same content as plain data, for building a hosted
front end on top of it.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["scenario_planner", "planner_data", "main"]


def _sig(x, digits: int = 6):
    """Round to significant digits for a compact page; NaN and infinity become None."""
    if isinstance(x, dict):
        return {k: _sig(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_sig(v, digits) for v in x]
    if isinstance(x, np.ndarray):
        return _sig(x.tolist(), digits)
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(f"{x:.{digits}g}") if np.isfinite(x) else None
    return x


def _period(dates: pd.Series) -> tuple[str, list[int]]:
    """Name of one period and the trailing windows worth offering as starting points."""
    days = float(pd.Series(pd.to_datetime(dates)).sort_values().diff().dt.days.median() or 7)
    if days <= 1.5:
        return "day", [365, 90, 30]
    if days <= 10:
        return "week", [52, 26, 13]
    if days <= 45:
        return "month", [12, 6, 3]
    return "period", []


def _components(model, idx: np.ndarray) -> dict:
    """Response curves of the paid channels for the posterior draws ``idx``.

    Channel j's steady-state contribution per period at spend x, draw d, is
    ``sum_g weights[j][d][g] * f(x * scales[j][g])`` with f the channel's saturation
    curve at ``half_sat[d][j]`` and ``shape[d][j]``: one component for a national
    model, one per region for :class:`papilon.GeoMMM`. Equal to ``model._steady``.
    """
    d, fam, K = model._draws, model._fam, len(model.media)
    weights, scales = [], []
    if type(model).__name__ == "GeoMMM":
        for j in range(K):
            s = model._share[:, j] / model._x_g[:, j]
            w = d["geo_beta"][idx][:, :, j] * model._y_g[None, :] * model.y_scale
            weights.append(w); scales.append(s)
    else:
        for j in range(K):
            mult = model._mult_now(d, j)[idx]
            weights.append((d["beta"][idx, j] * mult * model.y_scale)[:, None])
            scales.append(np.array([1.0 / model.spend_scale[j]]))
    return {"saturation": [fam.sat[c] for c in model.media], "weights": weights, "scales": scales,
            "half_sat": d["half_sat"][idx][:, :K], "shape": d["shape"][idx][:, :K]}


def _steady_from(comp: dict, x) -> np.ndarray:
    """Total contribution per draw at spend vector x, from :func:`_components` (reference for tests)."""
    total = 0.0
    for j, kind in enumerate(comp["saturation"]):
        hs, sh = comp["half_sat"][:, j][:, None], comp["shape"][:, j][:, None]
        a = float(x[j]) * comp["scales"][j][None, :]
        with np.errstate(all="ignore"):
            if kind == "logistic":
                e = np.exp(-a / hs); f = (1 - e) / (1 + e)
            elif kind == "hill":
                f = a / (a + hs)
            else:
                f = a ** sh / (a ** sh + hs ** sh)
        total = total + np.sum(comp["weights"][j] * np.nan_to_num(f), axis=1)
    return np.asarray(total)


def planner_data(model, n_draws: int = 500, title: str | None = None, currency: str = "",
                 units: str | None = None, horizon: int | None = None, seed: int | None = None) -> dict:
    """Everything the scenario planner shows, as JSON-ready data.

    ``n_draws`` posterior draws are sampled (without replacement) for the page's
    response curves; 500 keeps a page near 200 KB and its 5th and 95th percentiles
    within a few percent of the full posterior's. ``currency`` prefixes money (for
    example ``"$"``), ``units`` notes the scale (for example ``"thousands"``) and
    ``horizon`` sets how many periods the totals cover by default.
    """
    from . import __version__
    model._require_fit()
    is_geo = type(model).__name__ == "GeoMMM"
    rng = np.random.default_rng(model.seed + 97 if seed is None else seed)
    n = len(model._draws["beta"])
    idx = np.sort(rng.choice(n, min(n_draws, n), replace=False))
    comp = _components(model, idx)

    dates = pd.to_datetime(pd.Series(model.dates if is_geo else model.data[model.date_col]))
    period, lengths = _period(dates)
    spend = model.spend_raw
    T = len(spend)
    windows = [{"label": f"Average of all {T} {period}s", "periods": T, "spend": spend.mean(axis=0)}]
    for L in lengths:
        if L < T:
            windows.append({"label": f"Average of the last {L} {period}s", "periods": L, "spend": spend[-L:].mean(axis=0)})
    hist = spend.mean(axis=0)
    total = float(hist.sum())
    _, _, support_high = model._plan_limits(total, None, None, "flag", 0.95)

    plan = model.optimize_budget(total)
    s = plan.attrs["summary"]
    roi = model.get_roi()
    out = {
        "papilon_version": __version__,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "title": title or f"{model.target}: scenario planner",
        "target": model.target, "period": period, "currency": currency, "units": units or "",
        "horizon": int(horizon or {"week": 13, "day": 90, "month": 3}.get(period, 4)),
        "kind": "regional" if is_geo else "national",
        "n_periods": T, "first_date": str(dates.min().date()), "last_date": str(dates.max().date()),
        "media": list(model.media), "n_draws": int(len(idx)),
        "historical_spend": hist, "scale": np.asarray(model.spend_scale, dtype=float),
        "support_high": np.maximum(support_high, 0.0),
        "windows": windows,
        "curves": {"saturation": comp["saturation"], "scales": comp["scales"],
                   "weights": [w for w in comp["weights"]], "half_sat": comp["half_sat"], "shape": comp["shape"]},
        "recommended": {"spend": plan.optimal_spend.to_numpy(), "start": plan.current_spend.to_numpy(),
                        "expected_lift": s["expected_lift"], "lift_p05": s["lift_p05"], "lift_p95": s["lift_p95"],
                        "prob_better": s["prob_better"]},
        "indistinguishable": [list(p) for p in s.get("indistinguishable", [])],
        "roi": [{"channel": c, "spend": r.spend, "roas_mean": r.roas_mean, "roas_p05": r.roas_p05,
                 "roas_p95": r.roas_p95, "mroas_mean": r.mroas_mean, "share_of_sales": r.share_of_sales}
                for c, r in roi.iterrows()],
    }
    best = max(model.family_probabilities, key=model.family_probabilities.get)
    g = model.gate_results[best]
    out["gate"] = {"family": best, "verdict": g.verdict, "method": g.method, "ess": g.ess, "passed": g.passed}
    if is_geo:
        out["regions"] = [str(x) for x in model.geos]
        out["cautions"] = []
    else:
        from .ai import brief
        out["cautions"] = list(brief(model, plan=plan)["caveats"])
        dec = model.decompose()
        act, fit = dec["actual"].to_numpy(), dec["fitted"].to_numpy()
        nz = np.abs(act) > 1e-12
        out["fit"] = {"dates": [str(x.date()) for x in pd.to_datetime(dec[model.date_col])],
                      "actual": act, "fitted": fit,
                      "r2": 1 - np.var(act - fit) / np.var(act),
                      "mape": float(np.mean(np.abs((act - fit)[nz] / act[nz]))),
                      "wmape": float(np.sum(np.abs(act - fit)) / np.sum(np.abs(act)))}
        parts = ["baseline"] + list(model.controls) + list(model.channels)
        out["decomposition"] = [{"name": p, "total": float(dec[p].sum()),
                                 "kind": "baseline" if p == "baseline" else ("control" if p in model.controls else
                                         ("paid" if p in model.media else "organic"))} for p in parts]
    return _sig(out)


def _asset(name: str) -> str:
    return resources.files("papilon").joinpath("planner_assets", name).read_text(encoding="utf-8")


def scenario_planner(model, path, title: str | None = None, n_draws: int = 500, currency: str = "",
                     units: str | None = None, horizon: int | None = None) -> Path:
    """Write the no-code scenario planner for ``model`` to ``path`` (one HTML file) and return the path.

    The page needs no Python, server or network: open it in a browser, send it, or
    host it as a static file. See the module docstring for what it does, and
    :func:`planner_data` for the arguments.
    """
    data = planner_data(model, n_draws=n_draws, title=title, currency=currency, units=units, horizon=horizon)
    blob = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    page = (_asset("planner.html")
            .replace("{{TITLE}}", html.escape(data["title"]))
            .replace("/*{{STYLE}}*/", _asset("planner.css"))
            .replace("/*{{ENGINE}}*/", _asset("engine.js"))
            .replace("/*{{UI}}*/", _asset("ui.js"))
            .replace("{{DATA}}", blob))
    path = Path(path)
    path.write_text(page, encoding="utf-8")
    return path


def main(argv=None) -> None:
    """Command line: fit a model to a CSV (or open a saved one) and write the planner."""
    import argparse

    ap = argparse.ArgumentParser(prog="python -m papilon.planner",
                                 description="Write a no-code scenario planner (one HTML file) for a marketing mix model.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="a model saved with papilon.persist.save_model")
    src.add_argument("--data", help="CSV with one row per period")
    ap.add_argument("--target"); ap.add_argument("--date")
    ap.add_argument("--media", help="comma-separated spend columns")
    ap.add_argument("--controls", default=""); ap.add_argument("--delayed", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", default="planner.html")
    ap.add_argument("--title"); ap.add_argument("--currency", default="")
    ap.add_argument("--units", help='scale of money columns, e.g. "thousands"')
    ap.add_argument("--horizon", type=int, help="periods the totals cover (default: a quarter)")
    ap.add_argument("--draws", type=int, default=500)
    a = ap.parse_args(argv)
    if a.model:
        from .persist import load_model
        model = load_model(a.model)
    else:
        if not (a.target and a.date and a.media):
            ap.error("--data needs --target, --date and --media")
        from .mmm import MMM
        split = lambda s: [x for x in s.split(",") if x]
        df = pd.read_csv(a.data, parse_dates=[a.date])
        model = MMM(df, a.target, split(a.media), a.date, controls=split(a.controls),
                    delayed=split(a.delayed), seed=a.seed).fit()
    out = scenario_planner(model, a.out, title=a.title, n_draws=a.draws, currency=a.currency,
                           units=a.units, horizon=a.horizon)
    print(f"wrote {out}: open it in any browser")


if __name__ == "__main__":  # pragma: no cover
    main()
