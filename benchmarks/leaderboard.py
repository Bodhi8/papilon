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
"""The open marketing mix model leaderboard.

Scores each entry on the same simulated brands with known returns and writes
``LEADERBOARD.md`` and ``leaderboard.json`` next to this file.

    python benchmarks/leaderboard.py --worlds 16 --jobs 2

Tracks (all from ``pycausalsim.futures.marketing_worlds(60, seed=7, ...)``, the
first ``--worlds`` brands of each):

    standard   four channels, search spend follows demand, launches visible in spend
    hard       adds unobserved launches and creative wear-out
    s_curve    every channel's true response is S-shaped

To add an entry, write a function ``entry(df, info) -> {"roas": ..., "plan": ...}``
(see ``pycausalsim.futures.grading``), add it to ``ENTRIES`` and run this script.
``SUBMITTING.md`` has the rules.
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TRACKS = {"standard": {}, "hard": {"hard": True}, "s_curve": {"response": "s_curve"}}
FAMILIES = {"all_geometric": [], "video_delayed": ["online_video"]}


def papilon_default(df, info):
    from pycausalsim.futures import papilon_estimator
    return papilon_estimator(carryover=FAMILIES)(df, info)


def papilon_shape_chosen(df, info):
    """Papilon with the saturation shape chosen by Bayes factor."""
    import papilon as pp
    ch = info["media"]
    m = pp.MMM(df, target=info["target"], media=ch, date_col=info["date_col"], controls=info["controls"], seed=11)
    m.compare_models({"concave": {"delayed": ["online_video"]},
                      "s_curve": {"delayed": ["online_video"], "s_curve": ch}})
    roi = m.get_roi()
    return {"roas": {c: (roi.loc[c, "roas_mean"], roi.loc[c, "roas_p05"], roi.loc[c, "roas_p95"]) for c in ch},
            "plan": m.optimize_budget(info["total"])["optimal_spend"].to_numpy()}


def least_squares(df, info):
    from pycausalsim.futures import ols_estimator
    return ols_estimator()(df, info)


# name -> (function, tracks it enters, one-line description)
ENTRIES = {
    "Papilon 0.5, defaults": (papilon_default, ["standard", "hard", "s_curve"],
                              "carryover chosen by Bayes factor, downside-protected plan"),
    "Papilon 0.5, curve shape chosen": (papilon_shape_chosen, ["s_curve"],
                                        "also chooses concave or S-shaped saturation by Bayes factor"),
    "Least squares, fixed adstock": (least_squares, ["standard", "hard", "s_curve"],
                                     "reference baseline: regression on adstocked spend, no saturation"),
}


def _one(job):
    track, name, i = job
    from pycausalsim.futures import grade, marketing_worlds
    worlds = marketing_worlds(60, seed=7, **TRACKS[track])
    g = grade(ENTRIES[name][0], worlds, ids=[i])
    return track, name, g.to_dict("records")


def summarize(rows: pd.DataFrame) -> dict:
    plans = rows.drop_duplicates("world")
    return {"brands": int(rows.world.nunique()),
            "coverage_90": float(rows.covered.mean()),
            "abs_error": float(rows.error.abs().mean()),
            "bias": float(rows.error.mean()),
            "plan_capture": float(plans.plan_gain.sum() / plans.best_gain.sum()),
            # a plan that leaves the mix unchanged scores a rounding-error "loss"; do not count it
            "plan_lost_money": float((plans.plan_gain < -1e-6 * plans.best_gain.clip(lower=1.0)).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--worlds", type=int, default=16)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    jobs = [(t, n, i) for n, (_, tracks, _) in ENTRIES.items() for t in tracks for i in range(a.worlds)]
    out: dict = {}
    with Pool(a.jobs) as pool:
        for track, name, recs in pool.imap_unordered(_one, jobs):
            out.setdefault(track, {}).setdefault(name, []).extend(recs)
    import papilon, pycausalsim
    result = {"papilon": papilon.__version__, "pycausalsim": pycausalsim.__version__, "seed": 7,
              "brands_per_track": a.worlds,
              "tracks": {t: {n: summarize(pd.DataFrame(r)) for n, r in e.items()} for t, e in out.items()}}
    (HERE / "leaderboard.json").write_text(json.dumps(result, indent=2))
    lines = ["# Open marketing mix model leaderboard", "",
             "Every entry is fitted to the same simulated brands, whose true returns are known, and scored on",
             "what a marketer needs from a model: ranges that hold the truth, and budget plans that make money.",
             f"Brands: the first {a.worlds} of `marketing_worlds(60, seed=7, ...)` per track, PyCausalSim "
             f"{pycausalsim.__version__}. Regenerate with `python benchmarks/leaderboard.py`; to add an entry see",
             "[SUBMITTING.md](SUBMITTING.md).", ""]
    notes = {"standard": "Search spend follows demand; launches are visible in spend.",
             "hard": "Adds unobserved launches that lift spend and sales together, and creative wear-out.",
             "s_curve": "Every channel's true response is S-shaped."}
    for t in TRACKS:
        if t not in result["tracks"]:
            continue
        lines += [f"## Track: {t}", "", notes[t], "",
                  "| Entry | 90% range holds the truth | ROAS error | Bias | Best gain captured | Plans that lost money |",
                  "|---|---|---|---|---|---|"]
        rows = sorted(result["tracks"][t].items(), key=lambda kv: -kv[1]["plan_capture"])
        for n, s in rows:
            lines.append(f"| {n} | {s['coverage_90']:.0%} | {s['abs_error']:.2f} | {s['bias']:+.2f} | "
                         f"{s['plan_capture']:.0%} | {s['plan_lost_money']:.0%} |")
        lines.append("")
    lines += ["## Reading the columns", "",
              "- **90% range holds the truth**: share of channel ranges containing the true return. 90% is calibrated.",
              "- **ROAS error** and **Bias**: mean absolute and mean signed error of the estimate.",
              "- **Best gain captured**: true gain of the recommended plans as a share of the best possible gain,",
              "  with each channel free between zero and three times its history. Can be negative.",
              "- **Plans that lost money**: share of brands where the recommended plan did worse than the current mix.",
              "", "## Entries", ""]
    lines += [f"- **{n}**: {d}." for n, (_, _, d) in ENTRIES.items()]
    lines += ["", "Simulated brands are kinder than real data in ways the generator does not capture. A good score",
              "here is necessary, not sufficient."]
    (HERE / "LEADERBOARD.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
