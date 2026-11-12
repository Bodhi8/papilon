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
"""The learning-loop benchmark: does an always-on model converge on the truth?

Runs four policies on the same simulated brands for six quarters each, after two years
of history, and writes LOOP.md, loop_results.jsonl and loop.png next to this file.

    python benchmarks/loop.py --worlds 8 --jobs 2       (about an hour on two cores)

Policies:
    static          fit once on the history and keep that plan
    refresh         refit every quarter on all the data, no tests, no exploration
    loop            the learning loop, cost-aware tests (the default)
    loop_learning   the learning loop with a test every quarter (payback_periods very large)
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CYCLES, HISTORY, HORIZON = 6, 104, 13


def _job(args):
    hard, i, policy = args
    from pycausalsim.futures import marketing_worlds, simulate_loop
    worlds = marketing_worlds(12, weeks=HISTORY + CYCLES * HORIZON, hard=hard, seed=7)
    base = "loop" if policy.startswith("loop") else policy
    loop_kwargs = {"payback_periods": 10 ** 9} if policy == "loop_learning" else None   # test whenever informative
    r = simulate_loop(worlds, i, base, history=HISTORY, cycles=CYCLES, horizon=HORIZON,
                      model_kwargs={"delayed": ["online_video"]}, loop_kwargs=loop_kwargs, seed=3)
    r["policy"], r["hard"] = policy, hard
    return r.to_dict("records")


def summarize(d: pd.DataFrame) -> str:
    lines = ["# The learning-loop benchmark", "",
             "Does a marketing mix model that is refit every quarter converge on the truth, or lock in its",
             "own error? Each policy below ran for six quarters on the same simulated brands",
             f"(`marketing_worlds(12, seed=7)`, first {d.world.nunique()} per track), after two years of history. The",
             "world responds to the spend each policy chose; recommended tests run in the world and come",
             "back with the noise of a matched-market design (half the market held out, 90% of noise",
             "removed by the pre-period model).", "",
             "- **static**: fit once and keep the plan.",
             "- **refresh**: refit every quarter on all the data. No tests, no exploration.",
             "- **loop**: `papilon.Loop` defaults. Refit, a budget-neutral exploration pulse of 10% between the",
             "  two channels the model can least tell apart, and a lift test only when its expected gain to the",
             "  plan over a year exceeds the sales the holdout gives up.",
             "- **loop_learning**: the loop with a test whenever one is informative, whatever it costs.", ""]
    names = {False: "Standard brands", True: "Hard brands (unobserved launches, creative wear-out)"}
    for hard, g in d.groupby("hard"):
        s = g.groupby(["policy", "cycle"]).agg(err=("abs_error", "mean"), gain=("plan_gain", "sum"),
                                               best=("best_gain", "sum"), lost=("plan_gain", lambda v: (v < -1e-6).mean()))
        s["cap"] = s.gain / s.best
        last, first = s.xs(g.cycle.max(), level="cycle"), s.xs(0, level="cycle")
        tot = g.groupby(["policy", "world"]).realized.sum().unstack("policy")
        rel = 100 * tot.sub(tot["static"], axis=0).mean() / tot["static"].mean()
        tests = g.groupby("policy").tested.apply(lambda t: t.notna().mean())
        lines += [f"## {names[hard]}", "",
                  "| Policy | Error in returns, quarter 1 → 6 | Best gain captured, quarter 1 → 6 | Plans losing money, quarter 6 | Sales over six quarters vs static | Quarters with a test |",
                  "|---|---|---|---|---|---|"]
        for p in ["static", "refresh", "loop", "loop_learning"]:
            if p not in last.index:
                continue
            lines.append(f"| {p} | {first.loc[p, 'err']:.2f} → {last.loc[p, 'err']:.2f} | {first.loc[p, 'cap']:.0%} → "
                         f"{last.loc[p, 'cap']:.0%} | {last.loc[p, 'lost']:.0%} | {rel[p]:+.2f}% | {tests.get(p, 0):.0%} |")
        lines.append("")
    lines += ["## Reading it", "",
              "Refitting alone is unreliable. On standard brands its error grew as plans narrowed the data; on",
              "hard brands it improved, but less than the loop. The loop's tests and exploration add the",
              "information that moves estimates toward the truth and plans toward the best available.",
              "Learning has a price: testing every quarter learns fastest and costs sales while it does. The",
              "default tests only when a test is expected to pay for itself, which roughly breaks even on",
              "sales over six quarters while its plans improve. `payback_periods` sets the trade-off.", "",
              f"Eight brands per track and six quarters is a small sample; differences of a few points are noise.",
              "Simulated brands are kinder than real ones. Reproduce with `python benchmarks/loop.py`."]
    return "\n".join(lines) + "\n"


def chart(d: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"static": "#9aa3ab", "refresh": "#b9a8de", "loop": "#5b3f9e", "loop_learning": "#d1603d"}
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2), sharex=True)
    for row, (hard, g) in enumerate(d.groupby("hard")):
        s = g.groupby(["policy", "cycle"]).agg(err=("abs_error", "mean"), gain=("plan_gain", "sum"), best=("best_gain", "sum"))
        s["cap"] = s.gain / s.best
        for p, sp in s.groupby(level="policy"):
            axes[row, 0].plot(sp.index.get_level_values("cycle") + 1, sp.err, marker="o", color=colors.get(p), label=p)
            axes[row, 1].plot(sp.index.get_level_values("cycle") + 1, sp.cap, marker="o", color=colors.get(p), label=p)
        axes[row, 0].set_title(("Hard" if hard else "Standard") + " brands: error in estimated returns", loc="left", fontsize=10)
        axes[row, 1].set_title(("Hard" if hard else "Standard") + " brands: share of best gain captured", loc="left", fontsize=10)
        axes[row, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlabel("quarter"); axes[1, 1].set_xlabel("quarter")
    axes[0, 1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--from-results", action="store_true", help="rebuild LOOP.md and loop.png from loop_results.jsonl")
    a = ap.parse_args()
    res = HERE / "loop_results.jsonl"
    if not a.from_results:
        jobs = [(h, i, p) for h in (False, True) for i in range(a.worlds)
                for p in ("static", "refresh", "loop", "loop_learning")]
        with Pool(a.jobs) as pool, open(res, "w") as f:
            for recs in pool.imap_unordered(_job, jobs):
                for r in recs:
                    f.write(json.dumps(r, default=str) + "\n")
    d = pd.read_json(res, lines=True)
    (HERE / "LOOP.md").write_text(summarize(d))
    chart(d, HERE / "loop.png")
    print((HERE / "LOOP.md").read_text())


if __name__ == "__main__":
    main()
