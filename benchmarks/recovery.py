"""Recovery benchmark: grade Papilon on many simulated brands with known answers.

Worlds come from PyCausalSim's temporal causal model (pycausalsim.futures.marketing),
one brand per draw. For each brand this fits Papilon three ways and plans the same
weekly budget six ways, then scores everything against the answer key:

fits   default         Beta(2, 2) carryover prior (Papilon 0.3 behavior)
       short_carry     Beta(1.5, 3) carryover prior, which favors shorter carryover
       lift_search     default prior plus a simulated geo lift test on paid search
                       (true ROAS plus noise with sd 0.2, entered as a prior with sd 0.25)

plans  mean            maximize expected sales, each channel 0 to 3x history (0.3 default)
       cap25           same, each channel within 25% of current spend
       quantile        maximize the 10th percentile of the gain
       quantile_cap25  both
       quantile25      maximize the 25th percentile of the gain
       support         mean, capped at sustained historical spend
       cap25_support   cap25 plus the sustained-spend cap

Usage: python benchmarks/recovery.py --worlds 60 --out results.jsonl [--hard] [--jobs 2]
Needs pycausalsim (>= 0.4) on the path.
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np

import papilon as pp
from pycausalsim.futures.marketing import CHANNELS, compare_plans, marketing_worlds, media_plan_scm

PLANS = {
    "mean": dict(),
    "cap25": dict(max_change=0.25),
    "quantile": dict(objective="quantile"),
    "quantile_cap25": dict(objective="quantile", max_change=0.25),
    "quantile25": dict(objective="quantile", quantile=0.25),
    "support": dict(support="limit"),
    "cap25_support": dict(max_change=0.25, support="limit"),
}


def _truth_curves(t):
    ch = t["channels"]
    return {"media": CHANNELS,
            "beta": np.array([[ch[c]["beta_final"] for c in CHANNELS]]),
            "half_sat": np.array([[ch[c]["half_sat"] for c in CHANNELS]]),
            "alpha": np.array([[ch[c]["alpha"] for c in CHANNELS]]),
            "peak": np.array([[np.nan if ch[c]["peak"] is None else ch[c]["peak"] for c in CHANNELS]]),
            "l_max": 8}


def one_world(args):
    i, n, hard, seed = args
    W = marketing_worlds(n, hard=hard, seed=seed)
    df, truth = W.frame(i), W.truth(i)
    hist = df[CHANNELS].mean().to_numpy()
    total = hist.sum()
    bounds = [(0.0, 3 * h) for h in hist]
    best = W.best_plan(i, total, bounds)
    true_gain = lambda x: W.steady_sales(i, x) - W.steady_sales(i, hist)
    best_gain = true_gain(best)
    rng = np.random.default_rng(1000 + i)
    measured = truth["channels"]["paid_search"]["roas"] + rng.normal(0, 0.2)
    fits = {"default": {}, "short_carry": {"alpha_prior": (1.5, 3.0)},
            "lift_search": {"roi_priors": {"paid_search": (measured, 0.25)}}}
    out = {"world": i, "hard": hard, "best_gain": best_gain, "fits": {}}
    true_scm = media_plan_scm(_truth_curves(truth), hist, weeks=13, draws=1)
    for fname, kw in fits.items():
        t0 = time.time()
        m = pp.MMM(df, target="sales", media=CHANNELS, date_col="date",
                   controls=["price_index", "promo"], yearly_seasonality=2, seed=11, **kw)
        fam = m.compare_carryover({"all_geometric": [], "video_delayed": ["online_video"]})
        roi = m.get_roi()
        rec = {"seconds": time.time() - t0, "video_delayed_prob": float(fam.loc["video_delayed", "probability"]),
               "pareto_k": {k: g.pareto_k for k, g in m.gate_results.items()}, "roas": {}, "plans": {}}
        for c in CHANNELS:
            tr = truth["channels"][c]["roas"]
            rec["roas"][c] = {"est": float(roi.loc[c, "roas_mean"]), "p05": float(roi.loc[c, "roas_p05"]),
                              "p95": float(roi.loc[c, "roas_p95"]), "true": tr,
                              "mroas_est": float(roi.loc[c, "mroas_mean"]),
                              "mroas_true": truth["channels"][c]["marginal_roas"]}
        model_scm = media_plan_scm(m.posterior_curves(), hist, weeks=13, draws=1000, seed=i)
        for pname, pkw in PLANS.items():
            plan = m.optimize_budget(total, **pkw)
            x = plan["optimal_spend"].to_numpy()
            s = plan.attrs["summary"]
            q_model = compare_plans(model_scm, {"current": hist, "plan": x}).loc["plan"]
            q_true = compare_plans(true_scm, {"current": hist, "plan": x}).loc["plan", "gain_vs_first"]
            rec["plans"][pname] = {"spend": x.tolist(), "promised": s["expected_lift"],
                                   "p05": s["lift_p05"], "p95": s["lift_p95"], "prob_better": s["prob_better"],
                                   "true_gain": true_gain(x), "moved": float(np.abs(x - hist).sum() / 2),
                                   "flagged": int(plan["outside_support"].sum()),
                                   "q13_promised": float(q_model["gain_vs_first"]),
                                   "q13_p05": float(q_model["gain_p05"]), "q13_p95": float(q_model["gain_p95"]),
                                   "q13_true": float(q_true)}
        out["fits"][fname] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=60)
    ap.add_argument("--hard", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default="recovery.jsonl")
    a = ap.parse_args()
    jobs = [(i, a.worlds, a.hard, a.seed) for i in range(a.worlds)]
    with open(a.out, "a") as f, Pool(a.jobs) as pool:
        for r in pool.imap_unordered(one_world, jobs):
            f.write(json.dumps(r) + "\n"); f.flush()
            print("world", r["world"], "done", flush=True)


if __name__ == "__main__":
    main()
