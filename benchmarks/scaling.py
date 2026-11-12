"""The scaling benchmark: time and accuracy against the number of regions, gate against NUTS.

Simulates regional brands whose every region has its own carryover and saturation curve
(``make_synthetic_geo_mmm(G, seed, regional_curves=0.5)``, three channels, three years of
weeks), fits four models to each and grades them against the truth that generated the data:

    gate, shared curves       the 1.0 model and sampler: one curve per channel for all regions
    NUTS, shared curves       the same model on the JAX backend, to check the samplers agree
    gate, regional curves     a curve per region and channel, on the gate (small region counts only)
    NUTS, regional curves     a curve per region and channel, on the JAX backend

    python benchmarks/scaling.py --regions 4 8 25 50 100 200 400

Each fit is appended to ``scaling_results.jsonl`` as it finishes, so a long run can be
split over several commands (fits already in the file are skipped); ``--from-results``
rebuilds ``SCALING.md`` from the file. Fits run one at a time: the seconds are wall-clock
on the machine named in the report, with the library's default sampler settings.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MEDIA = ["search", "social", "video"]
SPREAD = 0.5
METHODS = {"gate, shared curves": ("gate", False), "NUTS, shared curves": ("nuts", False),
           "gate, regional curves": ("gate", True), "NUTS, regional curves": ("nuts", True)}


def truth_table(df: pd.DataFrame, truth: dict, l_max: int = 8) -> pd.DataFrame:
    """True return and marginal return (1% more spend in every week) per region and channel."""
    from papilon.mmm.transforms import geometric_adstock, hill
    rows = []
    for geo, part in df.groupby("region", sort=False):
        for c in MEDIA:
            t, x = truth["regional"][geo][c], part[c].to_numpy(dtype=float)
            contrib = lambda s: (t["max_effect"] * hill(geometric_adstock(s * x / x.mean(), t["alpha"], l_max),
                                                        t["half_sat"])).sum()
            rows.append({"region": geo, "channel": c, "roas": contrib(1.0) / x.sum(),
                         "mroas": (contrib(1.01) - contrib(1.0)) / (0.01 * x.sum())})
    return pd.DataFrame(rows)


def one_fit(n_regions: int, method: str, seed: int) -> dict:
    import papilon as pp
    sampler, regional = METHODS[method]
    df, truth = pp.make_synthetic_geo_mmm(n_geos=n_regions, seed=seed, regional_curves=SPREAD)
    true = truth_table(df, truth)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.perf_counter()
        m = pp.GeoMMM(df, "sales", MEDIA, "date", "region", seed=seed, sampler=sampler, regional_curves=regional)
        m.fit(n_starts=2)
        seconds = time.perf_counter() - start
    g = m.gate_results[m.family.name()]
    est = m.get_roi(by_geo=True)
    assert (est.region.to_numpy() == true.region.to_numpy()).all() and (est.channel.to_numpy() == true.channel.to_numpy()).all()
    nat = m.get_roi()
    nat_true = np.array([truth["national_roas"][c] for c in MEDIA])
    inside = lambda lo, hi, v: float(((lo <= v) & (v <= hi)).mean())
    fin = lambda v: None if not np.isfinite(v) else float(v)
    return {"regions": n_regions, "method": method, "seed": seed, "sampled_parameters": int(m._dim(m._fam)),
            "seconds": float(seconds), "verdict": g.verdict, "passed": bool(g.passed), "rhat": fin(g.rhat),
            "ess_bulk": fin(g.ess_bulk), "divergences": int(g.divergences),
            "national_roas_error": float(np.abs(nat.roas_mean.to_numpy() - nat_true).mean()),
            "national_roas_coverage": inside(nat.roas_p05.to_numpy(), nat.roas_p95.to_numpy(), nat_true),
            "regional_roas_error": float(np.abs(est.roas_mean - true.roas).mean()),
            "regional_roas_coverage": inside(est.roas_p05, est.roas_p95, true.roas),
            "regional_mroas_error": float(np.abs(est.mroas_mean - true.mroas).mean()),
            "regional_mroas_coverage": inside(est.mroas_p05, est.mroas_p95, true.mroas)}


def machine() -> dict:
    import papilon
    out = {"papilon": papilon.__version__, "python": platform.python_version(), "platform": platform.platform(),
           "cpus": os.cpu_count(), "numpy": np.__version__}
    for mod in ("jax", "numpyro"):
        try:
            out[mod] = __import__(mod).__version__
        except ImportError:
            out[mod] = None
    return out


def _time(s: float) -> str:
    return f"{s:.0f} s" if s < 120 else f"{s / 60:.0f} min"


def report(d: pd.DataFrame, info: dict) -> str:
    lines = ["# The scaling benchmark", "",
             "How do time and accuracy change with the number of regions, and where does the importance",
             "sampler (the gate) stop being enough? Every row is one fit to simulated regional data in which",
             f"each region has its own carryover and saturation curve (`make_synthetic_geo_mmm(G, seed=4,",
             f"regional_curves={SPREAD})`: three channels, 156 weeks), graded against the truth that generated it.",
             "",
             "- **gate, shared curves**: the 1.0 model and sampler. One curve per channel for all regions;",
             "  regional effects are integrated out, so the sampled parameters do not grow with regions.",
             "- **NUTS, shared curves**: the same model on the JAX backend. It should agree with the row above.",
             "- **gate, regional curves**: `regional_curves=True` on the gate, for small region counts only.",
             "- **NUTS, regional curves**: `regional_curves=True` on the JAX backend, the model 1.1 adds.",
             "",
             f"Machine: {info['cpus']} CPU cores, {info['platform']}, Python {info['python']}, Papilon {info['papilon']}, "
             f"JAX {info['jax']}, NumPyro {info['numpyro']}. No GPU. Default sampler settings (gate: 4,000 draws, two starts;",
             "NUTS: four chains, 1,000 warm-up and 1,000 kept draws each). Seconds are wall-clock for one fit, compilation",
             "included, with nothing else running. One simulated dataset per region count, so small differences in the",
             "accuracy columns at 4 and 8 regions (12 and 24 region-channel returns) are noise.",
             "",
             "| Regions | Model and sampler | Sampled parameters | Time | Fit check | Regional return: error | 90% range holds it "
             "| Regional marginal return: error | 90% range holds it | National return: error |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for n in sorted(d.regions.unique()):
        for method in METHODS:
            r = d[(d.regions == n) & (d.method == method)]
            if r.empty:
                continue
            r = r.iloc[0]
            if r.verdict.startswith("NUTS"):
                check = (f"{'passed' if r.passed else 'FAILED'} (R-hat {r.rhat:.3f}, ESS {r.ess_bulk:.0f}, "
                         f"{int(r.divergences)} divergent)")
            else:
                check = f"{r.verdict}" + ("" if r.passed else " (not passed)")
            lines.append(f"| {n} | {method} | {int(r.sampled_parameters):,} | {_time(r.seconds)} | {check} | "
                         f"{r.regional_roas_error:.2f} | {r.regional_roas_coverage:.0%} | {r.regional_mroas_error:.2f} | "
                         f"{r.regional_mroas_coverage:.0%} | {r.national_roas_error:.2f} |")
    lines += ["",
              "## Reading the columns", "",
              "- **Sampled parameters**: what the sampler explores. Baselines, noise and each region's effect sizes are",
              "  integrated out exactly in every row and are not counted.",
              "- **Fit check**: the gate reports ACCEPT (importance sampling passed its effective-sample-size check) or SMC",
              "  (it did not, and the tempered sequential Monte Carlo fallback ran). The fallback always reports a pass, so",
              "  read its accuracy columns. NUTS passes with every R-hat below 1.01, at least 400 effective draws for",
              "  every parameter and no divergent transitions.",
              "- **Regional return** and **regional marginal return**: mean absolute error of the estimated return on spend,",
              "  and of the return on 1% more spend, over every region and channel, and the share of their 90% ranges that",
              "  hold the truth (90% is calibrated). The marginal return is what a reallocation between regions rests on,",
              "  and it is where a region's own curve matters most.",
              "- **National return: error**: mean absolute error of the three national returns.",
              "",
              "Regenerate with `python benchmarks/scaling.py` (see the top of that file). Simulated regions are kinder than",
              "real ones: the model's functional forms are the generator's, spend varies independently by region, and",
              "nothing is missing from the data.",
              ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--regions", type=int, nargs="+", default=[4, 8, 25, 50, 100, 200, 400])
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--gate-regional-max", type=int, default=8,
                    help="largest region count at which regional curves are also fitted with the gate")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--from-results", action="store_true", help="rebuild SCALING.md from scaling_results.jsonl")
    a = ap.parse_args()
    res = HERE / "scaling_results.jsonl"
    done = set()
    if res.exists():
        done = {(r["regions"], r["method"]) for r in map(json.loads, res.read_text().splitlines()) if r}
    if not a.from_results:
        for n in a.regions:
            for method in a.methods:
                if (n, method) in done or (method == "gate, regional curves" and n > a.gate_regional_max):
                    continue
                r = one_fit(n, method, a.seed)
                with open(res, "a") as f:
                    f.write(json.dumps(r) + "\n")
                print(json.dumps(r), flush=True)
    d = pd.read_json(res, lines=True)
    info = machine()
    (HERE / "scaling.json").write_text(json.dumps({"machine": info, "spread": SPREAD,
                                                   "fits": json.loads(d.to_json(orient="records"))}, indent=2))
    (HERE / "SCALING.md").write_text(report(d, info))
    print((HERE / "SCALING.md").read_text())


if __name__ == "__main__":
    main()
