"""The no-code scenario planner: its curves equal the model's, its optimizer matches Papilon's,
and the page it writes is one self-contained file."""
import json
import re
import shutil
import subprocess
from importlib import resources

import numpy as np
import pytest

import papilon as pp
from papilon import planner as P
from papilon.persist import save_model

MEDIA = ["paid_search", "paid_social", "online_video", "display"]
NODE = shutil.which("node")


@pytest.fixture(scope="module")
def data():
    return pp.make_synthetic_mmm()


@pytest.fixture(scope="module")
def fitted(data):
    df, _ = data
    return pp.MMM(df, "sales", MEDIA, "date", controls=["price_index", "promo"], delayed=["online_video"], seed=11).fit(n_starts=2)


@pytest.fixture(scope="module")
def geo():
    df, _ = pp.make_synthetic_geo_mmm()
    return pp.GeoMMM(df, "sales", ["search", "social", "video"], "date", "region", seed=1).fit(n_starts=2)


def _subset(model, idx):
    n = len(model._draws["beta"])
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v) for k, v in model._draws.items()}


@pytest.mark.parametrize("which", ["fitted", "geo"])
def test_page_curves_equal_the_models_steady_state(which, request):
    model = request.getfixturevalue(which)
    idx = np.arange(0, len(model._draws["beta"]), 5)
    comp = P._components(model, idx)
    sub = _subset(model, idx)
    hist = model.spend_raw.mean(axis=0)
    for x in (hist, hist * np.linspace(0.3, 2.2, len(hist)), np.zeros_like(hist)):
        ref = np.sum(model._curve(x / model.spend_scale, P=sub), axis=-1) * model.y_scale
        assert np.allclose(P._steady_from(comp, x), ref, rtol=1e-9, atol=1e-9)


def test_planner_data_is_json_and_complete(fitted):
    d = P.planner_data(fitted, n_draws=200, currency="$", units="thousands")
    text = json.dumps(d, allow_nan=False)
    assert len(text) < 400_000
    assert d["media"] == MEDIA and d["period"] == "week" and d["n_draws"] == 200
    assert [w["periods"] for w in d["windows"]] == [156, 52, 26, 13]
    assert len(d["curves"]["half_sat"]) == 200 and len(d["curves"]["weights"]) == 4
    assert np.allclose(d["historical_spend"], fitted.spend_raw.mean(axis=0), rtol=1e-5)
    assert d["fit"]["r2"] > 0.5 and d["cautions"] and d["gate"]["passed"]
    assert {p["kind"] for p in d["decomposition"]} == {"baseline", "control", "paid"}
    assert np.isclose(sum(d["recommended"]["spend"]), sum(d["historical_spend"]), rtol=1e-5)


def test_page_is_one_self_contained_file(fitted, tmp_path):
    p = pp.scenario_planner(fitted, tmp_path / "planner.html", title="Brand </script> planner")
    page = p.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>")
    assert "{{" not in page and "/*{{" not in page
    assert not re.search(r"<script[^>]+src=", page) and not re.search(r"<link[^>]+href=", page)
    assert "Brand &lt;/script&gt; planner" in page
    blob = re.search(r'<script type="application/json" id="papilon-data">(.*?)</script>', page, re.S).group(1)
    assert json.loads(blob)["title"] == "Brand </script> planner"
    assert 50_000 < len(page) < 600_000


def test_geo_page(geo, tmp_path):
    d = P.planner_data(geo, n_draws=100)
    assert d["kind"] == "regional" and "fit" not in d and len(d["curves"]["scales"][0]) == len(geo.geos)
    pp.scenario_planner(geo, tmp_path / "geo.html")


def test_command_line(fitted, data, tmp_path, capsys):
    save_model(fitted, tmp_path / "m.papilon")
    P.main(["--model", str(tmp_path / "m.papilon"), "-o", str(tmp_path / "a.html"), "--currency", "$"])
    assert (tmp_path / "a.html").exists() and "wrote" in capsys.readouterr().out
    df, _ = data
    df.to_csv(tmp_path / "d.csv", index=False)
    P.main(["--data", str(tmp_path / "d.csv"), "--target", "sales", "--date", "date",
            "--media", ",".join(MEDIA), "-o", str(tmp_path / "b.html")])
    assert (tmp_path / "b.html").exists()
    with pytest.raises(SystemExit):
        P.main(["--data", str(tmp_path / "d.csv"), "-o", str(tmp_path / "c.html")])


HARNESS = """
const E = require(process.argv[2]);
const p = JSON.parse(require("fs").readFileSync(process.argv[3], "utf8"));
const m = new E.Model(p), start = p.historical_spend, total = start.reduce((a, b) => a + b, 0);
const c = {maxChange: 0.5, cap: 3}, free = {maxChange: null, cap: 3};
const out = {
  mean: E.plan(m, {mode: "fixed", start, total, objective: "mean", constraints: c}),
  quantile: E.plan(m, {mode: "fixed", start, total, objective: "quantile", quantile: 0.1, constraints: c}),
  bigger: E.plan(m, {mode: "fixed", start, total: total * 1.3, objective: "quantile", quantile: 0.1, constraints: c}),
  roi: E.plan(m, {mode: "target_roi", start, target: 1.0, constraints: free}),
  roi_high: E.plan(m, {mode: "target_roi", start, target: 50, constraints: free}),
  mroi: E.plan(m, {mode: "target_mroi", start, target: 0.5, constraints: free}),
  locked: E.plan(m, {mode: "fixed", start, total, objective: "mean", constraints: {maxChange: 0.5, cap: 3, locked: [true, false, false, false], lockedAt: start}}),
};
out.eval = E.evaluate(m, out.quantile.spend, start);
out.curve = E.curve(m, 2, start[2] * 2, 10);
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(NODE is None, reason="needs Node.js to run the page's engine")
@pytest.mark.parametrize("which", ["fitted", "geo"])
def test_page_optimizer_matches_papilons(which, request, tmp_path):
    model = request.getfixturevalue(which)
    d = P.planner_data(model, n_draws=400)
    (tmp_path / "p.json").write_text(json.dumps(d))
    (tmp_path / "h.js").write_text(HARNESS)
    engine = str(resources.files("papilon").joinpath("planner_assets", "engine.js"))
    r = json.loads(subprocess.check_output([NODE, str(tmp_path / "h.js"), engine, str(tmp_path / "p.json")], timeout=120))

    comp = {"saturation": d["curves"]["saturation"], "half_sat": np.array(d["curves"]["half_sat"]),
            "shape": np.array(d["curves"]["shape"]), "weights": [np.array(w) for w in d["curves"]["weights"]],
            "scales": [np.array(s) for s in d["curves"]["scales"]]}
    st = lambda x: P._steady_from(comp, np.asarray(x, dtype=float))
    hist = np.array(d["historical_spend"]); total = hist.sum()

    # on the same draws, the page's plans do at least as well as Papilon's own optimizer
    py = model.optimize_budget(total, objective="mean").optimal_spend.to_numpy()
    assert np.mean(st(r["mean"]["spend"])) >= np.mean(st(py)) - 1e-4 * abs(np.mean(st(py)))
    for key, tot in (("quantile", total), ("bigger", 1.3 * total)):
        py = model.optimize_budget(tot).optimal_spend.to_numpy()
        base = st(hist / hist.sum() * tot)
        q = lambda x: np.quantile(st(x) - base, 0.10)
        assert q(r[key]["spend"]) >= q(py) - 0.02 * abs(q(py)), key
        assert np.isclose(sum(r[key]["spend"]), tot, rtol=1e-6)
        assert np.all(np.array(r[key]["spend"]) >= 0.5 * hist / hist.sum() * tot - 1e-6)
        assert np.all(np.array(r[key]["spend"]) <= 1.5 * hist / hist.sum() * tot + 1e-6)

    roi_spend = np.array(r["roi"]["spend"])
    assert np.mean(st(roi_spend)) / roi_spend.sum() >= 1.0 - 0.01 and roi_spend.sum() > total
    assert not r["roi_high"]["feasible"] and r["roi_high"]["notes"]
    assert np.isclose(r["locked"]["spend"][0], hist[0]) and np.isclose(sum(r["locked"]["spend"]), total)

    # each channel stops where its marginal return falls to the target (or at a limit)
    for j, x in enumerate(r["mroi"]["spend"]):
        if 1e-9 < x < 3 * hist[j] - 1e-6:
            e = np.zeros_like(hist); e[j] = 1e-3 * hist[j]
            base = np.array(r["mroi"]["spend"])
            mr = (np.mean(st(base + e)) - np.mean(st(base))) / e[j]
            assert abs(mr - 0.5) < 0.05

    ev = r["eval"]
    assert np.isclose(ev["contribution"], np.mean(st(r["quantile"]["spend"])), rtol=1e-9)
    assert 0 <= ev["prob_better"] <= 1 and ev["lift_p05"] <= ev["lift"] <= ev["lift_p95"]
    assert r["curve"][0]["mean"] == 0 and all(a["mean"] <= b["mean"] for a, b in zip(r["curve"], r["curve"][1:]))
