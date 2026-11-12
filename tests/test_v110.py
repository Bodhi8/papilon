"""Papilon 1.1: sampler choice, NUTS diagnostics, geo lift tests and forecasting, regional curves, and
compatibility with 1.0. Nothing here needs the optional JAX backend; its tests are in test_jax.py."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import papilon as pp
from papilon.mmm import GateResult, _diagnostics, _jax
from papilon.mmm import model as mm
from papilon.mmm.transforms import geometric_adstock, hill

FIX = Path(__file__).parent / "fixtures"
MEDIA = ["paid_search", "paid_social", "online_video", "display"]
KW = dict(target="sales", media=MEDIA, date_col="date", controls=["price_index", "promo"], yearly_seasonality=2)
GEO_MEDIA = ["search", "social", "video"]


@pytest.fixture(scope="module")
def data():
    return pp.make_synthetic_mmm()[0]


@pytest.fixture(scope="module")
def geo_data():
    return pp.make_synthetic_geo_mmm(n_geos=4, n_weeks=104, seed=4)


@pytest.fixture(scope="module")
def geo_fit(geo_data):
    df, _ = geo_data
    return pp.GeoMMM(df, "sales", GEO_MEDIA, "date", "region", seed=2).fit(n_starts=1)


def _numbers(frame):
    return frame.select_dtypes("number").to_numpy(dtype=float)


# ------------------------------------------------------------ 1.0 compatibility
def test_default_fits_match_1_0_0(data):
    """The default sampler on the default install gives the numbers 1.0.0 gave (recorded before 1.1 changed anything).

    With the library versions that recorded them the match is exact. Across NumPy and SciPy versions the national
    model agrees to about 1e-10 (its plan, which ends in an optimizer, to 1e-4); the geo fit moves by about 1%,
    because its mode search stops at slightly different points. The tolerances allow for that and no more."""
    want = json.loads((FIX / "defaults_1_0_0.json").read_text())

    def close(got, key, tol):
        np.testing.assert_allclose(np.asarray(got, dtype=float), np.asarray(want[key]), rtol=tol, atol=tol, err_msg=key)

    m = pp.MMM(data, seed=11, **KW).fit()
    g = m.gate_results[m.family.name()]
    assert (g.method, g.verdict) == ("importance", "ACCEPT") and m.sampler == "auto"
    close(_numbers(m.get_roi()), "mmm.roi", 1e-6)
    close([g.log_evidence, g.ess, g.pareto_k, g.modes], "mmm.gate", 1e-6)
    close(_numbers(m.optimize_budget(total=float(data[MEDIA].sum(axis=1).mean()))), "mmm.plan", 2e-3)
    close(_numbers(m.forecast(m.future_frame(8))), "mmm.forecast", 1e-6)

    gdf, _ = pp.make_synthetic_geo_mmm(n_geos=6, seed=4)
    gm = pp.GeoMMM(gdf, "sales", GEO_MEDIA, "date", "region", seed=2).fit(n_starts=2)
    g = gm.gate_results[gm.family.name()]
    assert (g.method, g.verdict) == ("importance", "ACCEPT")
    close(_numbers(gm.get_roi()), "geo.roi", 0.05)
    close(gm.get_roi(by_geo=True)[["spend", "roas_mean", "roas_p05", "roas_p95"]].to_numpy(), "geo.by_geo", 0.06)
    close(_numbers(gm.pooling()), "geo.pooling", 0.05)
    close([g.log_evidence, g.ess], "geo.gate", 0.02)
    plan = gm.optimize_budget(float(gm.spend_raw.mean(0).sum()))
    np.testing.assert_allclose(plan[["current_spend", "optimal_spend"]].to_numpy(dtype=float),
                               np.asarray(want["geo.plan"])[:, :2], rtol=0.10, err_msg="geo.plan")


@pytest.mark.parametrize("name", ["mmm", "geo"])
def test_files_saved_by_1_0_0_load(name, tmp_path):
    m = pp.MMM.load(FIX / f"{name}_1_0_0.zip")
    want = json.loads((FIX / f"{name}_1_0_0.expected.json").read_text())
    roi = pd.DataFrame(**want["roi"])
    np.testing.assert_allclose(m.get_roi().to_numpy(dtype=float), roi.to_numpy(dtype=float), rtol=1e-9)
    g = next(iter(m.gate_results.values()))
    assert m.sampler == "auto" and g.passed and g.method == "importance" and np.isnan(g.rhat) and g.divergences == 0
    if name == "geo":
        np.testing.assert_allclose(m.pooling().to_numpy(dtype=float),
                                   pd.DataFrame(**want["pooling"]).to_numpy(dtype=float), rtol=1e-9)
        assert not m.regional_curves and len(m.forecast(m.future_frame(4))) == 4 * len(m.geos)
    m.save(tmp_path / "again.zip")                                  # written in the 1.1 format, read back the same
    again = pp.MMM.load(tmp_path / "again.zip")
    np.testing.assert_allclose(again.get_roi().to_numpy(dtype=float), m.get_roi().to_numpy(dtype=float), rtol=1e-12)


def test_synthetic_geo_data_is_unchanged_and_gains_regional_curves():
    a, ta = pp.make_synthetic_geo_mmm(n_geos=3, n_weeks=60, seed=1)
    b, tb = pp.make_synthetic_geo_mmm(n_geos=3, n_weeks=60, seed=1, regional_curves=0.5)
    assert a[GEO_MEDIA].equals(b[GEO_MEDIA]) and not a.sales.equals(b.sales)
    alphas = lambda t: {round(t["regional"][r]["video"]["alpha"], 6) for r in t["regional"]}
    assert alphas(ta) == {0.6} and len(alphas(tb)) == 3
    assert abs(a.sales.sum() - 325170.42) < 0.5                     # the 1.0.0 data, to the cent it was rounded to


# -------------------------------------------------------------- sampler choice
def test_sampler_argument_and_auto_rule(data, monkeypatch):
    with pytest.raises(ValueError, match="sampler must be one of"):
        pp.MMM(data, sampler="hmc", **KW)
    m = pp.MMM(data, **KW)
    assert m.sampler == "auto" and m._config["sampler"] == "auto"
    monkeypatch.setattr(_jax, "available", lambda: True)
    assert not m._use_nuts(m._fam)                                  # 12 sampled parameters: the gate
    assert not pp.MMM(data, sampler="gate", **KW)._use_nuts(m._fam)
    monkeypatch.setattr(mm, "AUTO_NUTS_DIM", 5)
    assert m._use_nuts(m._fam)                                      # too large for the gate, backend installed
    monkeypatch.setattr(_jax, "available", lambda: False)
    with pytest.warns(RuntimeWarning, match=r"papilon\[jax\]"):
        assert not m._use_nuts(m._fam)                              # too large, no backend: the gate, with a warning
    with pytest.raises(ImportError, match=r"papilon\[jax\]"):
        pp.MMM(data, sampler="nuts", **KW).fit()
    with pytest.raises(ValueError, match="marginal likelihood"):
        pp.MMM(data, sampler="nuts", **KW).compare_carryover({"a": [], "b": ["online_video"]})


def test_gate_result_folds_in_nuts_diagnostics():
    base = dict(family="f", delayed=[], log_evidence=float("nan"), ess=0.5, method="nuts")
    assert GateResult(verdict="NUTS", rhat=1.002, ess_bulk=900.0, **base).passed
    assert not GateResult(verdict="NUTS_FAILED", rhat=1.2, ess_bulk=30.0, divergences=7, **base).passed
    assert GateResult("f", [], 1.0, 0.3, "ACCEPT").passed and GateResult("f", [], 1.0, 1.0, "SMC").passed
    assert not GateResult("f", [], 1.0, 0.001, "ESCALATE").passed


def test_convergence_diagnostics():
    rng = np.random.default_rng(0)
    good = rng.standard_normal((4, 1000, 3))
    s = _diagnostics.summarize(good)
    assert s["passed"] and s["rhat"] < 1.01 and 2500 < s["ess_bulk"] < 6000
    assert not _diagnostics.summarize(good, divergences=3)["passed"]
    assert not _diagnostics.summarize(good[:1])["passed"]           # one chain cannot show agreement
    shifted = good.copy(); shifted[0, :, 1] += 3.0                  # one chain somewhere else
    s = _diagnostics.summarize(shifted)
    assert s["rhat"] > 1.3 and not s["passed"]
    wide = good.copy(); wide[0, :, 2] *= 4.0                        # same center, different spread
    assert _diagnostics.rhat(wide)[2] > 1.05 and _diagnostics.rhat(wide)[0] < 1.01
    ar = np.empty((4, 2000))                                        # AR(1), rho 0.9: about 1/19 of the draws
    e = rng.standard_normal((4, 2000)); ar[:, 0] = e[:, 0]
    for t in range(1, 2000):
        ar[:, t] = 0.9 * ar[:, t - 1] + np.sqrt(1 - 0.81) * e[:, t]
    ess = _diagnostics.ess_bulk(ar)[0]
    assert 200 < ess < 800 and not _diagnostics.summarize(ar[:, :500])["passed"]
    assert np.isnan(_diagnostics.rhat(good[:, :3])).all()


# ------------------------------------------------------- geo: lift tests, forecasts
def test_geo_lift_test_calibrates_its_region(geo_data, geo_fit):
    df, truth = geo_data
    dates = sorted(df.date.unique())
    reg = df[df.region == "r1"].reset_index(drop=True)
    tv = truth["regional"]["r1"]["video"]
    x = reg.video.to_numpy()
    contrib = lambda v: tv["max_effect"] * hill(geometric_adstock(v / x.mean(), tv["alpha"], 8), tv["half_sat"])
    off = x.copy(); off[40:52] = 0.0                                # a twelve-week holdout of video in r1
    true_lift = float((contrib(off) - contrib(x))[40:59].sum())
    assert true_lift < 0
    test = [{"region": "r1", "channel": "video", "start": dates[40], "end": dates[51], "lift": true_lift,
             "se": 0.05 * abs(true_lift), "factor": 0.0}]
    m = pp.GeoMMM(df, "sales", GEO_MEDIA, "date", "region", seed=2, lift_tests=test).fit(n_starts=1)
    assert m.gate_results[m.family.name()].passed
    row = lambda model: model.get_roi(by_geo=True).query("region == 'r1' and channel == 'video'").iloc[0]
    with_test, without = row(m), row(geo_fit)
    assert with_test.roas_p95 - with_test.roas_p05 < 0.8 * (without.roas_p95 - without.roas_p05)
    assert with_test.roas_p05 - 0.1 <= tv["roas"] <= with_test.roas_p95 + 0.1
    assert {"mroas_mean", "mroas_p05", "mroas_p95"} <= set(m.get_roi(by_geo=True).columns)
    for bad in ({"region": "nowhere"}, {"channel": "radio"}, {"se": 0.0}, {"start": "1999-01-04", "end": "1999-02-01"}):
        with pytest.raises(ValueError):
            pp.GeoMMM(df, "sales", GEO_MEDIA, "date", "region", lift_tests=[{**test[0], **bad}])


def test_geo_forecast_and_scenarios(geo_data, geo_fit):
    df, _ = geo_data
    m = geo_fit
    future = m.future_frame(8)
    assert len(future) == 8 * len(m.geos) and future.date.min() > df.date.max()
    fc = m.forecast(future)
    assert len(fc) == len(future) and (fc.p05 <= fc.predicted).all() and (fc.predicted <= fc.p95).all()
    national = fc.attrs["national"]
    recent = df.groupby("date").sales.sum().iloc[-13:].mean()
    assert len(national) == 8 and (national.predicted.between(0.7 * recent, 1.3 * recent)).all()
    np.testing.assert_allclose(fc.groupby("date").predicted.sum().to_numpy(), national.predicted.to_numpy(), rtol=1e-9)
    more = m.future_frame(8, spend={"video": 1.5 * m.spend_raw[:, 2].mean()})
    assert more.video.sum() == pytest.approx(1.5 * future.video.sum())
    out = m.scenarios({"as_is": future, "more_video": more})
    assert out.loc["more_video", "gain_vs_first"] > 0 and out.loc["more_video", "prob_beats_first"] > 0.8
    with pytest.raises(ValueError):
        m.forecast(future[future.region != "r0"])                   # a region is missing
    with pytest.raises(ValueError):
        m.future_frame(4, spend={"radio": 1.0})
    with pytest.raises(NotImplementedError):
        m.backtest()


# ------------------------------------------------------------- regional curves
def test_regional_curves_model_and_results(geo_data, tmp_path):
    df, _ = geo_data
    args = (df, "sales", GEO_MEDIA, "date", "region")
    shared = pp.GeoMMM(*args, seed=3)
    m = pp.GeoMMM(*args, seed=3, regional_curves=True, sampler="gate")
    G, d0 = len(m.geos), shared._dim(shared._fam)
    nr = len(m._regional_index(m._fam))
    assert nr == 6 and m._dim(m._fam) == d0 + nr * (1 + G)          # carryover and half-saturation per channel
    z0 = m._init_point(m._fam)
    # with every regional offset at zero the regions share one curve: the shared model plus the offsets' prior
    lp_shared = shared._log_post_batch(z0[None, :d0], shared._fam)[0]
    lp = m._log_post_batch(z0[None, :], m._fam)[0]
    _, prior = m._unpack_batch(z0[None, :], m._fam)
    _, prior_shared = shared._unpack_batch(z0[None, :d0], shared._fam)
    assert np.isfinite(lp) and lp - prior[0] == pytest.approx(lp_shared - prior_shared[0], abs=1e-8)
    # stand-in draws around the starting point exercise every result without a long fit
    rng = np.random.default_rng(0)
    z = z0 + 0.15 * rng.standard_normal((60, len(z0)))
    m._store_draws(m._fam, z, np.full(60, 1 / 60), thin=True)
    by_geo = m.parameters(by_geo=True)
    assert len(by_geo) == G * 3 and by_geo.groupby("channel").alpha.nunique().min() == G
    assert len(m.regional_spread()) == nr and (m.regional_spread().spread_mean > 0).all()
    assert len(m.get_roi(by_geo=True)) == G * 3 and len(m.forecast(m.future_frame(3))) == 3 * G
    assert np.isfinite(m.optimize_budget(float(m.spend_raw.mean(0).sum())).optimal_spend).all()
    m.gate_results = {"all_geometric": GateResult("all_geometric", [], float("nan"), 0.4, "NUTS", method="nuts",
                                                  rhat=1.003, ess_bulk=640.0)}
    m.family_probabilities = {"all_geometric": 1.0}
    m.save(tmp_path / "regional.zip")
    back = pp.MMM.load(tmp_path / "regional.zip")
    g = back.gate_results["all_geometric"]
    assert back.regional_curves and back.sampler == "gate" and g.passed and np.isnan(g.log_evidence) and g.rhat == 1.003
    np.testing.assert_allclose(back.get_roi(by_geo=True).roas_mean, m.get_roi(by_geo=True).roas_mean, rtol=1e-9)
    with pytest.raises(ValueError, match="shared curves"):
        m.compare_models({"a": {}, "b": {"delayed": ["video"]}})
    with pytest.raises(ValueError, match="regional_curves=True"):
        shared._store_draws(shared._fam, z[:, :d0], np.full(60, 1 / 60), thin=True) or shared.regional_spread()
    with warnings.catch_warnings():
        warnings.simplefilter("error")                              # the gate was asked for by name: no warning
        assert not m._use_nuts(m._fam)
    auto = pp.GeoMMM(*args, regional_curves=True)
    assert auto._dim(auto._fam) > mm.AUTO_NUTS_DIM                  # "auto" sends this model to NUTS when it can
