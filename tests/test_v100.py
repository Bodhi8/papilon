"""Papilon 1.0: model families, calibration, geo pooling, inference fallback, planning,
forecasting, production tooling, causal discovery and the learning loop."""
import sys
import types
import warnings

import numpy as np
import pandas as pd
import pytest

import papilon as pp
from papilon.datasets import TRUTH, _effect
from papilon.mmm.model import Family

MEDIA = ["paid_search", "paid_social", "online_video", "display"]
CTRL = ["price_index", "promo"]


@pytest.fixture(scope="module")
def data():
    return pp.make_synthetic_mmm()


@pytest.fixture(scope="module")
def fitted(data):
    df, _ = data
    return pp.MMM(df, "sales", MEDIA, "date", controls=CTRL, delayed=["online_video"], seed=11).fit(n_starts=2)


def test_family_spec_and_validation():
    f = Family.from_spec(MEDIA, {"weibull": ["online_video"], "logistic": ["display"], "time_varying": ["paid_social"]})
    assert f.carry["online_video"] == "weibull" and f.sat["display"] == "logistic" and f.tv == ["paid_social"]
    assert f.name() == "weibull:online_video+logistic:display+time_varying:paid_social" and not f.simple()
    with pytest.raises(ValueError):
        Family(MEDIA, {"online_video": "exponential"})
    with pytest.raises(ValueError):
        Family(MEDIA, time_varying=["tv"])


def test_weibull_carryover_finds_the_delayed_peak(data):
    df, _ = data
    m = pp.MMM(df, "sales", MEDIA, "date", controls=CTRL, carryover={"online_video": "weibull"}, seed=11).fit(n_starts=2)
    kern = m.posterior_curves()["kernel"][:, MEDIA.index("online_video")].mean(axis=0)
    assert 1 <= int(np.argmax(kern)) <= 3            # truth: peaks two weeks after spend
    assert m.gate_results[m.family.name()].passed


def test_logistic_saturation_fits_and_plans(data):
    df, _ = data
    m = pp.MMM(df, "sales", MEDIA, "date", controls=CTRL, delayed=["online_video"],
               saturation={c: "logistic" for c in MEDIA}, seed=11).fit(n_starts=2)
    plan = m.optimize_budget(float(df[MEDIA].sum(axis=1).mean()))
    assert abs(plan.optimal_spend.sum() - df[MEDIA].sum(axis=1).mean()) < 1e-3
    rc = m.response_curve("paid_social", [0, 50, 500])
    assert rc.contribution_mean.iloc[0] == 0 and rc.contribution_mean.is_monotonic_increasing


def test_time_varying_effect_detects_wear_out():
    # Wear-out is only identifiable on a channel whose spend varies: a near-constant effect that
    # fades looks the same as a trend. Video runs in flights, so its fading is visible.
    df, truth = pp.make_synthetic_mmm(seed=6)
    eff = _effect(df["online_video"].to_numpy(), TRUTH["online_video"], 8)
    df = df.assign(sales=df["sales"] - eff * np.linspace(0, 0.7, len(df)))   # video loses 70% of its effect
    m = pp.MMM(df, "sales", MEDIA, "date", controls=CTRL, delayed=["online_video"], seed=3)
    fam = m.compare_models({"static": {"delayed": ["online_video"]},
                            "wears_out": {"delayed": ["online_video"], "time_varying": ["online_video"]}}, n_starts=2)
    assert fam.loc["wears_out", "probability"] > 0.5
    assert m.parameters().loc["online_video", "effect_change"] < -0.1


def test_lift_test_calibrates_on_the_weeks_it_ran(data):
    df, truth = data
    p = TRUTH["paid_search"]
    x = df["paid_search"].to_numpy()
    held = x.copy(); held[100:108] = 0
    true_lift = float((_effect(held, p, 8) - _effect(x, p, 8))[100:115].sum())    # negative
    test = {"channel": "paid_search", "start": df.date[100], "end": df.date[107], "factor": 0.0,
            "lift": true_lift, "se": abs(true_lift) * 0.15}
    plain = pp.MMM(df, "sales", MEDIA, "date", controls=CTRL, delayed=["online_video"], seed=11).fit(n_starts=2)
    cal = pp.MMM(df, "sales", MEDIA, "date", controls=CTRL, delayed=["online_video"], seed=11,
                 lift_tests=[test]).fit(n_starts=2)
    w = lambda m: m.get_roi().loc["paid_search", "roas_p95"] - m.get_roi().loc["paid_search", "roas_p05"]
    true_roas = truth["summary"]["paid_search"]["roas"]
    assert w(cal) < 0.7 * w(plain)
    assert abs(cal.get_roi().loc["paid_search", "roas_mean"] - true_roas) <= \
        abs(plain.get_roi().loc["paid_search", "roas_mean"] - true_roas) + 0.05
    with pytest.raises(ValueError):
        pp.MMM(df, "sales", MEDIA, "date", lift_tests=[{**test, "channel": "tv"}])


def test_organic_and_reach_frequency_channels(data):
    df, _ = data
    rng = np.random.default_rng(1)
    t = np.arange(len(df))
    emails = rng.poisson(20, len(df)).astype(float)
    reach = rng.uniform(0.5, 1.5, len(df)) * 1e5
    freq = rng.uniform(1.0, 6.0, len(df))
    g = freq ** 2 / (freq ** 2 + 2.5 ** 2)
    ctv = reach / 1e3 * 0.08
    d2 = df.assign(emails=emails, ctv=ctv, ctv_reach=reach, ctv_freq=freq,
                   sales=df.sales + 2.0 * emails + 30 * (reach / reach.mean() * g) / (reach / reach.mean() * g + 0.5))
    m = pp.MMM(d2, "sales", MEDIA + ["ctv"], "date", controls=CTRL, delayed=["online_video"], organic=["emails"],
               reach_frequency={"ctv": ("ctv_reach", "ctv_freq")}, seed=11).fit(n_starts=2)
    assert "emails" not in m.get_roi().index and "emails" in m.contributions().columns
    assert m.contributions()["emails"].mean() > 10
    of = m.parameters().loc["ctv", "optimal_frequency"]
    assert 0.5 < of < 10
    assert m.forecast(m.future_frame(4)).shape[0] == 4


def test_smc_fallback_runs_when_the_gate_fails(data):
    df, _ = data
    m = pp.MMM(df, "sales", MEDIA, "date", controls=CTRL, delayed=["online_video"], seed=11)
    m.fit(n_starts=2, accept=0.999, refine=0.998, max_refine=0)
    g = m.gate_results[m.family.name()]
    assert g.verdict == "SMC" and g.method == "smc" and g.passed
    ref = pp.MMM(df, "sales", MEDIA, "date", controls=CTRL, delayed=["online_video"], seed=11).fit(n_starts=2)
    assert abs(g.log_evidence - ref.gate_results[ref.family.name()].log_evidence) < 2.0
    assert np.allclose(m.get_roi().roas_mean, ref.get_roi().roas_mean, atol=0.15)
    none = pp.MMM(df, "sales", MEDIA, "date", controls=CTRL, seed=11, escalate="none").fit(
        n_starts=1, accept=0.999, refine=0.998, max_refine=0)
    assert none.gate_results[none.family.name()].verdict == "ESCALATE"


def test_forecast_scenarios_and_monitor():
    full, _ = pp.make_synthetic_mmm(n_weeks=169, seed=8)
    hist, new = full.iloc[:156], full.iloc[156:]
    m = pp.MMM(hist, "sales", MEDIA, "date", controls=CTRL, delayed=["online_video"], seed=2).fit(n_starts=2)
    fc = m.forecast(new.drop(columns="sales"))
    a = new.sales.to_numpy()
    assert np.mean((a >= fc.p05) & (a <= fc.p95)) >= 0.7
    mon = m.monitor(new)
    assert not mon["refit_recommended"], mon["alerts"]
    shifted = new.assign(sales=new.sales * 0.7)
    assert m.monitor(shifted)["refit_recommended"]
    sc = m.scenarios({"now": m.future_frame(13), "no video": m.future_frame(13, spend={"online_video": 0.0})})
    assert sc.loc["no video", "gain_vs_first"] < 0 and sc.loc["no video", "prob_beats_first"] < 0.2


def test_decompose_adds_up(fitted, data):
    df, _ = data
    d = fitted.decompose()
    assert np.allclose(d["fitted"] + d["residual"], df["sales"])
    assert set(CTRL) <= set(d.columns) and d["residual"].std() < 35     # the generator's noise sd is 30


def test_plan_stability_bagging_and_indistinguishable(fitted, data):
    df, _ = data
    total = float(df[MEDIA].sum(axis=1).mean())
    st = fitted.plan_stability(total, n_splits=6)
    assert set(st.columns) >= {"direction_agreement", "stable"} and st.direction_agreement.between(0, 1).all()
    bag = fitted.optimize_budget(total, bagging=4)
    assert abs(bag.optimal_spend.sum() - total) < 1e-6 and bag.attrs["summary"]["bagging"] == 4
    assert isinstance(fitted.indistinguishable_channels(), pd.DataFrame)


def test_save_and_load_round_trip(tmp_path, data):
    df, _ = data
    m = pp.MMM(df, "sales", MEDIA, "date", controls=CTRL, carryover={"online_video": "weibull"},
               time_varying=["paid_social"], roi_priors={"display": (0.7, 0.3)}, seed=4,
               escalate="none").fit(n_starts=1)
    m.save(tmp_path / "m.papilon")
    m2 = pp.MMM.load(tmp_path / "m.papilon")
    assert np.allclose(m.get_roi().to_numpy(), m2.get_roi().to_numpy())
    assert m2.family.name() == m.family.name()
    f = m.future_frame(5)
    assert np.allclose(m.forecast(f).predicted, m2.forecast(f).predicted)


def test_validate_frame_and_long_to_wide(data):
    df, _ = data
    from papilon.io import long_to_wide, validate_frame
    assert validate_frame(df, "sales", MEDIA, "date", CTRL).empty
    bad = df.copy()
    bad.loc[3, "display"] = -1
    bad = pd.concat([bad, bad.iloc[[5]]])
    issues = validate_frame(bad, "sales", MEDIA, "date", CTRL)
    assert {"negative_spend", "duplicate_periods"} <= set(issues.check) and issues.severity.iloc[0] == "error"
    long = df.melt(id_vars=["date"], value_vars=MEDIA, var_name="channel", value_name="spend")
    wide = long_to_wide(long, "date", "channel", "spend")
    assert np.allclose(wide.sort_values("date")[MEDIA].to_numpy(), df[MEDIA].to_numpy())


def test_mlflow_logging_with_a_stand_in(fitted, monkeypatch):
    calls = {}

    class Run:
        info = types.SimpleNamespace(run_id="abc")
        def __enter__(self): return self
        def __exit__(self, *a): return False
    fake = types.SimpleNamespace(
        set_experiment=lambda n: calls.setdefault("exp", n), start_run=lambda run_name=None: Run(),
        set_tags=lambda t: calls.setdefault("tags", t), log_params=lambda p: calls.setdefault("params", p),
        log_metrics=lambda m: calls.setdefault("metrics", {}).update(m),
        log_artifact=lambda path: calls.setdefault("artifacts", []).append(path))
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    from papilon.tracking import log_to_mlflow
    assert log_to_mlflow(fitted, experiment="mmm") == "abc"
    assert "roas_paid_search" in calls["metrics"] and len(calls["artifacts"]) == 2


def test_discovery_and_identification():
    from papilon.discovery import discover_structure
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=600), rng.normal(size=600)
    c = a + b + 0.5 * rng.normal(size=600)
    d = 2 * c + rng.normal(size=600)
    G = discover_structure(pd.DataFrame(dict(a=a, b=b, c=c, d=d)), preprocess="none")
    assert {(u, v) for u, v in G.edges} == {("a", "c"), ("b", "c"), ("c", "d")}
    g = pp.MarketingGraph("sales")
    g.channel("search").channel("social", stage="mid").channel("video", stage="upper")
    g.driver("seasonality", ["search", "sales"], modeled=True).event("launch", ["search", "sales"])
    g.mediates("video", "search")
    ident = g.identification().set_index("channel")
    assert not ident.loc["search", "identified"] and ident.loc["search", "unobserved"] == "launch"
    assert ident.loc["video", "identified"] and ident.loc["social", "identified"]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pp.infer_causal_structure(pd.DataFrame(dict(a=a, c=c)), visualize=False)
        assert any(issubclass(x.category, FutureWarning) for x in w)


def test_geo_model_pools_and_recovers():
    df, truth = pp.make_synthetic_geo_mmm(n_geos=6, seed=4)
    m = pp.GeoMMM(df, "sales", ["search", "social", "video"], "date", "region", seed=2).fit(n_starts=2)
    g = m.gate_results[m.family.name()]
    assert g.passed
    roi = m.get_roi()
    covered = [roi.loc[c, "roas_p05"] - 0.1 <= truth["national_roas"][c] <= roi.loc[c, "roas_p95"] + 0.1
               for c in ["search", "social", "video"]]
    assert sum(covered) >= 2
    by_geo = m.get_roi(by_geo=True)
    assert len(by_geo) == 18 and m.pooling().tau_mean.between(0.02, 1.5).all()
    plan = m.optimize_budget(float(m.spend_raw.mean(0).sum()))
    assert abs(plan.optimal_spend.sum() - m.spend_raw.mean(0).sum()) < 1e-3
    with pytest.raises(NotImplementedError):
        m.backtest()
    with pytest.raises(ValueError):
        pp.GeoMMM(df, "sales", ["search"], "date", "region", time_varying=["search"])


def test_learning_loop_cycles_and_recommends(data):
    full, _ = pp.make_synthetic_mmm(n_weeks=182, seed=9)
    lp = pp.Loop(full.iloc[:156], "sales", MEDIA, "date", controls=CTRL, horizon=13,
                 model_kwargs={"delayed": ["online_video"]}, n_sim=15)
    first = lp.cycle()
    assert len(first["schedule"]) == 13 and first["forecast"].shape[0] == 13
    assert abs(first["schedule"][MEDIA].sum(axis=1) - first["plan"].optimal_spend.sum()).max() < 1e-6
    assert len(first["explore"]) == 2
    second = lp.cycle(new_data=full.iloc[156:169],
                      tests=[{"channel": "paid_search", "start": full.date[160], "end": full.date[165],
                              "factor": 2.0, "lift": 120.0, "se": 40.0}])
    assert second["forecast_score"]["periods"] == 13 and second["tests_used"] == 1
    h = lp.history()
    assert list(h.cycle) == [0, 1] and "last_model_coverage_90" in h.columns and "last_model_plan_adherence" in h.columns


def test_report_is_self_contained(fitted, tmp_path):
    from papilon.report import report
    p = report(fitted, tmp_path / "r.html")
    html = p.read_text()
    assert html.count("data:image/png;base64") >= 5 and "<script" not in html
