import numpy as np
import pytest

import papilon as pp

MEDIA = ["paid_search", "paid_social", "online_video", "display"]


@pytest.fixture(scope="module")
def data():
    return pp.make_synthetic_mmm()


@pytest.fixture(scope="module")
def quick_model(data):
    df, _ = data
    m = pp.MMM(df, target="sales", media=MEDIA, date_col="date",
               controls=["price_index", "promo"], delayed=["online_video"], seed=3)
    return m.fit(n_draws=600, n_starts=1)


def test_roi_table_is_finite_and_positive(quick_model):
    roi = quick_model.get_roi()
    assert list(roi.index) == MEDIA
    assert np.isfinite(roi.values).all()
    assert (roi["roas_mean"] > 0).all()
    assert (roi["roas_p05"] <= roi["roas_mean"]).all() and (roi["roas_mean"] <= roi["roas_p95"]).all()


def test_contributions_add_up_to_sales(quick_model, data):
    df, _ = data
    c = quick_model.contributions()
    total = c[MEDIA].sum(axis=1) + c["baseline_and_controls"]
    assert np.allclose(total.to_numpy(), df["sales"].to_numpy())


def test_optimize_budget_respects_total_and_bounds(quick_model):
    total = 150.0
    bounds = {"display": (5.0, 15.0)}
    out = quick_model.optimize_budget(total=total, bounds=bounds, objective="mean", max_change=None)
    assert np.isclose(out["optimal_spend"].sum(), total, atol=1e-4)
    assert 5.0 - 1e-6 <= out.loc["display", "optimal_spend"] <= 15.0 + 1e-6
    assert out["optimal_contribution"].sum() >= out["current_contribution"].sum() - 1e-6


def test_response_curve_is_increasing_and_concave(quick_model):
    rc = quick_model.response_curve("paid_social", np.linspace(0, 200, 9))
    y = rc["contribution_mean"].to_numpy()
    assert np.all(np.diff(y) > 0)
    assert np.all(np.diff(y, 2) < 0)


def test_input_validation(data):
    df, _ = data
    with pytest.raises(ValueError):
        pp.MMM(df, target="sales", media=MEDIA, date_col="date", delayed=["tv"])
    bad = df.copy()
    bad.loc[0, "display"] = -1.0
    with pytest.raises(ValueError):
        pp.MMM(bad, target="sales", media=MEDIA, date_col="date")
    with pytest.raises(RuntimeError):
        pp.MMM(df, target="sales", media=MEDIA, date_col="date").get_roi()


@pytest.mark.slow
def test_bayes_factor_recovers_delayed_video(data):
    df, truth = data
    m = pp.MMM(df, target="sales", media=MEDIA, date_col="date",
               controls=["price_index", "promo"], seed=11)
    fam = m.compare_carryover({"all_geometric": [], "video_delayed": ["online_video"]})
    assert fam["probability"].idxmax() == "video_delayed"
    assert fam.loc["video_delayed", "probability"] > 0.99
    assert abs(m.parameters().loc["online_video", "peak_lag"] - 2.0) < 0.5
    roi = m.get_roi()
    true_video = truth["summary"]["online_video"]["roas"]
    assert roi.loc["online_video", "roas_p05"] <= true_video <= roi.loc["online_video", "roas_p95"]


# ---------------------------------------------------------------- 0.4.0
def test_max_change_keeps_every_channel_within_band(quick_model):
    out = quick_model.optimize_budget(total=150.0, max_change=0.25)
    cur, opt = out["current_spend"], out["optimal_spend"]
    assert np.isclose(opt.sum(), 150.0, atol=1e-4)
    assert ((opt >= cur * 0.75 - 1e-6) & (opt <= cur * 1.25 + 1e-6)).all()


def test_plan_summary_and_quantile_objective(quick_model):
    mean_plan = quick_model.optimize_budget(total=150.0, objective="mean", max_change=None)
    q_plan = quick_model.optimize_budget(total=150.0, objective="quantile", quantile=0.1, max_change=None)
    for plan in (mean_plan, q_plan):
        s = plan.attrs["summary"]
        assert s["lift_p05"] <= s["expected_lift"] <= s["lift_p95"]
        assert 0.0 <= s["prob_better"] <= 1.0
    # the downside-focused plan has a better 10th percentile, the mean plan a better mean
    lo = lambda p: np.quantile(quick_model.evaluate_plan(p["optimal_spend"], p["current_spend"])["draws"], 0.1)
    assert lo(q_plan) >= lo(mean_plan) - 1e-6
    assert mean_plan.attrs["summary"]["expected_lift"] >= q_plan.attrs["summary"]["expected_lift"] - 1e-6
    with pytest.raises(ValueError):
        quick_model.optimize_budget(total=150.0, objective="median")


def test_support_limit_caps_at_sustained_history(quick_model, data):
    df, _ = data
    out = quick_model.optimize_budget(total=150.0, support="limit")
    k = quick_model.l_max
    for c in MEDIA:
        roll = np.convolve(df[c].to_numpy(), np.ones(k) / k, mode="valid")
        cap = max(np.quantile(roll, 0.95), out.loc[c, "current_spend"])
        assert out.loc[c, "optimal_spend"] <= cap + 1e-6
    assert not out["outside_support"].any()


def test_posterior_curves_match_response_curve(quick_model):
    pc = quick_model.posterior_curves()
    j = pc["media"].index("paid_social")
    x = 60.0
    direct = (pc["beta"][:, j] * x / (x + pc["half_sat"][:, j])).mean()
    rc = quick_model.response_curve("paid_social", np.array([x]))["contribution_mean"].iloc[0]
    assert np.isclose(direct, rc)
    assert pc["beta"].shape == pc["half_sat"].shape == pc["alpha"].shape


def test_pareto_k_is_reported(quick_model):
    res = list(quick_model.gate_results.values())[0]
    assert np.isfinite(res.pareto_k)


def test_roi_prior_pulls_estimate_toward_experiment(data):
    df, truth = data
    kw = dict(target="sales", media=MEDIA, date_col="date", controls=["price_index", "promo"],
              delayed=["online_video"], seed=3)
    true_search = truth["summary"]["paid_search"]["roas"]
    base = pp.MMM(df, **kw).fit(n_draws=600, n_starts=1).get_roi().loc["paid_search", "roas_mean"]
    cal = pp.MMM(df, roi_priors={"paid_search": (true_search, 0.15)}, **kw).fit(n_draws=600, n_starts=1)
    est = cal.get_roi().loc["paid_search", "roas_mean"]
    assert abs(est - true_search) < abs(base - true_search)
    with pytest.raises(ValueError):
        pp.MMM(df, roi_priors={"tv": (1.0, 0.1)}, **kw)


def test_default_plan_is_downside_protected_and_capped(quick_model):
    out = quick_model.optimize_budget(total=150.0)
    s = out.attrs["summary"]
    assert s["objective"] == "quantile" and s["max_change"] == 0.5
    cur, opt = out["current_spend"], out["optimal_spend"]
    assert ((opt >= cur * 0.5 - 1e-6) & (opt <= cur * 1.5 + 1e-6)).all()
    lift = quick_model.evaluate_plan(opt, cur)["draws"]
    assert np.quantile(lift, 0.10) >= -1e-6   # staying put scores 0, so the plan does no worse
