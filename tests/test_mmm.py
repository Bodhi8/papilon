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
    out = quick_model.optimize_budget(total=total, bounds=bounds)
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
