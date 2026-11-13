"""The optional JAX backend: the same posterior as the NumPy code, sampled with NUTS.

Run with ``pytest -m jax`` after ``pip install -e ".[dev,jax]"``. Skipped when JAX is not installed.
"""
import numpy as np
import pytest

import papilon as pp

pytestmark = pytest.mark.jax
pytest.importorskip("jax")
pytest.importorskip("numpyro")

from papilon.mmm import _jax  # noqa: E402

MEDIA = ["paid_search", "paid_social", "online_video", "display"]
KW = dict(target="sales", media=MEDIA, date_col="date", controls=["price_index", "promo"])


@pytest.fixture(scope="module")
def data():
    return pp.make_synthetic_mmm()


@pytest.fixture(scope="module")
def geo():
    return pp.make_synthetic_geo_mmm(n_geos=5, n_weeks=104, seed=4)


def _agree(model, n=6, spread=0.3, seed=0):
    """Largest gap between the JAX and NumPy log posteriors, and the gradient at the last point."""
    import jax
    fam = model._fam
    logp = jax.jit(_jax.log_posterior(model, fam))
    rng = np.random.default_rng(seed)
    z = model._init_point(fam) + spread * rng.standard_normal((n, model._dim(fam)))
    ref = model._log_post_batch(z, fam)
    got = np.array([float(logp(x)) for x in z])
    assert np.all(np.isfinite(ref))
    grad = np.asarray(jax.grad(logp)(z[-1]))
    return float(np.max(np.abs(got - ref))), grad


def test_national_log_posterior_matches_numpy_for_every_family(data):
    df, _ = data
    lift = [{"channel": "paid_search", "start": df.date.iloc[60], "end": df.date.iloc[67], "lift": -900.0,
             "se": 300.0, "factor": 0.0}]
    df = df.assign(reach=df.online_video / 3.0, freq=2.0 + np.sin(np.arange(len(df)) / 9.0),
                   email=50 + 10 * np.cos(np.arange(len(df)) / 5.0))
    models = [
        pp.MMM(df, **KW),
        pp.MMM(df, delayed=["online_video"], s_curve=["display"], roi_priors={"display": (0.8, 0.3)}, **KW),
        pp.MMM(df, carryover={"online_video": "weibull"}, saturation={"display": "logistic", "paid_social": "s_curve"},
               time_varying=["paid_social"], baseline_drift=2, lift_tests=lift, **KW),
        pp.MMM(df, reach_frequency={"online_video": ("reach", "freq")}, organic=["email"], **KW),
    ]
    for m in models:
        gap, grad = _agree(m)
        assert gap < 1e-7, (m.family.name(), gap)
        assert np.all(np.isfinite(grad))


def test_geo_log_posterior_matches_numpy(geo):
    df, _ = geo
    args = (df, "sales", ["search", "social", "video"], "date", "region")
    dates = sorted(df.date.unique())
    lift = [{"region": "r1", "channel": "video", "start": dates[40], "end": dates[47], "lift": -300.0, "se": 120.0,
             "factor": 0.0},
            {"region": "r3", "channel": "search", "start": dates[70], "end": dates[75], "lift": 150.0, "se": 90.0,
             "factor": 2.0}]
    for kw in (dict(), dict(lift_tests=lift, delayed=["video"], s_curve=["social"]),
               dict(regional_curves=True), dict(regional_curves=True, lift_tests=lift, carryover={"video": "weibull"})):
        m = pp.GeoMMM(*args, **kw)
        gap, grad = _agree(m, spread=0.2)
        assert gap < 1e-6, (kw, gap)
        assert np.all(np.isfinite(grad))


def test_nuts_agrees_with_the_gate_on_the_bundled_brand(data):
    df, _ = data
    kw = dict(delayed=["online_video"], seed=11, **KW)
    gate = pp.MMM(df, sampler="gate", **kw).fit()
    nuts = pp.MMM(df, sampler="nuts", **kw).fit(warmup=500, samples=500)
    g = nuts.gate_results[nuts.family.name()]
    assert (g.method, g.verdict) == ("nuts", "NUTS") and g.passed
    assert g.rhat < 1.01 and g.ess_bulk >= 400 and g.divergences == 0 and np.isnan(g.log_evidence)
    a, b = gate.get_roi(), nuts.get_roi()
    assert np.abs(a.roas_mean - b.roas_mean).max() < 0.12 and np.abs(a.roas_p95 - b.roas_p95).max() < 0.25
    total = float(df[MEDIA].sum(axis=1).mean())                    # everything downstream reads the same draws
    assert nuts.optimize_budget(total).optimal_spend.sum() == pytest.approx(total, rel=1e-6)
    assert len(nuts.forecast(nuts.future_frame(4))) == 4
    one = pp.MMM(df, sampler="nuts", **kw).fit(chains=1, warmup=100, samples=100)
    assert not one.gate_results[one.family.name()].passed          # one chain cannot show that chains agree


def test_geo_nuts_agrees_with_the_gate(geo):
    df, _ = geo
    args = (df, "sales", ["search", "social", "video"], "date", "region")
    gate = pp.GeoMMM(*args, seed=2, sampler="gate").fit(n_starts=1)
    nuts = pp.GeoMMM(*args, seed=2, sampler="nuts").fit(warmup=400, samples=400)
    g = nuts.gate_results[nuts.family.name()]
    assert g.method == "nuts" and g.rhat < 1.02 and g.divergences == 0
    assert np.abs(gate.get_roi().roas_mean - nuts.get_roi().roas_mean).max() < 0.1
    assert np.abs(gate.get_roi(by_geo=True).roas_mean - nuts.get_roi(by_geo=True).roas_mean).mean() < 0.1
    assert np.abs(gate.pooling().tau_mean - nuts.pooling().tau_mean).max() < 0.15


def test_auto_fits_regional_curves_with_nuts(tmp_path):
    df, truth = pp.make_synthetic_geo_mmm(n_geos=6, n_weeks=104, seed=4, regional_curves=0.5)
    media = ["search", "social", "video"]
    m = pp.GeoMMM(df, "sales", media, "date", "region", seed=2, regional_curves=True)
    assert m.sampler == "auto" and m._dim(m._fam) == 12 + 6 * 7
    m.fit(warmup=300, samples=300)
    g = m.gate_results[m.family.name()]
    assert g.method == "nuts" and g.verdict in ("NUTS", "NUTS_FAILED") and g.rhat < 1.1
    assert len(m._draws["beta"]) == 1000 and m._draws["zgeo"].shape == (1000, 6, 12)
    by_geo = m.get_roi(by_geo=True)
    true = np.array([truth["regional"][r][c]["roas"] for r in m.geos for c in media])
    assert ((by_geo.roas_p05 - 0.05 <= true) & (true <= by_geo.roas_p95 + 0.05)).mean() >= 0.7
    assert len(m.regional_spread()) == 6 and len(m.forecast(m.future_frame(4))) == 24
    m.save(tmp_path / "regional.zip")
    back = pp.MMM.load(tmp_path / "regional.zip")
    assert back.gate_results[m.family.name()].method == "nuts"
    np.testing.assert_allclose(back.get_roi().roas_mean, m.get_roi().roas_mean, rtol=1e-9)
