"""Papilon 0.5: batched posterior, S-curves, self-checks, lift-test value, knowledge graph, AI tools."""
import asyncio
import json

import numpy as np
import pytest

import papilon as pp
from papilon.ai import ToolBox, brief
from papilon.mmm.transforms import delayed_adstock, geometric_adstock

MEDIA = ["paid_search", "paid_social", "online_video", "display"]


@pytest.fixture(scope="module")
def data():
    return pp.make_synthetic_mmm()


@pytest.fixture(scope="module")
def model(data):
    df, _ = data
    return pp.MMM(df, "sales", MEDIA, "date", controls=["price_index", "promo"],
                  delayed=["online_video"], seed=11).fit(n_starts=2)


def test_batch_and_single_posterior_agree(data):
    df, _ = data
    m = pp.MMM(df, "sales", MEDIA, "date", controls=["price_index", "promo"], seed=1,
               roi_priors={"paid_search": (1.2, 0.25)})
    rng = np.random.default_rng(0)
    for delayed, shaped in [([], []), (["online_video"], []), (["online_video"], ["online_video", "display"])]:
        Z = rng.normal(0, 0.5, (40, m._dim(delayed, shaped)))
        batch = m._log_post_batch(Z, delayed, shaped)
        single = np.array([m._log_post(z, delayed, shaped) for z in Z])
        assert np.allclose(batch, single, atol=1e-8)


def test_effects_match_reference_transforms(data):
    df, _ = data
    m = pp.MMM(df, "sales", MEDIA, "date", seed=1)
    P = {"alpha": np.array([[0.3, 0.5, 0.6, 0.4]]), "half_sat": np.array([[1.0, 0.8, 1.2, 1.5]]),
         "beta": np.array([[0.1, 0.2, 0.3, 0.05]]), "peak": np.array([[np.nan, np.nan, 2.0, np.nan]]),
         "shape": np.ones((1, 4))}
    eff = m._effects(P)[:, 0, :]
    for j in range(4):
        x = m.spend[:, j]
        ad = delayed_adstock(x, 0.6, 2.0, 8) if j == 2 else geometric_adstock(x, P["alpha"][0, j], 8)
        assert np.allclose(eff[:, j], P["beta"][0, j] * ad / (ad + P["half_sat"][0, j]))


def test_s_curve_is_recovered_when_true():
    rng = np.random.default_rng(3)
    df, _ = pp.make_synthetic_mmm()
    x = df["online_video"].to_numpy() / df["online_video"].mean()
    old = 200 * x / (x + 1.0)
    ad = geometric_adstock(x, 0.3, 8)
    df = df.assign(sales=df["sales"] - 0 * old + 260 * ad ** 3 / (ad ** 3 + 1.2 ** 3) + rng.normal(0, 5, len(df)))
    m = pp.MMM(df, "sales", MEDIA, "date", controls=["price_index", "promo"], seed=5)
    fam = m.compare_models({"concave": {}, "video_s": {"s_curve": ["online_video"]}}, n_starts=2)
    assert fam.loc["video_s", "probability"] > 0.9
    assert m.s_curve == ["online_video"] and m.parameters().loc["online_video", "shape"] > 1.3
    assert m.posterior_curves()["shape"].shape == (2000, 4)
    plan = m.optimize_budget(float(df[MEDIA].sum(axis=1).mean()))
    assert abs(plan.optimal_spend.sum() - df[MEDIA].sum(axis=1).mean()) < 1e-4


def test_backtest_beats_naive(model):
    bt = model.backtest(13)
    s = bt.attrs["summary"]
    assert len(bt) == 13 and s["mape"] < s["naive_mape"] and s["coverage_90"] >= 0.7


def test_self_check_reports_calibration_and_confounding(model):
    clean = model.self_check(3, seed=2).attrs["summary"]
    assert 0 <= clean["coverage_90"] <= 1 and set(clean["coverage_by_channel"]) == set(MEDIA)
    bad = model.self_check(3, confounded={"paid_search": 1.0}, seed=2).attrs["summary"]
    assert bad["bias_by_channel"]["paid_search"] > clean["bias_by_channel"]["paid_search"] + 0.3


def test_value_of_lift_test_ranks_uncertain_channel_first(model):
    v = model.value_of_lift_test()
    assert set(v.index) == set(MEDIA)
    assert (v.width_after < v.width_now + 1e-9).all()
    # paid search and display are both wide and statistically tied for first; video is already well measured
    assert v.index[-1] == "online_video" and set(v.index[:2]) == {"paid_search", "display"}
    assert (v.expected_gain > -3 * v.gain_se - 1e-9).all()


def graph():
    g = pp.MarketingGraph("sales")
    g.channel("paid_search", stage="lower", group="search").channel("paid_social", stage="mid")
    g.channel("online_video", stage="upper").channel("display", stage="mid")
    g.driver("seasonality", ["paid_search", "sales"], modeled=True)
    g.driver("price", ["sales"], column="price_index").driver("promo", ["sales"], column="promo")
    g.event("launch", ["paid_search", "paid_social", "sales"])
    g.mediates("online_video", "paid_search")
    return g


def test_graph_compiles_and_audits(data):
    df, _ = data
    g = graph()
    kw = g.model_kwargs(df)
    assert kw["media"] == MEDIA and kw["controls"] == ["price_index", "promo"] and kw["roi_priors"] == {}
    assert g.carryover_candidates() == {"all_geometric": [], "upper_funnel_delayed": ["online_video"]}
    audit = g.audit(df)
    high = audit[audit.severity == "high"]
    assert set(high.subject) == {"paid_search", "paid_social"} and (high.issue == "unobserved_confounder").all()
    assert "mediated" in set(audit.issue)
    g.experiment("geo_q3", "paid_search", 1.2, 0.3).experiment("geo_q4", "paid_search", 1.4, 0.3)
    mean, sd = g.model_kwargs()["roi_priors"]["paid_search"]
    assert mean == pytest.approx(1.3) and sd == pytest.approx(0.3 / np.sqrt(2))
    a2 = g.audit()
    assert a2[(a2.subject == "paid_search") & (a2.issue == "unobserved_confounder")].severity.iloc[0] == "low"
    assert isinstance(g.build_model(df, "date"), pp.MMM)


def test_graph_relations_proposals_and_round_trip():
    g = graph()
    moves = g.what_moves("online_video")
    assert set(moves.affected) == {"paid_search", "sales"}
    assert any("mediates" in p for p in moves.path)
    g.propose("display", "mediates", "paid_search", by="llm", note="retargeting")
    assert "PROPOSED" in g.describe() and "proposed" in set(g.audit().issue)
    assert "display [mediates]" not in " ".join(g.what_moves("display").path)
    g.confirm("display", "mediates", "paid_search")
    assert "display [mediates]" in " ".join(g.what_moves("display").path)
    g2 = pp.MarketingGraph.from_dict(json.loads(json.dumps(g.to_dict())))
    assert g2.describe() == g.describe()
    ld = json.loads(g.to_jsonld())
    assert ld["@context"]["@vocab"].startswith("https://vector1.ai/ns/") and len(ld["@graph"]) == g.g.number_of_nodes()


def test_toolbox_grounds_and_guards(model):
    box = ToolBox(model, graph=graph())
    names = [t["name"] for t in box.specs("tool_use")]
    assert "what_moves" in names and all("input_schema" in t for t in box.specs("tool_use"))
    assert all(t["type"] == "function" for t in box.specs("function"))
    assert "what_moves" not in [t["name"] for t in ToolBox(model).specs()]
    roi = box.call("get_roi")
    json.dumps(roi)
    assert {r["channel"] for r in roi["result"]} == set(MEDIA)
    plan = box.call("optimize_budget", {})
    assert plan["result"]["summary"]["objective"] == "quantile" and plan["caveats"]
    assert "error" in box.call("response_curve", {"channel": "tv", "spend_levels": [1]})
    assert "error" in box.call("nope")
    far = box.call("response_curve", {"channel": "display", "spend_levels": [1e6]})
    assert any("extrapolation" in c for c in far["caveats"])
    b = brief(model, graph=graph())
    assert any("launch" in c for c in b["caveats"]) and b["channels"]["online_video"]["carryover"] == "delayed"


def test_mcp_server_serves_the_tools(model):
    pytest.importorskip("mcp")
    from papilon.mcp_server import build_server
    server = build_server(ToolBox(model, graph=graph()))
    tools = asyncio.run(server.list_tools())
    assert {"brief", "get_roi", "optimize_budget", "what_moves"} <= {t.name for t in tools}
