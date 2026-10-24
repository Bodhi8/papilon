import numpy as np

from papilon import make_synthetic_mmm


def test_shape_and_columns():
    df, truth = make_synthetic_mmm()
    assert len(df) == 156
    assert {"date", "paid_search", "paid_social", "online_video", "display",
            "price_index", "promo", "sales"} <= set(df.columns)
    assert (df[["paid_search", "paid_social", "online_video", "display"]] >= 0).all().all()


def test_reproducible_with_seed():
    a, _ = make_synthetic_mmm(seed=7)
    b, _ = make_synthetic_mmm(seed=7)
    assert a.equals(b)


def test_truth_summary_is_consistent():
    _, truth = make_synthetic_mmm()
    s = truth["summary"]
    for c in ["paid_search", "paid_social", "online_video", "display"]:
        assert np.isclose(s[c]["roas"], s[c]["contribution_total"] / s[c]["spend_total"])
        assert 0 < s[c]["marginal_roas"] < s[c]["roas"]  # diminishing returns
    assert 0.7 < s["baseline_share"] < 0.85
