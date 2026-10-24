import numpy as np
import pandas as pd

import papilon as pp


def test_shannon_entropy_of_fair_coin():
    assert np.isclose(pp.shannon_entropy([0, 1, 0, 1]), 1.0)


def test_analyze_relationships_returns_tables():
    df, _ = pp.make_synthetic_mmm()
    out = pp.analyze_relationships(df.drop(columns="date"), target="sales", visualize=False)
    assert "correlation_matrix" in out and "mutual_information" in out
    assert "sales" not in out["mutual_information"].index


def test_grid_search_optimize_sorts_best_first():
    res = pp.grid_search_optimize(pd.DataFrame(), {"a": [1, 2, 3], "b": [0, 1]},
                                  objective_fn=lambda c: c["a"] + c["b"],
                                  constraints=lambda c: c["a"] + c["b"] <= 3)
    assert res.iloc[0]["score"] == 3
    assert (res["a"] + res["b"] <= 3).all()
