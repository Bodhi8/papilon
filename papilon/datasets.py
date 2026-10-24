"""Synthetic marketing data with a known ground truth, for testing MMMs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .mmm.transforms import delayed_adstock, geometric_adstock, hill

__all__ = ["make_synthetic_mmm"]

CHANNELS = ["paid_search", "paid_social", "online_video", "display"]
TRUTH = {
    "paid_search": {"adstock": "geometric", "alpha": 0.20, "half_sat": 40.0, "beta": 90.0},
    "paid_social": {"adstock": "geometric", "alpha": 0.45, "half_sat": 55.0, "beta": 160.0},
    "online_video": {"adstock": "delayed", "alpha": 0.60, "peak": 2.0, "half_sat": 70.0, "beta": 220.0},
    "display": {"adstock": "geometric", "alpha": 0.30, "half_sat": 35.0, "beta": 45.0},
}
BASE = {"intercept": 600.0, "trend_per_year": 40.0, "season_amp": 90.0,
        "price_elasticity": -350.0, "promo_lift": 120.0, "noise_sd": 30.0}


def _effect(x, p, l_max):
    ad = (delayed_adstock(x, p["alpha"], p["peak"], l_max) if p["adstock"] == "delayed"
          else geometric_adstock(x, p["alpha"], l_max))
    return p["beta"] * hill(ad, p["half_sat"])


def make_synthetic_mmm(n_weeks: int = 156, seed: int = 2026, l_max: int = 8):
    """Weekly sales for a brand running four digital channels, with the truth that generated it.

    Spend and sales are in $ thousands. Paid search spend follows seasonal demand
    (mild confounding), online video runs in flights and peaks two weeks after
    spend, and price and promotions move sales independently of media.

    Returns
    -------
    df : DataFrame with date, spend per channel, price_index, promo and sales.
    truth : dict with the generating parameters and, per channel, true ROAS,
        marginal ROAS and share of sales.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_weeks)
    dates = pd.date_range("2023-01-02", periods=n_weeks, freq="W-MON")
    season = np.sin(2 * np.pi * t / 52.18) + 0.5 * np.sin(4 * np.pi * t / 52.18)

    spend = {
        "paid_search": np.clip(30 + 8 * season + rng.normal(0, 7, n_weeks), 5, None),
        "paid_social": np.clip(rng.gamma(4.0, 12.0, n_weeks), 0, None),
    }
    on = (t % 10) < 4
    spend["online_video"] = np.where(on, rng.uniform(60, 140, n_weeks), rng.uniform(0, 10, n_weeks))
    spend["display"] = np.where(rng.random(n_weeks) < 0.8, rng.uniform(10, 45, n_weeks), 0.0)

    price_index = 1.0 + 0.04 * np.sin(2 * np.pi * t / 26) + rng.normal(0, 0.015, n_weeks)
    promo = (rng.random(n_weeks) < 0.12).astype(int)
    base = (BASE["intercept"] + BASE["trend_per_year"] * t / 52.18 + BASE["season_amp"] * season
            + BASE["price_elasticity"] * (price_index - 1.0) + BASE["promo_lift"] * promo)
    contrib = {c: _effect(spend[c], TRUTH[c], l_max) for c in CHANNELS}
    sales = base + sum(contrib.values()) + rng.normal(0, BASE["noise_sd"], n_weeks)

    df = pd.DataFrame({"date": dates, **{c: np.round(spend[c], 2) for c in CHANNELS},
                       "price_index": np.round(price_index, 4), "promo": promo,
                       "sales": np.round(sales, 2)})
    summary = {}
    for c in CHANNELS:
        s = spend[c].sum()
        bumped = _effect(spend[c] * 1.01, TRUTH[c], l_max).sum()
        summary[c] = {"spend_total": float(s), "contribution_total": float(contrib[c].sum()),
                      "roas": float(contrib[c].sum() / s),
                      "marginal_roas": float((bumped - contrib[c].sum()) / (0.01 * s)),
                      "share_of_sales": float(contrib[c].sum() / sales.sum())}
    summary["baseline_share"] = float(base.sum() / sales.sum())
    truth = {"channels": TRUTH, "base": BASE, "l_max": l_max, "seed": seed, "summary": summary}
    return df, truth
