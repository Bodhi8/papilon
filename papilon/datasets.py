#   Copyright 2024 - 2026 Brian Curry / Vector1 Research
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Synthetic marketing data with a known ground truth, for testing MMMs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .mmm.transforms import delayed_adstock, geometric_adstock, hill

__all__ = ["make_synthetic_mmm", "make_synthetic_geo_mmm"]

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


def make_synthetic_geo_mmm(n_geos: int = 8, n_weeks: int = 156, seed: int = 2026, l_max: int = 8):
    """Weekly sales and spend for ``n_geos`` regions of different sizes, with the truth that generated them.

    Three channels share carryover and curve shapes across regions; each region's
    effect size is the pooled effect times a regional factor (log-normal, sd 0.25).
    Regions plan spend separately, which is what lets a geo model separate channels.

    Returns a long DataFrame (date, region, search, social, video, sales) and a dict
    with the pooled parameters, each region's effects and the true national and
    regional ROAS.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_weeks)
    dates = pd.date_range("2023-01-02", periods=n_weeks, freq="W-MON")
    season = np.sin(2 * np.pi * t / 52.18)
    media = ["search", "social", "video"]
    shared = {"search": {"alpha": 0.2, "half_sat": 0.9, "mu": 0.12},
              "social": {"alpha": 0.45, "half_sat": 1.1, "mu": 0.18},
              "video": {"alpha": 0.6, "half_sat": 1.3, "mu": 0.25}}
    rows, regional, num, den = [], {}, np.zeros(3), np.zeros(3)
    for g in range(n_geos):
        size = rng.uniform(0.5, 3.0)
        base = 600 * size * (1 + 0.1 * season) + 20 * size * t / 52.18
        sales = base.copy()
        spend, eff_g = {}, {}
        for j, c in enumerate(media):
            x = size * 40 * rng.gamma(4, 1, n_weeks) / 4 * (1 + 0.3 * np.sin(2 * np.pi * t / 26 + rng.uniform(0, 6)))
            ad = geometric_adstock(x / x.mean(), shared[c]["alpha"], l_max)
            b = shared[c]["mu"] * np.exp(rng.normal(0, 0.25))
            eff = b * hill(ad, shared[c]["half_sat"]) * base.mean()
            spend[c], eff_g[c] = x, b
            sales += eff
            num[j] += eff.sum(); den[j] += x.sum()
            regional.setdefault(f"r{g}", {})[c] = {"beta": float(b), "roas": float(eff.sum() / x.sum())}
        sales += rng.normal(0, 0.03 * base.mean(), n_weeks)
        rows.append(pd.DataFrame({"date": dates, "region": f"r{g}", **{c: np.round(spend[c], 2) for c in media},
                                  "sales": np.round(sales, 2)}))
    truth = {"shared": shared, "regional": regional,
             "national_roas": {c: float(num[j] / den[j]) for j, c in enumerate(media)}}
    return pd.concat(rows, ignore_index=True), truth
