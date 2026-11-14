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
"""Leaderboard entry: PyMC-Marketing MMM with its default priors.

Follows the contract in papilon/benchmarks/SUBMITTING.md:

    entry(df, info) -> {"roas": {channel: (estimate, p05, p95)}, "plan": weekly spend or None}

Configuration (one for every brand, no per-brand tuning):
    adstock      GeometricAdstock(l_max=8)
    saturation   LogisticSaturation()
    seasonality  yearly_seasonality=2
    priors       library defaults
    sampler      NUTS, 4 chains, 1000 tune + 1000 draws, target_accept 0.9, seed 11
    ROAS         incrementality.contribution_over_spend("all_time"), mean and 5-95% range
    plan         BudgetOptimizerWrapper.optimize_budget over the last 52 weeks, weekly budget equal to
                 the historical total, each channel bounded to [0, 3x] its historical weekly spend
                 (the grader's own bounds)

Maintainers of PyMC-Marketing are invited to correct this configuration.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

DRAWS, TUNE, CHAINS, SEED = 1000, 1000, 4, 11


def _fit(df: pd.DataFrame, info: dict, draws: int, tune: int, chains: int):
    from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

    mmm = MMM(
        date_column=info["date_col"],
        channel_columns=list(info["media"]),
        target_column=info["target"],
        control_columns=list(info["controls"]) or None,
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        yearly_seasonality=2,
    )
    cols = [info["date_col"], *info["media"], *info["controls"]]
    mmm.fit(df[cols], df[info["target"]], draws=draws, tune=tune, chains=chains,
            target_accept=0.9, random_seed=SEED, progressbar=False)
    return mmm


def _roas(mmm, media) -> dict:
    r = mmm.incrementality.contribution_over_spend(frequency="all_time")
    r = r.stack(sample=("chain", "draw"))
    out = {}
    for c in media:
        v = np.asarray(r.sel(channel=c)).ravel()
        v = v[np.isfinite(v)]
        out[c] = (float(v.mean()), float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95)))
    return out


def _plan(mmm, df, info):
    """Weekly spend by channel at the same total, or None if the optimizer fails."""
    import xarray as xr
    from pymc_marketing.mmm import BudgetOptimizerWrapper

    media = list(info["media"])
    dates = pd.to_datetime(df[info["date_col"]])
    hist = np.asarray(info["current"], dtype=float)
    # The optimizer's budget and bounds are per period (per week here): verified by sampling the
    # response at the optimum, where each week's spend equals the returned budget.
    bounds = xr.DataArray(np.array([[0.0, 3.0 * h] for h in hist]),
                          dims=["channel", "bound"],
                          coords={"channel": media, "bound": ["lower", "upper"]})
    wrapper = BudgetOptimizerWrapper(model=mmm, start_date=str(dates.iloc[-52].date()),
                                     end_date=str(dates.iloc[-1].date()))
    res = wrapper.optimize_budget(budget=float(info["total"]), budget_bounds=bounds)
    x = np.clip(np.array([float(res.budgets.sel(channel=c)) for c in media]), 0.0, None)
    return x * info["total"] / x.sum()


def pymc_marketing_default(df, info, draws: int = DRAWS, tune: int = TUNE, chains: int = CHAINS):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mmm = _fit(df, info, draws, tune, chains)
        roas = _roas(mmm, info["media"])
        try:
            plan = _plan(mmm, df, info)
        except Exception as exc:  # report, never guess a plan
            print(f"[pymc-marketing] plan failed on world {info.get('world')}: {type(exc).__name__}: {exc}")
            plan = None
    return {"roas": roas, "plan": plan}
