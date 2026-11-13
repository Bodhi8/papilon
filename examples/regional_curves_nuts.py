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
"""A geo model with a curve per region, fitted with NUTS, and checked against known truth.

Needs the optional backend: pip install "papilon[jax]". Takes a few minutes on a laptop.

    python examples/regional_curves_nuts.py
"""
import numpy as np

import papilon as pp
from papilon.mmm.transforms import geometric_adstock, hill

REGIONS = 12
MEDIA = ["search", "social", "video"]

# Twelve regions whose carryover and saturation differ (spread 0.5), with the truth that generated them.
df, truth = pp.make_synthetic_geo_mmm(n_geos=REGIONS, seed=4, regional_curves=0.5)

# An eight-week holdout of video in one region, entered as a lift test, calibrates that region and, through
# pooling, the rest. (The true lift comes from the data generator here; in practice it is your experiment's readout.)
dates = sorted(df.date.unique())
t, x = truth["regional"]["r3"]["video"], df.loc[df.region == "r3", "video"].to_numpy()
contribution = lambda v: t["max_effect"] * hill(geometric_adstock(v / x.mean(), t["alpha"], 8), t["half_sat"])
held = x.copy(); held[100:108] = 0
true_lift = float((contribution(held) - contribution(x))[100:115].sum())
tests = [{"region": "r3", "channel": "video", "start": dates[100], "end": dates[107], "lift": true_lift,
          "se": 0.15 * abs(true_lift), "factor": 0.0}]

model = pp.GeoMMM(df, "sales", MEDIA, "date", geo_col="region", regional_curves=True, lift_tests=tests, seed=2)
print(f"sampled parameters: {model._dim(model.family)}; sampler: {model.sampler}")
model.fit()                                             # "auto" picks NUTS for a model this size

g = model.gate_results[model.family.name()]
print(f"\nfit check: {g.verdict} (method {g.method}), passed={g.passed}, R-hat {g.rhat:.3f}, "
      f"smallest effective sample {g.ess_bulk:.0f}, {g.divergences} divergent transitions")

roi = model.get_roi()
roi["true_roas"] = [truth["national_roas"][c] for c in MEDIA]
print("\nNational returns\n", roi[["roas_mean", "roas_p05", "roas_p95", "true_roas"]].round(2))

by_geo = model.get_roi(by_geo=True)
true = np.array([truth["regional"][r][c]["roas"] for r in model.geos for c in MEDIA])
print(f"\nRegional returns: 90% range holds the truth for {((by_geo.roas_p05 <= true) & (true <= by_geo.roas_p95)).mean():.0%} "
      f"of {len(true)} region-channel pairs; mean absolute error {np.abs(by_geo.roas_mean - true).mean():.2f}")

print("\nHow much curves differ across regions (0 would mean one shared curve)\n", model.regional_spread().round(2))

future = model.future_frame(8, spend={"video": 1.2 * model.spend_raw[:, 2].mean()})
print("\nEight weeks ahead with 20% more video, national\n",
      model.forecast(future).attrs["national"].set_index("date").round(0))
print("\n", model.scenarios({"as is": model.future_frame(8), "more video": future}).round(2).to_string())
