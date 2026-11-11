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
"""Papilon 0.3.0 quickstart: fit a marketing mix model on synthetic data with a known truth."""
import pandas as pd
import papilon as pp

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

df, truth = pp.make_synthetic_mmm()

model = pp.MMM(
    data=df,
    target="sales",
    media=["paid_search", "paid_social", "online_video", "display"],
    date_col="date",
    controls=["price_index", "promo"],
    yearly_seasonality=2,
    seed=11,
)

# Which carryover shape fits online video? Compare families by Bayes factor.
families = model.compare_carryover({
    "all_geometric": [],
    "video_delayed": ["online_video"],
})
print(families.round(3), "\n")
print("Gate history:", {k: v.history for k, v in model.gate_results.items()}, "\n")
print(model.parameters().round(2), "\n")

roi = model.get_roi()
roi["true_roas"] = [truth["summary"][c]["roas"] for c in roi.index]
print(roi.round(2), "\n")

# Reallocate the current average weekly budget.
weekly_total = df[model.media].sum(axis=1).mean()
print(model.optimize_budget(total=weekly_total).round(1))
