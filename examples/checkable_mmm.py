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
"""Papilon 0.5 end to end: declare the brand, fit, check the fit, decide what to test.

Run: python examples/checkable_mmm.py     (about a minute)
"""
import papilon as pp
from papilon.ai import ToolBox, brief

df, truth = pp.make_synthetic_mmm()

# 1. Write down what you believe about the brand.
g = pp.MarketingGraph("sales")
g.channel("paid_search", stage="lower").channel("paid_social", stage="mid")
g.channel("online_video", stage="upper").channel("display", stage="mid")
g.driver("seasonality", ["paid_search", "sales"], modeled=True)       # search spend follows demand
g.driver("price", ["sales"], column="price_index").driver("promo", ["sales"], column="promo")
g.mediates("online_video", "paid_search")                             # video lifts branded search
print(g.audit(df).to_string(index=False), "\n")

# 2. Build the model from the graph and let the data choose the carryover shape.
model = g.build_model(df, date_col="date", seed=11)
print(model.compare_carryover(g.carryover_candidates()).round(3), "\n")
print(model.get_roi().round(2), "\n")

# 3. Check it: held-out weeks, then simulated copies of this brand with known returns.
print("backtest:", model.backtest(13).attrs["summary"])
print("self-check:", model.self_check(6).attrs["summary"], "\n")

# 4. Which lift test is worth running first?
print(model.value_of_lift_test().round(2), "\n")

# 5. Plan, and hand the whole thing to an assistant as grounded tools.
plan = model.optimize_budget(float(df[model.media].sum(axis=1).mean()))
for line in brief(model, plan=plan, graph=g)["caveats"]:
    print("caveat:", line)
box = ToolBox(model, graph=g)
print([t["name"] for t in box.specs("tool_use")])
print(box.call("what_moves", {"node": "online_video"})["result"]["downstream"])
