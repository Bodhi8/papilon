"""Papilon 1.0 in a production workflow: check, fit, calibrate, plan, forecast, save, report, loop.

Run: python examples/production_workflow.py     (about two minutes)
"""
import tempfile
from pathlib import Path

import papilon as pp
from papilon.io import validate_frame
from papilon.report import report

out = Path(tempfile.mkdtemp())
full, truth = pp.make_synthetic_mmm(n_weeks=169)
history, new_quarter = full.iloc[:156], full.iloc[156:]
media, controls = ["paid_search", "paid_social", "online_video", "display"], ["price_index", "promo"]

# 1. Check the data before anything else.
issues = validate_frame(history, "sales", media, "date", controls)
print("data issues:", "none" if issues.empty else issues.to_string(index=False))

# 2. Let the data choose the model form, with a lift test on the weeks it ran.
# (the true lift comes from the data generator here; in practice it is your experiment's readout)
from papilon.datasets import _effect
x = history["paid_search"].to_numpy(); held = x.copy(); held[100:108] = 0
true_lift = float((_effect(held, truth["channels"]["paid_search"], 8) - _effect(x, truth["channels"]["paid_search"], 8))[100:115].sum())
test = {"channel": "paid_search", "start": history.date[100], "end": history.date[107],
        "factor": 0.0, "lift": true_lift, "se": abs(true_lift) * 0.15}
model = pp.MMM(history, "sales", media, "date", controls=controls, lift_tests=[test], seed=11)
print(model.compare_models({"delayed video": {"delayed": ["online_video"]},
                            "weibull video": {"weibull": ["online_video"]}}).round(3), "\n")
print(model.get_roi().round(2), "\n")

# 3. Plan, check the plan is stable, and play it forward.
total = float(history[media].sum(axis=1).mean())
plan = model.optimize_budget(total, bagging=8)
print(model.plan_stability(total, n_splits=6).round(2), "\n")
print(model.scenarios({"current mix": model.future_frame(13),
                       "plan": model.future_frame(13, spend=plan.optimal_spend.to_dict())}).round(1), "\n")

# 4. Save, reload, report.
model.save(out / "brand.papilon")
model = pp.MMM.load(out / "brand.papilon")
print("report:", report(model, out / "report.html", plan=plan))

# 5. A quarter later: monitor, then let the learning loop take over.
print("monitor:", {k: v for k, v in model.monitor(new_quarter).items() if k != "detail"})
loop = pp.Loop(history, "sales", media, "date", controls=controls, model_kwargs={"delayed": ["online_video"]}, n_sim=20)
first = loop.cycle()
print("recommended test:", (first["experiment"] or {}).get("design"))
print("exploration pulse between:", first["explore"])
loop.cycle(new_data=new_quarter)
print(loop.history().round(3).T)
