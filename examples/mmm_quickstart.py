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
