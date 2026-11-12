# Papilon

Papilon is a Bayesian marketing mix modeling framework built around one idea: a model
that moves budget should be able to show that it is right, and say when it is not.

- **Model what is there.** Geometric, delayed and Weibull carryover; concave, S-shaped
  and logistic saturation; effects that fade over time; reach and frequency; organic
  channels; regional data with partial pooling. The data chooses between forms by
  Bayes factor.
- **Calibrate with experiments.** Lift tests enter the likelihood on the weeks they ran,
  at the spend level they tested.
- **Check before you trust.** Every fit passes a quality gate, falls back to a stronger
  sampler when it fails, and can be graded on simulated copies of your own brand with
  known returns.
- **Plan for being wrong.** Budget plans maximize the downside, report whether they
  are stable, and can be played forward week by week.
- **Learn on purpose.** The learning loop recommends the experiment worth the most to
  the plan, adds a small budget-neutral pulse to separate channels the model cannot
  tell apart, and is itself graded quarter by quarter against known truth.
- **Run in production.** Data checks, warehouse loaders, saved models that open without
  Papilon, MLflow logging, drift monitoring, HTML reports, a stable API, and tools for
  AI assistants that cannot invent numbers.

```bash
pip install papilon            # add [mcp], [mlflow], [snowflake] or [bigquery] as needed
```

```python
import papilon as pp

df, truth = pp.make_synthetic_mmm()
model = pp.MMM(df, "sales", ["paid_search", "paid_social", "online_video", "display"], "date",
               controls=["price_index", "promo"])
model.compare_models({"concave": {"delayed": ["online_video"]},
                      "video_weibull": {"weibull": ["online_video"]}})
model.get_roi()
model.optimize_budget(total=df[model.media].sum(axis=1).mean())
```

Start with the [quickstart](quickstart.md). Papilon is research software from Vector1
Research, free under the Apache License 2.0; see [API stability](api-stability.md) for
what 1.0 promises.
