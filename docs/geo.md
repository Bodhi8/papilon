# Geo models

Regional data multiplies the evidence: each region's spend moves somewhat on its own,
which separates channels that move together nationally.

```python
df, truth = pp.make_synthetic_geo_mmm(n_geos=8)          # long: date, region, spend columns, sales
model = pp.GeoMMM(df, "sales", ["search", "social", "video"], "date", geo_col="region").fit()

model.get_roi()                  # national returns, all regions combined
model.get_roi(by_geo=True)       # one row per region and channel
model.pooling()                  # how much effects differ across regions
model.optimize_budget(total)     # national plan, each channel split across regions as it was
```

**How it works.** Curve shapes (carryover, saturation) are shared by all regions, in each
region's own spend units, so a small region and a large one saturate at the same share
of their usual spend. Each region has its own effect size, drawn around a pooled effect
with a spread the data estimates: regions with little information borrow from the rest.

Given the shapes, the model is linear in each region's baseline and effects, so those
are integrated out exactly. Sampling runs only over the shared shapes, the pooled
effects and their spreads, about four parameters per channel whatever the number of
regions. Eight regions and three channels fit in under half a minute.

## Lift tests by region (1.1)

A test that ran in one region calibrates that region's effect, and the others through
pooling:

```python
tests = [{"region": "r3", "channel": "video", "start": "2025-03-03", "end": "2025-04-21",
          "lift": -4200.0, "se": 900.0, "factor": 0.0}]          # an eight-week holdout in r3
model = pp.GeoMMM(df, "sales", media, "date", geo_col="region", lift_tests=tests).fit()
```

The columns are those of a national lift test ([calibration](calibration.md)) plus the
region, under the name of your `geo_col`. `lift` and `se` are in that region's outcome
units. A test measured as one number over several regions is not supported: enter it
per region, or fit a national model.

## Forecasts and scenarios (1.1)

```python
future = model.future_frame(13, spend={"video": 400.0})   # national spend per week, split by historical shares
fc = model.forecast(future)        # one row per region and week: predicted, p05, p95, each channel's part
fc.attrs["national"]               # the same summary for the sum over regions
model.scenarios({"as is": model.future_frame(13), "more video": future})
```

`future` is a long frame like the data; build it yourself to set each region's spend
separately.

## A curve per region (1.1)

Shared curves assume every region saturates and carries over alike. With
`regional_curves=True` each region gets its own carryover and saturation parameters,
drawn around the pooled value with a spread the data sets:

```python
model = pp.GeoMMM(df, "sales", media, "date", geo_col="region", regional_curves=True).fit()

model.regional_spread()            # how much each curve parameter differs across regions
model.parameters(by_geo=True)      # each region's own curve and effect
model.get_roi(by_geo=True)         # returns and marginal returns per region
```

The sampled parameters now grow with the regions (two per channel and region, more with
delayed or S-shaped curves), which is the job of the NUTS sampler:
`pip install "papilon[jax]"`, and the default `sampler="auto"` uses it. See
[Inference](inference.md) for what it reports and how to read a failed fit. Without the
backend the model still fits with the gate, with a warning; past a handful of regions do
not rely on that.

When to use it: when regions differ in how fast a channel saturates (large and small
markets bought at different weights) and you plan to move money between regions, which
rests on each region's marginal return. For national returns, shared curves are faster
and about as accurate. The [scaling benchmark](https://github.com/Bodhi8/papilon/blob/main/benchmarks/SCALING.md)
measures both models, with both samplers, from 4 to 400 regions.

Bayes-factor comparison of families (`compare_models`) uses shared curves: choose the
family first, then fit the winner with regional curves.

**Not yet in geo models:** time-varying effects, reach and frequency, organic channels,
the backtest, the self-check and `monitor`. Fit a national `MMM` on summed data for
those. Regional effects are not constrained to be positive; `pooling()` reports the share
of draws where one is negative.
