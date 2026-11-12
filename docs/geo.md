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

**Not yet in geo models:** time-varying effects, reach and frequency, organic channels,
lift tests, forecasting and the self-check. Fit a national `MMM` on summed data for
those. Regional effects are not constrained to be positive; `pooling()` reports the share
of draws where one is negative.
