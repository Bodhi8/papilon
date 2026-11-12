# Model families

Each channel's response has three parts: how spend carries over into later periods,
how the carried-over spend saturates, and whether the effect is constant. Each part has
several forms, and `compare_models` lets the data choose between them by marginal
likelihood.

| Part | Form | Parameters | Use when |
|---|---|---|---|
| Carryover | `geometric` (default) | retention | the effect peaks in the period of spend |
| | `delayed` | retention, peak lag | the effect peaks some periods later (video, brand) |
| | `weibull` | shape, scale | the lag shape is unknown; covers both of the above and long tails |
| Saturation | `hill` (default) | half-saturation | returns diminish from the first dollar |
| | `s_curve` | half-saturation, shape | a channel needs a threshold of spend before it works |
| | `logistic` | scale | returns flatten more sharply than the Hill curve |
| Change over time | `time_varying` | 2 slow terms | creative wears out, platforms change |

```python
model = pp.MMM(df, "sales", media, "date", controls=controls,
               carryover={"online_video": "weibull"}, saturation={"display": "logistic"},
               time_varying=["online_video"])

model.compare_models({
    "baseline":   {"delayed": ["online_video"]},
    "weibull":    {"weibull": ["online_video"]},
    "s_curves":   {"delayed": ["online_video"], "s_curve": ["paid_social", "online_video"]},
    "wears_out":  {"delayed": ["online_video"], "time_varying": ["online_video"]},
})
```

**Time-varying effects** multiply a channel's effect by a slow curve over the window.
Plans and forecasts use the effect as of the last period. A fading effect is only
identifiable when the channel's spend varies: a near-constant effect that fades looks
the same as a downward trend in the baseline. Estimates are shrunk toward no change, so
a large decline shows up as a smaller one; read `effect_change` as direction and rough
size.

**Baseline drift** (`baseline_drift=3`) adds slow terms to the baseline for demand that
moves in ways a straight trend and fixed seasonality cannot.

**Reach and frequency.** For channels bought on reach (connected TV, YouTube), pass
`reach_frequency={"ctv": ("ctv_reach", "ctv_frequency")}`. The input becomes reach
times an S-shaped curve in frequency, and `parameters()` reports `optimal_frequency`,
the frequency that buys the most response per impression. Spend still defines the
channel's return.

**Organic channels.** `organic=["email_sends"]` models unpaid activity with carryover and
saturation. It appears in `contributions()` and `decompose()` and is left out of returns
and budget plans.

**Controls** (price, promotions, distribution) are linear terms whose contributions
`decompose()` reports separately.
