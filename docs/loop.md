# The learning loop

A model refit every quarter on data shaped by its own plans can lock in its mistakes.
It overrates a channel, the plan leans on that channel, the other channels stop
varying, and the next fit grows more confident in the same error. `Loop` feeds back
only new information:

1. **New periods**, scored against the forecast the last cycle made for them.
2. **Experiment results.** Each cycle recommends the lift test worth the most to the
   plan; results come back as calibration on the weeks they ran.
3. **Designed variation.** A small budget-neutral pulse moves money back and forth
   between the two channels the model can least tell apart, so the data the plan
   produces can separate them.

```python
loop = pp.Loop(history, "sales", media, "date", controls=controls, horizon=13, explore=0.10,
               model_kwargs={"delayed": ["online_video"]})
cycle = loop.cycle()                      # first plan
cycle["schedule"]                         # spend per week for the quarter, pulse included
cycle["experiment"]                       # the recommended test and its design
# ... a quarter later
cycle = loop.cycle(new_data=last_quarter, tests=[measured_test])
loop.history()                            # one row per cycle: forecast accuracy, returns, ranges, tests
```

## Does it converge?

`pycausalsim.futures.compare_loops` runs the loop against simulated brands with known
returns, quarter after quarter, alongside a model fitted once ("static") and one refit
every quarter without tests or exploration ("refresh"). Six quarters, eight brands per
track ([full results](https://github.com/Bodhi8/papilon/blob/main/benchmarks/LOOP.md)):

| Policy | Standard brands: best gain captured, quarter 6 | Hard brands: error in returns, quarter 1 → 6 | Sales over six quarters vs static (standard, hard) |
|---|---|---|---|
| Fit once | 34% | 0.59 → 0.61 | 0, 0 |
| Refit each quarter | 42% | 0.59 → 0.50 | +0.04%, +0.01% |
| Loop (default) | 59% | 0.59 → 0.39 | +0.03%, −0.07% |
| Loop, test whenever informative | 78% | 0.59 → 0.31 | −0.27%, −0.45% |

Refitting alone is unreliable: on standard brands its error grew as the plans narrowed
the data. The loop's tests and exploration add the information that fixes that. By
default it runs a test only when the expected gain to the plan over a year
(`payback_periods=52`) exceeds the sales the holdout gives up, which roughly breaks even
while it learns. Raise `payback_periods` to learn faster at a short-term cost.

Each cycle reports `test_candidates`: every channel's expected value, expected cost and
net, so the decision not to test is as visible as the decision to test.
