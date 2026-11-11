# Checking the model

Each check answers a different doubt. None of them is sufficient alone.

| Doubt | Check | Passing looks like |
|---|---|---|
| Did the fit work? | `verdict`, `pareto_k` in `model.gate_results` | `ACCEPT`, Pareto k under 0.7 |
| Is the carryover or curve shape right? | `compare_carryover()`, `compare_models()` | One family with most of the probability |
| Does it predict sales it has not seen? | `backtest(13)` | Error below the naive forecast; about 90% of weeks inside the range |
| Could it recover returns like these from data like mine? | `self_check(12)` | Coverage near 90%; plans that do not lose money |
| What if something I cannot measure moves with spend? | `self_check(12, confounded={"paid_search": 0.5})` | Shows the bias such a driver would cause |
| Which estimate should I test in the real world? | `value_of_lift_test()` | A ranking, with the expected gain to the plan |
| Does the plan hold up over time? | `pycausalsim.futures.media_plan_scm` | Gain positive by the end of the quarter |

## Backtest

```python
bt = model.backtest(holdout=13)
bt.attrs["summary"]     # mape, naive_mape, coverage_90
```

The model is refit without the last 13 periods and forecasts them using the spend that
actually ran. A model can forecast well and still credit the wrong channels, because
two channels that move together can swap credit without changing the forecast.

## Self-check

```python
sc = model.self_check(n_worlds=12)
sc.attrs["summary"]     # coverage_90, bias_by_channel, plan_lost_money, plan_capture
```

Papilon builds simulated copies of your brand: your spend, your controls, your
calendar, and channel returns drawn from the fitted model. It refits to each copy and
grades the answers. Low coverage or large bias for a channel means your data cannot
pin that channel down, usually because its spend follows the season or another channel.

With twelve worlds the figures are rough: coverage of 80% to 100% is consistent with a
calibrated model. Use more worlds for a firmer number; each takes a few seconds.

The self-check assumes the model's form is right. It cannot find a driver you never
thought of. That is what lift tests are for.

## The value of a lift test

```python
model.value_of_lift_test(sd=0.25)
```

For each channel: if you tested it and learned its return to within `sd`, how much
better would the budget plan be? Channels whose range is wide and whose spend is large
rank first. `expected_gain` is in sales per period; treat it as a ranking.

After a test, add the result and refit:

```python
model = pp.MMM(df, "sales", media, "date", roi_priors={"paid_search": (1.2, 0.25)})
```
