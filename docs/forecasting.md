# Planning and forecasting

**Steady-state plans.** `optimize_budget(total)` allocates a per-period budget using each
channel's response at constant spend. By default it maximizes the 10th percentile of the
gain over the current mix, holding each channel within 50% of its current spend.

**Is the plan stable?** Two channels with nearly the same marginal return let a
planner swing money between them on noise.

```python
model.indistinguishable_channels()        # pairs the posterior cannot rank
model.plan_stability(total)               # re-plans on halves of the posterior; direction_agreement per channel
model.optimize_budget(total, bagging=10)  # averages plans from 10 halves: moves less, only where they agree
```

`optimize_budget` also lists indistinguishable pairs in `attrs["summary"]`.

**Forecasts.** A plan is a path, not a steady state: cuts cost sales at once, delayed
channels pay back later, carryover from history fades out.

```python
future = model.future_frame(13, spend={"online_video": [80] * 6 + [0] * 7})   # flighting is allowed
model.forecast(future)            # predicted outcome per period, 90% range, each channel's part

model.scenarios({"current": model.future_frame(13),
                 "shift to video": model.future_frame(13, spend={"online_video": 70, "display": 10})})
```

`scenarios` evaluates every plan on the same posterior draws, so `prob_beats_first` is
the probability the plan beats the first one, not two separate guesses.
