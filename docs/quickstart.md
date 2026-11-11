# Quickstart

You need one row per week (or other period) with a date, sales, and spend per channel
in the same currency as sales. Add columns for anything else that moves sales and that
you can measure: price, promotions.

```bash
pip install papilon
```

```python
import papilon as pp

df, _ = pp.make_synthetic_mmm()          # or pd.read_csv("your_data.csv", parse_dates=["date"])
media = ["paid_search", "paid_social", "online_video", "display"]

model = pp.MMM(df, target="sales", media=media, date_col="date", controls=["price_index", "promo"], seed=11)
model.compare_carryover({"all_geometric": [], "video_delayed": ["online_video"]})
```

`compare_carryover` fits each candidate and keeps the one the data supports. Check the
`verdict` column: `ACCEPT` means the fit passed its own quality gate.

```python
model.get_roi()                          # return on spend per channel with a 90% range
```

Read the range, not only the mean. If two channels' ranges overlap, the data has not
established which is better.

```python
plan = model.optimize_budget(total=df[media].sum(axis=1).mean())
plan.attrs["summary"]                    # expected gain, its range, probability the plan beats today's mix
```

The default plan moves each channel at most 50% and is built to hold up if the model is
wrong. Before acting on it:

```python
model.backtest(13).attrs["summary"]      # does it forecast weeks it has not seen?
model.self_check(12).attrs["summary"]    # could it recover returns like these from data like yours?
model.value_of_lift_test()               # which channel is worth testing first
```

Next: [declare what you know about the brand](knowledge-graph.md), or
[read what each check means](checking.md).
