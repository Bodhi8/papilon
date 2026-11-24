# The scenario planner

A marketing mix model earns its keep when someone who controls a budget acts on it.
Most of those people will never open Python. `scenario_planner` turns a fitted model
into one HTML file they can open in any browser and plan with: no code, no server, no
login, and nothing sent anywhere.

```python
import papilon as pp

model = pp.MMM(df, "sales", media, "date", controls=controls).fit()
pp.scenario_planner(model, "planner.html", currency="$", units="thousands")
```

Or from a terminal, without writing any code:

```bash
python -m papilon.planner --data sales.csv --target sales --date week \
    --media search,social,video --controls price,promo -o planner.html

python -m papilon.planner --model fitted.papilon -o planner.html   # a saved model
papilon-planner --model fitted.papilon -o planner.html             # same, installed script
```

Send the file, attach it to a deck, or host it as a static page. It works offline and
weighs about 150 KB for a four-channel model.

## What the planner does

**Start from** the average spend of the whole history or of the last 52, 26 or 13
weeks (days and months get matching windows). That mix is what every scenario is
compared with.

**Pick a goal:**

| Goal | What it finds |
|---|---|
| Best mix for a fixed budget | The split of a given total that does best. *Protect the downside* (the default) maximizes the 10th percentile of the gain over the starting mix; *maximize the expected outcome* maximizes the mean. |
| Grow while ROI holds | The largest total whose best split keeps total ROI at or above a target. |
| Spend to a marginal return | Each channel's spend up to where the next unit returns a target (1.0 means the last unit just pays for itself). |
| Build my own | Sliders for every channel, with live results. |

**Set channel limits:** how far each channel may move from its starting spend (±50% by
default, as in `optimize_budget`), a ceiling as a multiple of its historical average,
and optional per-channel minimums, maximums and locks.

**Read the result:** spend, the outcome driven by media and ROI over a horizon you
choose, each with its 90% range; the gain over the starting mix and the chance the
scenario beats it; a chart and table of spend by channel; and every channel's response
curve with its 90% band, the starting and scenario points, and a hatched zone past the
spend the brand has sustained. The page warns when a plan pushes a channel into that
zone and when it moves money between channels the model cannot tell apart.

**Save and share:** save scenarios to compare them side by side, download them as CSV,
or copy a link that reopens the current scenario and the saved ones.

A second tab shows the model: fit (R², MAPE, wMAPE), what drove the outcome, ROI by
channel with ranges and marginal ROI, the quality gate, and every caution
`papilon.ai.brief` raises.

## How it computes

The page carries 500 draws (by default) of each paid channel's steady-state response
curve: the same curves `optimize_budget` uses. A test checks that the page's curves
equal `MMM._steady` to machine precision, for national and regional models.

Plans are solved in the browser, typically in under 50 milliseconds:

- the expected outcome is split exactly on a grid by dynamic programming, so S-shaped
  curves, where greedy allocation stalls at zero, are handled, then refined;
- the downside objective is searched from the starting mix and from the best point on
  the path to the expected-outcome plan, by transfers between pairs of channels with
  shrinking steps;
- for *grow while ROI holds*, one grid pass gives the best outcome at every budget, and
  the planner keeps the largest budget that meets the target.

The tests run the page's engine under Node.js and check that, on the same draws, its
plans do at least as well as `optimize_budget` (within 2% on the downside objective,
which is rough).

## Compared with Meridian's Scenario Planner

Google's Meridian Scenario Planner builds a Looker Studio report from a trained Meridian
model, with fixed and flexible budgets, target ROI and marginal ROI, and channel
constraints. Papilon's planner covers the same planning modes, and differs in a few
ways:

- **One file, no accounts.** It needs no Google Drive, Cloud Storage, Colab or Looker
  Studio, and no GPU to precompute a response grid.
- **Ranges everywhere.** Each result carries its 90% range and the probability it beats
  the starting mix, and the default fixed-budget plan protects the downside instead of
  maximizing the mean.
- **Extrapolation is visible.** Spend beyond what the brand has sustained is shaded and
  flagged, and so are moves between channels the posterior cannot rank.

Like Meridian's planner, it reports incremental outcome from media, not a forecast of
total sales. For that, play a plan forward with `model.forecast`.

## Limits

- Results are steady state: they ignore the periods carryover takes to build up after a
  change. `model.forecast` and `model.scenarios` play plans forward period by period.
- Regional models plan national spend split across regions in their historical shares,
  and their page has no fit chart (`GeoMMM.decompose` is not available yet).
- `planner_data(model)` returns the page's content as plain data, for building a hosted
  front end with the same numbers.
