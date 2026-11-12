# Open marketing mix model leaderboard

Every entry is fitted to the same simulated brands, whose true returns are known, and scored on
what a marketer needs from a model: ranges that hold the truth, and budget plans that make money.
Brands: the first 16 of `marketing_worlds(60, seed=7, ...)` per track, PyCausalSim 0.6.0. Regenerate with `python benchmarks/leaderboard.py`; to add an entry see
[SUBMITTING.md](SUBMITTING.md).

## Track: standard

Search spend follows demand; launches are visible in spend.

| Entry | 90% range holds the truth | ROAS error | Bias | Best gain captured | Plans that lost money |
|---|---|---|---|---|---|
| Papilon 1.0, defaults | 98% | 0.32 | +0.13 | 48% | 0% |
| Least squares, fixed adstock | 45% | 0.66 | -0.33 | -24% | 50% |

## Track: hard

Adds unobserved launches that lift spend and sales together, and creative wear-out.

| Entry | 90% range holds the truth | ROAS error | Bias | Best gain captured | Plans that lost money |
|---|---|---|---|---|---|
| Papilon 1.0, defaults | 69% | 0.92 | +0.82 | 29% | 31% |
| Least squares, fixed adstock | 42% | 0.98 | +0.34 | -24% | 44% |

## Track: s_curve

Every channel's true response is S-shaped.

| Entry | 90% range holds the truth | ROAS error | Bias | Best gain captured | Plans that lost money |
|---|---|---|---|---|---|
| Papilon 1.0, defaults | 55% | 0.80 | +0.78 | 38% | 6% |
| Papilon 1.0, curve shape chosen | 81% | 0.52 | +0.43 | 37% | 0% |
| Least squares, fixed adstock | 64% | 0.70 | +0.25 | 4% | 50% |

## Reading the columns

- **90% range holds the truth**: share of channel ranges containing the true return. 90% is calibrated.
- **ROAS error** and **Bias**: mean absolute and mean signed error of the estimate.
- **Best gain captured**: true gain of the recommended plans as a share of the best possible gain,
  with each channel free between zero and three times its history. Can be negative.
- **Plans that lost money**: share of brands where the recommended plan did worse than the current mix.

## Entries

- **Papilon 1.0, defaults**: carryover chosen by Bayes factor, downside-protected plan.
- **Papilon 1.0, curve shape chosen**: also chooses concave or S-shaped saturation by Bayes factor.
- **Least squares, fixed adstock**: reference baseline: regression on adstocked spend, no saturation.

Simulated brands are kinder than real data in ways the generator does not capture. A good score
here is necessary, not sufficient.
