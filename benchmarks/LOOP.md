# The learning-loop benchmark

Does a marketing mix model that is refit every quarter converge on the truth, or lock in its
own error? Each policy below ran for six quarters on the same simulated brands
(`marketing_worlds(12, seed=7)`, first 8 per track), after two years of history. The
world responds to the spend each policy chose; recommended tests run in the world and come
back with the noise of a matched-market design (half the market held out, 90% of noise
removed by the pre-period model).

- **static**: fit once and keep the plan.
- **refresh**: refit every quarter on all the data. No tests, no exploration.
- **loop**: `papilon.Loop` defaults. Refit, a budget-neutral exploration pulse of 10% between the
  two channels the model can least tell apart, and a lift test only when its expected gain to the
  plan over a year exceeds the sales the holdout gives up.
- **loop_learning**: the loop with a test whenever one is informative, whatever it costs.

## Standard brands

| Policy | Error in returns, quarter 1 → 6 | Best gain captured, quarter 1 → 6 | Plans losing money, quarter 6 | Sales over six quarters vs static | Quarters with a test |
|---|---|---|---|---|---|
| static | 0.37 → 0.37 | 34% → 34% | 12% | +0.00% | 0% |
| refresh | 0.37 → 0.40 | 34% → 42% | 0% | +0.04% | 0% |
| loop | 0.37 → 0.39 | 34% → 59% | 0% | +0.03% | 23% |
| loop_learning | 0.37 → 0.32 | 34% → 78% | 0% | -0.27% | 73% |

## Hard brands (unobserved launches, creative wear-out)

| Policy | Error in returns, quarter 1 → 6 | Best gain captured, quarter 1 → 6 | Plans losing money, quarter 6 | Sales over six quarters vs static | Quarters with a test |
|---|---|---|---|---|---|
| static | 0.59 → 0.61 | 33% → 33% | 25% | +0.00% | 0% |
| refresh | 0.59 → 0.50 | 33% → 35% | 25% | +0.01% | 0% |
| loop | 0.59 → 0.39 | 33% → 40% | 25% | -0.07% | 23% |
| loop_learning | 0.59 → 0.31 | 33% → 56% | 12% | -0.45% | 79% |

## Reading it

Refitting alone is unreliable. On standard brands its error grew as plans narrowed the data; on
hard brands it improved, but less than the loop. The loop's tests and exploration add the
information that moves estimates toward the truth and plans toward the best available.
Learning has a price: testing every quarter learns fastest and costs sales while it does. The
default tests only when a test is expected to pay for itself, which roughly breaks even on
sales over six quarters while its plans improve. `payback_periods` sets the trade-off.

Eight brands per track and six quarters is a small sample; differences of a few points are noise.
Simulated brands are kinder than real ones. Reproduce with `python benchmarks/loop.py`.
