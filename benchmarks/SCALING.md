# The scaling benchmark

How do time and accuracy change with the number of regions, and where does the importance
sampler (the gate) stop being enough? Every row is one fit to simulated regional data in which
each region has its own carryover and saturation curve (`make_synthetic_geo_mmm(G, seed=4,
regional_curves=0.5)`: three channels, 156 weeks), graded against the truth that generated it.

- **gate, shared curves**: the 1.0 model and sampler. One curve per channel for all regions;
  regional effects are integrated out, so the sampled parameters do not grow with regions.
- **NUTS, shared curves**: the same model on the JAX backend. It should agree with the row above.
- **gate, regional curves**: `regional_curves=True` on the gate, for small region counts only.
- **NUTS, regional curves**: `regional_curves=True` on the JAX backend, the model 1.1 adds.

Machine: 2 CPU cores, Linux-6.18.44-fc-v378-x86_64-with-glibc2.39, Python 3.13.18, Papilon 1.1.0, JAX 0.12.0.1, NumPyro 0.22.0. No GPU. Default sampler settings (gate: 4,000 draws, two starts;
NUTS: four chains, 1,000 warm-up and 1,000 kept draws each). Seconds are wall-clock for one fit, compilation
included, with nothing else running. One simulated dataset per region count, so small differences in the
accuracy columns at 4 and 8 regions (12 and 24 region-channel returns) are noise.

| Regions | Model and sampler | Sampled parameters | Time | Fit check | Regional return: error | 90% range holds it | Regional marginal return: error | 90% range holds it | National return: error |
|---|---|---|---|---|---|---|---|---|---|
| 2 | gate, shared curves | 12 | 5 s | ACCEPT | 0.29 | 83% | 0.08 | 83% | 0.28 |
| 2 | NUTS, shared curves | 12 | 15 s | passed (R-hat 1.003, ESS 2935, 0 divergent) | 0.28 | 83% | 0.08 | 100% | 0.27 |
| 2 | gate, regional curves | 30 | 39 s | SMC | 0.29 | 83% | 0.09 | 83% | 0.29 |
| 2 | NUTS, regional curves | 30 | 23 s | passed (R-hat 1.004, ESS 1739, 0 divergent) | 0.34 | 83% | 0.08 | 83% | 0.33 |
| 3 | gate, shared curves | 12 | 9 s | ACCEPT | 0.21 | 100% | 0.09 | 78% | 0.23 |
| 3 | NUTS, shared curves | 12 | 13 s | passed (R-hat 1.002, ESS 3448, 0 divergent) | 0.20 | 100% | 0.09 | 89% | 0.22 |
| 3 | gate, regional curves | 36 | 59 s | SMC | 0.24 | 78% | 0.08 | 89% | 0.23 |
| 3 | NUTS, regional curves | 36 | 20 s | passed (R-hat 1.003, ESS 2136, 0 divergent) | 0.24 | 100% | 0.09 | 100% | 0.24 |
| 4 | gate, shared curves | 12 | 9 s | ACCEPT | 0.24 | 92% | 0.09 | 92% | 0.23 |
| 4 | NUTS, shared curves | 12 | 17 s | passed (R-hat 1.002, ESS 3290, 0 divergent) | 0.23 | 92% | 0.08 | 92% | 0.22 |
| 4 | gate, regional curves | 42 | 81 s | SMC | 0.48 | 75% | 0.09 | 92% | 0.61 |
| 4 | NUTS, regional curves | 42 | 32 s | passed (R-hat 1.006, ESS 1910, 0 divergent) | 0.21 | 100% | 0.08 | 100% | 0.21 |
| 8 | gate, shared curves | 12 | 16 s | ACCEPT | 0.33 | 79% | 0.10 | 88% | 0.13 |
| 8 | NUTS, shared curves | 12 | 19 s | passed (R-hat 1.001, ESS 3094, 0 divergent) | 0.33 | 83% | 0.10 | 88% | 0.13 |
| 8 | gate, regional curves | 66 | 3 min | SMC | 0.67 | 54% | 0.12 | 75% | 0.36 |
| 8 | NUTS, regional curves | 66 | 47 s | passed (R-hat 1.005, ESS 1392, 0 divergent) | 0.30 | 96% | 0.09 | 100% | 0.12 |
| 25 | gate, shared curves | 12 | 53 s | ACCEPT | 0.31 | 69% | 0.09 | 83% | 0.14 |
| 25 | NUTS, shared curves | 12 | 35 s | passed (R-hat 1.002, ESS 4401, 0 divergent) | 0.31 | 68% | 0.09 | 85% | 0.13 |
| 25 | NUTS, regional curves | 168 | 3 min | passed (R-hat 1.005, ESS 927, 0 divergent) | 0.30 | 88% | 0.08 | 95% | 0.08 |
| 50 | gate, shared curves | 12 | 98 s | ACCEPT | 0.33 | 64% | 0.10 | 78% | 0.07 |
| 50 | NUTS, shared curves | 12 | 59 s | passed (R-hat 1.002, ESS 5389, 0 divergent) | 0.33 | 64% | 0.10 | 79% | 0.07 |
| 50 | NUTS, regional curves | 318 | 6 min | passed (R-hat 1.007, ESS 693, 0 divergent) | 0.29 | 89% | 0.08 | 91% | 0.04 |
| 100 | gate, shared curves | 12 | 4 min | ACCEPT | 0.31 | 62% | 0.10 | 82% | 0.08 |
| 100 | NUTS, shared curves | 12 | 106 s | passed (R-hat 1.002, ESS 5841, 0 divergent) | 0.31 | 62% | 0.10 | 81% | 0.09 |
| 100 | NUTS, regional curves | 618 | 11 min | passed (R-hat 1.007, ESS 422, 0 divergent) | 0.29 | 87% | 0.08 | 90% | 0.05 |

## Reading the columns

- **Sampled parameters**: what the sampler explores. Baselines, noise and each region's effect sizes are
  integrated out exactly in every row and are not counted.
- **Fit check**: the gate reports ACCEPT (importance sampling passed its effective-sample-size check) or SMC
  (it did not, and the tempered sequential Monte Carlo fallback ran). The fallback always reports a pass, so
  read its accuracy columns. NUTS passes with every R-hat below 1.01, at least 400 effective draws for
  every parameter and no divergent transitions.
- **Regional return** and **regional marginal return**: mean absolute error of the estimated return on spend,
  and of the return on 1% more spend, over every region and channel, and the share of their 90% ranges that
  hold the truth (90% is calibrated). The marginal return is what a reallocation between regions rests on,
  and it is where a region's own curve matters most.
- **National return: error**: mean absolute error of the three national returns.

Regenerate with `python benchmarks/scaling.py` (see the top of that file). Simulated regions are kinder than
real ones: the model's functional forms are the generator's, spend varies independently by region, and
nothing is missing from the data.
