# Papilon

<p align="center">
  <img src="https://img.shields.io/badge/version-0.3.0-blue.svg" alt="Version 0.3.0">
  <img src="https://img.shields.io/badge/python-3.10+-green.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-Apache%202.0-lightgrey.svg" alt="Apache 2.0 license">
</p>

<p align="center"><em>"Does the flap of a butterfly's wings in Brazil set off a tornado in Texas?" Edward Lorenz</em></p>

Papilon (French for butterfly) is a Python library for marketing mix modeling and for
studying how small changes move complex systems. Version 0.3.0 adds a Bayesian marketing
mix model that chooses its own carryover shape by Bayes factor and reports when its
posterior can't be trusted. It also includes the statistical-mechanics and simulation
tools first released in 0.2.0.

## Install

```bash
pip install papilon
```

## Quickstart: a marketing mix model with a known truth

`make_synthetic_mmm()` creates three years of weekly sales for a brand running four
digital channels, and returns the parameters that generated it, so you can check every
estimate against the truth.

```python
import papilon as pp

df, truth = pp.make_synthetic_mmm()

model = pp.MMM(
    data=df,
    target="sales",
    media=["paid_search", "paid_social", "online_video", "display"],
    date_col="date",
    controls=["price_index", "promo"],
    yearly_seasonality=2,
    seed=11,
)

# Does online video's effect peak after the week it runs? Let the data decide.
families = model.compare_carryover({
    "all_geometric": [],
    "video_delayed": ["online_video"],
})

roi = model.get_roi()                                   # ROAS with 90% intervals
plan = model.optimize_budget(total=df[model.media].sum(axis=1).mean())
```

Output from `examples/mmm_quickstart.py`, abridged (spend and sales in $ thousands):

```
               log_evidence  log_bayes_factor_vs_best  probability  ess_fraction verdict
all_geometric       241.223                    -18.72          0.0         0.134  ACCEPT
video_delayed       259.943                      0.00          1.0         0.197  ACCEPT

                spend  roas_mean  roas_p05  roas_p95  mroas_mean  true_roas
paid_search   4729.95       0.64      0.03      1.65        0.32       1.25
paid_social   7678.08       1.02      0.47      1.79        0.49       1.50
online_video  7369.53       1.76      1.50      2.05        0.88       1.57
display       3497.10       0.71      0.08      1.68        0.35       0.74
```

The model picks the delayed carryover shape for online video and puts its peak at 2.0
weeks, the true value. Every true ROAS falls inside its 90% interval. The point
estimates for paid search and paid social are low: search spend in this dataset follows
seasonal demand, which makes it hard to separate from seasonality with three years of
weekly data. That is the job lift tests do, and calibration to experiments is on the
roadmap.

## How the model is fit

`pp.MMM` follows the inference path in
[RFC 001, amortized structural inference](papilon-amortized-inference-design.md):

1. **Proposal.** A distribution over the media parameters (carryover, saturation,
   effect size). In 0.3.0, a Laplace fit at the posterior mode stands in for the
   trained amortized network the RFC describes.
2. **Importance sampling** against the exact posterior. Baseline, trend, seasonality,
   controls and noise are integrated out in closed form, so sampling covers only the
   media parameters.
3. **Effective-sample-size gate.** If the proposal matches the posterior well, it is
   accepted. If not, it is refined from the weighted draws and tried again, or the fit
   is escalated. `model.gate_results` records every step.
4. **Bayes factors.** The same importance weights estimate each family's marginal
   likelihood, so `compare_carryover()` can choose between geometric and delayed
   adstock per channel.

Channel effects are constrained to be positive. The single-channel prototypes behind
this design, with their results, are in [`prototypes/`](prototypes/).

## Other tools (from 0.2.0)

```python
import papilon as pp

pp.shannon_entropy(values)                               # information content
pp.simulate_kde_scenarios(df, columns, n_samples=1000)   # scenario simulation
pp.analyze_relationships(df, target="sales")             # correlation and mutual information
pp.infer_causal_structure(df, threshold=0.01)            # pairwise dependency screen
pp.grid_search_optimize(df, feature_ranges, objective_fn, constraints)
```

`infer_causal_structure` tests each pair of variables for a significant linear
relationship. It is a screen for dependencies, not causal discovery: it does not orient
edges or adjust for confounders.

The notebook `Papilon_Marketing_Optimization_Workflow.ipynb` walks through these tools.

## Tests

```bash
pip install -e ".[dev]"
pytest -m "not slow"     # about 30 seconds
pytest                   # includes the full Bayes-factor recovery test, a few minutes
```

## Roadmap

These are planned and **not in the released package**:

- The trained amortized network from RFC 001 in place of the Laplace proposal
- Calibration to lift tests and geo experiments
- Geo-level models with partial pooling
- Weibull adstock and logistic saturation
- Causal discovery and effect estimation beyond the dependency screen

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Citation

If you use Papilon in research, please cite:

```
Curry, B. (2026). Papilon: Bayesian marketing mix modeling with gated importance
sampling (Version 0.3.0). https://github.com/Bodhi8/papilon
```

Built by [Brian Curry](https://vector1.ai), Vector1 Research.
