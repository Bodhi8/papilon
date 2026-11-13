# Inference: the gate and NUTS

Papilon fits one posterior with one of two samplers. Both sample the same thing: the
curve parameters of each channel (carryover, saturation, effect size), with baselines
and noise integrated out exactly, and in geo models each region's effects too. They
differ in how many parameters they can handle and in what they report.

| | The gate (default) | NUTS (optional) |
|---|---|---|
| Method | importance sampling from a mixture over every posterior mode, with a quality gate and a tempered SMC fallback | the No-U-Turn sampler, a gradient-based Markov chain |
| Install | `pip install papilon` (NumPy and SciPy) | `pip install "papilon[jax]"` (adds JAX and NumPyro) |
| Good for | up to about 40 sampled parameters: national models, geo models with shared curves | hundreds of sampled parameters (measured to about 1,200): geo models with a curve per region |
| Reports | effective sample share, Pareto k, modes found, and the marginal likelihood | R-hat, effective sample size, divergent transitions |
| Bayes factors (`compare_models`) | yes | no |
| Several separate modes | yes, each gets a mixture component | no: chains stay in the mode they start near, and R-hat shows it |

## Choosing a sampler

```python
pp.MMM(df, "sales", media, "date")                     # sampler="auto", the default
pp.MMM(df, "sales", media, "date", sampler="gate")     # always the gate
pp.MMM(df, "sales", media, "date", sampler="nuts")     # always NUTS; needs papilon[jax]
```

`"auto"` uses the gate unless the model samples more than 40 parameters **and** the
JAX backend is installed. A model has about three sampled parameters per channel
(more with delayed or Weibull carryover, S-curves, time-varying effects), so every
national model in these guides, and every geo model with shared curves, stays on the
gate and gives the results it gave in 1.0. If the model is larger than that and the
backend is not installed, Papilon fits with the gate and warns: at that size the
fallback sampler still returns draws, but the [scaling benchmark](https://github.com/Bodhi8/papilon/blob/main/benchmarks/SCALING.md)
shows they stop covering the truth.

The threshold is `papilon.mmm.model.AUTO_NUTS_DIM`. `model._dim(model.family)` is the
number of sampled parameters.

## Reading the result

`model.gate_results[name]` is a `GateResult` for either sampler, and `.passed` is the
one question to ask of it:

```python
g = model.gate_results[model.family.name()]
g.method        # "importance", "smc" or "nuts"
g.passed        # can the draws be relied on?
g.rhat, g.ess_bulk, g.divergences      # NUTS; NaN, NaN and 0 for the gate
g.ess, g.pareto_k, g.log_evidence      # gate; for NUTS ess is ess_bulk / draws, the rest NaN
```

For NUTS, `passed` is true (verdict `"NUTS"`) when all three hold, and false (verdict
`"NUTS_FAILED"`) otherwise:

- **R-hat below 1.01 for every sampled parameter.** R-hat compares the chains with each
  other; above 1.01 they have not settled on the same distribution. Papilon uses the
  rank-normalized split version, which also reacts to chains that agree on the center
  and disagree on the spread.
- **At least 100 effective draws per chain for every parameter** (400 with the default
  four chains). Fewer, and the 90% ranges are not stable.
- **No divergent transitions.** A divergence means the sampler met a region it could not
  follow, and may have skipped it.

One chain cannot show that chains agree, so a one-chain fit never passes. The learning
loop, the HTML report, MLflow logging and the AI tools all read `passed`, so a failed
NUTS fit is flagged the way a failed gate is.

## Tuning NUTS

```python
model.fit(chains=4, warmup=1000, samples=1000, target_accept=0.9)   # the defaults
```

- `NUTS_FAILED` on R-hat or effective sample size: run longer (`warmup=2000, samples=2000`).
  Expect this in geo models with a curve per region beyond about 100 regions: in the
  scaling benchmark the smallest effective sample size at default settings was about 1,900
  at 4 regions, 420 at 100 and 270 at 200, where the check failed with no divergences and
  accurate estimates.
- Divergences: raise `target_accept` (0.95, then 0.99). Steps get smaller and the fit slower.
- R-hat far above 1.01 that does not improve with length usually means separate modes.
  Fit with `sampler="gate"` if the model is small enough; otherwise look at which
  parameter disagrees (a carryover that is either very short or very long is the usual one)
  and fix its form or tighten its prior.

The gate's own arguments (`n_draws`, `n_starts`, `accept`, `refine`, `max_refine`) are
ignored by NUTS, and NUTS's by the gate. NUTS keeps 2,000 evenly spaced draws (1,000 in
geo models) for everything downstream, so returns, plans, forecasts, scenarios, saved
models and reports work the same after either sampler.

## What NUTS does not do

- **Bayes factors.** NUTS estimates no marginal likelihood. `compare_models` and
  `compare_carryover` always run the gate, and raise under `sampler="nuts"`. Choose the
  family with the gate (in geo models, with shared curves), then fit the winner with NUTS.
- **Separate modes.** See the table above. The gate's mixture proposal is the better tool
  for a small multimodal posterior.
- **A faster small model.** On the bundled four-channel brand the gate takes about a
  second and NUTS about ten, most of it compilation, for the same returns. NUTS earns its
  place only where the gate cannot go.

## The backend

The JAX backend is a second implementation of the same log posterior, in the same
parametrization, so draws from either sampler are unpacked by the same code. A test
compares the two implementations to six decimal places for every model family, and
on models small enough for both, compares the fitted returns.

Using the backend switches JAX to 64-bit floats for the whole Python process
(`jax_enable_x64`): the closed-form marginal likelihood loses too much precision in 32
bits. Fits run on the CPU unless your JAX install has an accelerator. Results are
reproducible for a given `seed` on the same machine and library versions.

Geo models with a curve per region are described in [Geo models](geo.md); the method is
recorded in [RFC 004](design/004-nuts-backend.md).
