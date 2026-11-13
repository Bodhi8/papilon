# RFC 004: A gradient-based sampler on a JAX backend, and regional curves

Author: Brian Curry, Vector1 Research
Status: implemented in Papilon 1.1.0
Date: 2026-11-12

Papilon's importance sampler works because the sampled dimension is small. RFC 003 kept
it small in geo models by integrating each region's effects out exactly. That stops
working the moment regions need their own curve shapes: the shapes are not conjugate,
so they must be sampled, and their number grows with the regions. This note records
how 1.1 samples them.

## 1. One posterior, two implementations

The target is unchanged: the collapsed posterior of the national model and of RFC 003, over the
unconstrained curve parameters z, with baselines and noise (and in geo models the
regional effects) integrated out in closed form.

`papilon/mmm/_jax.py` implements log p(z | data) a second time, as a JAX function of one
parameter vector, with the layout, transforms, priors and Jacobians of the NumPy code.
Two consequences:

- Draws from NUTS are plain z vectors. They go through the same `_unpack_batch` and
  `_store_draws` as draws from the gate, so every result (returns, plans, forecasts,
  saved models) is computed by the code that already existed.
- The two implementations can be compared number for number. `tests/test_jax.py`
  evaluates both at random points for every model family (each carryover and saturation
  kind, time-varying effects, reach and frequency, organic channels, lift tests, return
  priors, geo models with and without lift tests and regional curves) and requires
  agreement to six decimal places, and a finite gradient.

Three details of the port matter for gradients: powers of zero inputs (`x ** s` at
x = 0, which appear in lagged spend and in the Weibull kernel) are computed so that the
derivative in s is zero and not undefined; each region's linear system is solved by
Cholesky factorization; and the backend runs in 64-bit floats, because the marginal
likelihood subtracts quantities of similar size.

Collapsing matters as much for NUTS as for the gate. Sampling regional effects jointly
with their pooling spread is the textbook funnel; here the effects never appear in the
sampled vector.

## 2. The sampler

NUTS with step-size and mass-matrix adaptation, four chains, 1,000 warm-up and 1,000
kept draws per chain, target acceptance 0.9. The mass matrix is dense up to 30 sampled
parameters (national models have strongly correlated effect sizes and half-saturation
points) and diagonal above. Chains start at the same initial point the gate's mode
search starts from, jittered.

**Why NumPyro.** Papilon brings its own log density, so it needs a sampler, not a
modeling language. Two libraries fit: NumPyro, whose NUTS accepts a potential function
directly, and BlackJAX, a library of sampler kernels. NumPyro was chosen because it
ships the whole driver (warm-up schedule, mass-matrix adaptation, vectorized chains,
divergence bookkeeping) behind an interface that has been stable for years, and because
it adds two small dependencies to JAX, where the alternative brings an optimizer
library and its dependencies and leaves the warm-up loop to maintain here. The
dependence is confined to one function, `_jax.sample`, of about twenty lines.

**Diagnostics** are computed by Papilon in NumPy (`papilon/mmm/_diagnostics.py`):
rank-normalized split R-hat (the larger of the bulk and folded versions) and bulk
effective sample size with Geyer's initial monotone sequence. They are folded into
`GateResult`: `method="nuts"`, `rhat` (worst over parameters), `ess_bulk` (smallest),
`divergences`, and a verdict of `NUTS` when R-hat < 1.01, effective sample size is at
least 100 per chain and nothing diverged, else `NUTS_FAILED`. `passed` covers both
samplers, so nothing downstream needs to know which one ran.

**What is lost.** NUTS gives no marginal likelihood, so no Bayes factors, and it does
not move between separated modes. Family comparison therefore stays on the gate.

## 3. Choosing the sampler

`sampler="auto"` (the default) uses NUTS when the model samples more than 40 parameters
and the backend is installed. The threshold comes from the scaling benchmark
(`benchmarks/SCALING.md`): fitted to the same data, the gate's SMC fallback matches
NUTS at 30 and 36 sampled parameters, has more than twice its error in regional returns
at 42, and has lost calibration at 66, while NUTS passes its checks at all four. No national model or shared-curve geo model in the documentation or
the test suite reaches 40, so their results are the 1.0 results, draw for draw.

## 4. Regional curves

With `GeoMMM(regional_curves=True)`, every curve parameter of a channel (carryover
retention, delay, Weibull shape and scale, half-saturation, Hill shape) gets a regional
version. On the unconstrained scale, for region g and parameter i:

    z[g, i] = z[i] + s[i] * e[g, i],    e[g, i] ~ N(0, 1),    s[i] ~ HalfNormal(0.5)

z[i] is the pooled value with the prior it has in the shared model; s[i] is the spread
across regions, learned per parameter. The offsets e are sampled in this non-centered
form because a region's own curve is weakly identified by three years of its data: the
posterior of e is close to its prior, which is the geometry NUTS handles well.

Given z[g, .], region g is the conjugate model of RFC 003: its baseline and its effects
(around the pooled effect mu with spread tau) are integrated out. The sampled vector is
the shared block, the spreads and the offsets: with C channels and two curve parameters
each, 4C + 2C + 2CG numbers. For 3 channels and 200 regions that is 1,218.

The effect sizes stay partially pooled as before; `regional_spread()` reports s, and
`parameters(by_geo=True)` each region's own curve.

## 5. Lift tests in geo models

A regional lift test measures, in one region, the change in outcome from a change in one
channel's spend. Given the curve parameters, the model's prediction of that change is
linear in the region's effect for the channel: d * beta[g, c], where d is the change in
saturated carried-over spend summed over the test and readout periods. That is one more
linear observation of beta[g, c].

The region's noise scale is integrated out, and conjugacy needs every observation's
noise to be a multiple of it. A test's standard error is known in outcome units, so the
test enters as a row weighted by w = s_hat[g] / se, with s_hat[g] the plug-in noise scale
RFC 003 already uses for the pooling prior: at that scale the row has exactly the stated
standard error. The test then informs its own region directly and the others through
the pooled effect.

A test reported as one number over several regions couples their effects and breaks the
region-by-region integration. It is not supported; enter tests per region.

## 6. Forecasts in geo models

Given the curve parameters, each region's future outcome has a Student-t predictive
distribution, with the baseline and effects integrated out, exactly as in the national
model. `forecast` returns it per region and period, with carryover from the fitted
window; the national figure sums regions draw by draw.

## 7. Persistence

Saved models carry format version 2. The file layout is unchanged (JSON, CSV, NPZ; no
pickles); version 2 adds the sampler, its diagnostics, geo lift tests and the regional
draws. 1.1 reads version 1 files written by 1.0; 1.0 refuses version 2 files with a
message to upgrade.

## 8. Limits

- Time-varying effects, reach and frequency and organic channels are still national only.
- Regional curves multiply the parameters by the number of regions, and the fit time by
  more: 11 minutes at 100 regions, 31 at 200 and 53 at 400 on two CPU cores. The spreads s mix
  slowest, and their effective sample size at default settings falls as regions are
  added (about 1,900 at 4 regions, 420 at 100, 270 at 200 and at 400), so the default chain length
  stops passing the check between 100 and 200 regions. Chains twice as long at 200 regions
  reach the effective-sample bar (579) and leave the worst R-hat at 1.0102. The worst
  R-hat is a maximum over all sampled parameters, so with thousands of them it sits
  further into the tail and the 1.01 bar is harder to clear. `benchmarks/SCALING.md` has the
  measurements.
- The simulated regions used to grade this are kinder than real ones. A good score there
  is necessary, not sufficient.
