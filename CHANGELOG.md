# Changelog

## 0.4.1 (2026-11-07)

### Fixed
- Display, and any channel with a small share of spend, was overrated. Every channel
  had the same half-normal prior (scale 0.5) on its effect in units of mean sales,
  which for a channel spending little implied a prior return on spend of roughly 4 to
  10. The prior is now set on the ROAS scale for each channel: a half-normal with
  scale `roas_prior_scale` (default 2.0, prior median about 1.35) on its return at
  typical spend. On the 120-brand benchmark, display's average overstatement fell
  from +0.49 to +0.21 on standard brands and roughly halved in every other setting;
  paid search's fell from +0.72 to +0.26. Interval coverage held or improved
  (display 90% to 95%). The default plan lost money for 2% of standard brands
  instead of 7%, and for 37% instead of 45% when launches were hidden, while
  capturing a similar share of the available gain (43% vs 46% on standard brands).

### Added
- `MMM(roas_prior_scale=...)`: one number or a dict by channel (channels left out
  keep 2.0). `None` restores the 0.4.0 prior.

### Changed
- `benchmarks/recovery.py` spells out each plan rule's objective and caps, so rules
  such as `mean` no longer inherit the 0.4 defaults.

## 0.4.0 (2026-11-06)

### Changed
- `optimize_budget()` defaults to `objective="quantile"` (maximize the 10th percentile
  of the gain over the current mix) and `max_change=0.5` (each channel within 50% of
  current spend). On 120 simulated brands the 0.3 rule (maximize expected sales, 0 to
  3x history) captured 32% of the best possible gain and lost money for 33% of
  standard brands; the new default captured 46% and lost money for 7%. When launches
  were hidden from the model, uncapped rules lost money on average and every capped
  rule, including this one, did not. `objective="mean", max_change=None` restores 0.3 behavior.
- Fits are about 11 times faster: prior log densities are computed in closed form
  instead of through `scipy.stats`, whose per-call overhead was about 87% of fit time.
  Results match to floating point, so Monte Carlo draws can differ slightly from 0.3.

### Added
- `optimize_budget(support=...)`: flags (or with `"limit"` caps) channels planned
  above the 95th percentile of their sustained historical spend. Plans carry
  `attrs["summary"]` with the expected gain, its 90% interval and the probability of
  beating the current mix; the result has an `outside_support` column.
- `MMM(roi_priors={channel: (mean, sd)})` to calibrate channel returns to lift tests
  or geo experiments.
- `MMM(alpha_prior=...)` for the carryover prior, globally or by channel.
- `MMM.evaluate_plan()` and `MMM.posterior_curves()`, the hand-off to simulators such
  as `pycausalsim.futures.media_plan_scm`.
- `GateResult.pareto_k`, the PSIS tail diagnostic for the importance weights.
- `benchmarks/recovery.py` and `benchmarks/summarize.py`.

## 0.3.0 (unreleased)

### Added
- `papilon.MMM`: a Bayesian marketing mix model fit by gated importance sampling,
  following the inference path in RFC 001 (`papilon-amortized-inference-design.md`).
  - Geometric and delayed adstock, Hill saturation, positive channel effects.
  - Baseline, trend, yearly seasonality and controls integrated out in closed form.
  - An effective-sample-size gate that accepts, refines or escalates the proposal.
  - `compare_carryover()` to choose carryover shapes by Bayes factor.
  - `get_roi()`, `parameters()`, `contributions()`, `response_curve()` and
    `optimize_budget()`.
- `papilon.make_synthetic_mmm()`: weekly marketing data with a known ground truth.
- A test suite (`tests/`) and a quickstart example (`examples/mmm_quickstart.py`).

### Changed
- License changed from a proprietary license to the Apache License, Version 2.0.
- README rewritten to describe only released functionality; planned work moved
  to a Roadmap section.
- Package metadata fixed (version, license, classifiers, dependencies).

### Restored
- The 0.2.0 modules (entropy, energy, states, relationships, causal, simulation,
  optimizer, topology), which were published on PyPI but had been removed from
  the repository.

## 0.2.0 (2025-04-20)

- First public release on PyPI: entropy and energy measures, microstate counting,
  KDE scenario simulation, relationship analysis, a dependency screen
  (`infer_causal_structure`) and grid-search optimization.
