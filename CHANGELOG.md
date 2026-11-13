# Changelog

## 1.1.0 (unreleased)

Papilon 1.1 adds a gradient-based sampler for models too large for importance sampling,
and uses it for regional models with a curve per region across hundreds of markets. The
default install still needs only NumPy and SciPy, and default fits give the numbers they
gave in 1.0.0.

### Added: inference
- `sampler="auto" | "gate" | "nuts"` on `MMM` and `GeoMMM`. `"gate"` is the importance
  sampler with its quality gate and SMC fallback. `"nuts"` is the No-U-Turn sampler on an
  optional JAX backend. `"auto"`, the default, uses the gate unless the model samples more
  than 40 parameters and the backend is installed.
- Optional extra `papilon[jax]` (JAX and NumPyro). The backend is a second implementation
  of the same collapsed posterior: baselines, noise and regional effects stay integrated
  out, and NUTS samples the curve parameters. Tests hold the two implementations to six
  decimal places for every model family.
- `fit(chains=4, warmup=1000, samples=1000, target_accept=0.9)` steer NUTS.
- `GateResult.rhat`, `.ess_bulk` and `.divergences`, with `method="nuts"` and the verdicts
  `NUTS` and `NUTS_FAILED`. `GateResult.passed` is true for NUTS when every R-hat is below
  1.01, every parameter has at least 100 effective draws per chain and no transition
  diverged. The loop, reports, MLflow logging and AI tools read `passed` as before.

### Added: geo models
- `GeoMMM(regional_curves=True)`: every region gets its own carryover and saturation
  parameters, partially pooled toward the shared ones, with `regional_spread()` and
  `parameters(by_geo=True)`. Regional effects are still integrated out exactly.
- `GeoMMM(lift_tests=...)`: lift tests by region, calibrating the region and, through
  pooling, the others.
- `GeoMMM.future_frame`, `forecast` and `scenarios`: forecasts per region and nationally.
- `get_roi(by_geo=True)` adds each region's marginal return (`mroas_mean`, `mroas_p05`,
  `mroas_p95`).
- `make_synthetic_geo_mmm(regional_curves=...)` simulates regions with their own curves;
  the truth it returns gains each region's `alpha`, `half_sat` and `max_effect`.
- `examples/regional_curves_nuts.py`.

### Added: benchmarks and documentation
- `benchmarks/scaling.py` and [SCALING.md](benchmarks/SCALING.md): time and accuracy
  against known truth from 2 to 200 regions, gate against NUTS.
- A NUTS entry on the leaderboard.
- An inference guide (`docs/inference.md`) and RFC 004 (`docs/design/004-nuts-backend.md`).
- CI job `jax-backend`: the extra on Linux, Windows and macOS. The main test job installs
  neither JAX nor NumPyro and deselects the backend tests (`-m "not slow and not jax"`).

### Changed
- The default `sampler="auto"` sends a model with more than 40 sampled parameters to NUTS
  when the backend is installed. Benchmark result behind the threshold
  ([SCALING.md](benchmarks/SCALING.md)): on the same simulated
  regions the gate's fallback matches NUTS at 36 sampled parameters, has more than twice
  its error in regional returns at 42, and at 66 its 90% ranges hold the truth 54% of the
  time against 96%.
  Smaller models, and every install without the backend, fit as in 1.0.0.
- Saved models are written in format version 2 (sampler, diagnostics, regional draws, geo
  lift tests). 1.1 reads files written by 1.0; 1.0 refuses 1.1 files with a message to
  upgrade.
- Geo models with very many regions compute returns region by region instead of holding
  every draw, region, period and channel in memory at once.

### Benchmarks
- Default fits: 21 seeded outputs (returns, gate statistics, plans, forecasts, the SMC
  fallback, family comparison, geo returns and pooling) recorded on 1.0.0 are reproduced
  bit for bit by 1.1.0 with the same library versions; `tests/test_v110.py` keeps a subset
  as a regression test, and loads model files saved by 1.0.0.
- Leaderboard, rerun: the default entries score exactly as in 1.0.0. The NUTS sampler,
  with the family fixed, scores within sampling noise of the gate: on standard brands the
  90% range holds the truth 97% of the time (gate 98%) and plans capture 48% of the best
  gain (gate 48%); on hard brands 69% and 28% (gate 69% and 29%).
- Loop benchmark, rerun: reproduces the 1.0.0 results to ten decimal places.
- Scaling: with a curve per region, NUTS passed its checks from 2 to 100 regions (618
  sampled parameters, 11 minutes on two CPU cores, no GPU). From 25 to 200 regions its
  regional 90% ranges hold the truth 87% to 90% of the time, against 62% to 69% with
  shared curves; the error of regional returns improves by less than a tenth. At 200
  regions (1,218 parameters, 31 minutes) default settings failed the convergence check
  narrowly (R-hat 1.013, 266 effective draws against 400) with accuracy unchanged; fits
  that large need longer chains.

### Not in this release
- Bayes factors under NUTS: `compare_models` and `compare_carryover` always use the gate
  and raise under `sampler="nuts"`. In geo models families are compared with shared curves.
- Lift tests measured as one number over several regions.
- Time-varying effects, reach and frequency and organic channels in geo models.

## 1.0.0 (2026-11-12)

Papilon 1.0 is a production release: the public API is stable under semantic
versioning (docs/api-stability.md), every release is tested on Linux, Windows and
macOS, and fitted models can be saved, tracked, monitored and reported.

### Added: modeling
- Carryover families `geometric`, `delayed` and `weibull`; saturation families `hill`,
  `s_curve` and `logistic`; chosen per channel (`carryover=`, `saturation=`) and compared
  by Bayes factor with `compare_models`.
- Time-varying channel effects (`time_varying=[...]`) for creative wear-out, and slow
  baseline drift (`baseline_drift=`). Plans and forecasts use the latest effect.
- Reach and frequency inputs (`reach_frequency=`), with the optimal frequency reported.
- Organic channels (`organic=`), modeled with carryover and saturation, reported, and left
  out of returns and plans.
- Lift tests in the likelihood on the weeks they ran (`lift_tests=`): the model predicts the
  lift of the same spend change in the same periods, carryover included.
- `GeoMMM`: regional models with partial pooling of effects across regions. Baselines and
  regional effects are integrated out exactly, so sampling stays low-dimensional.
- `decompose()`: outcome split into baseline, each control and each channel.

### Added: inference
- Mixture proposals built from every distinct posterior mode found.
- Tempered sequential Monte Carlo fallback when importance sampling fails its gate
  (`escalate="smc"`, the default); `GateResult.method`, `.modes` and `.passed`.

### Added: planning and forecasting
- `forecast`, `future_frame` and `scenarios`: play spend plans forward period by period,
  flighting included, with paired comparisons between plans.
- `plan_stability`, `indistinguishable_channels`, and `optimize_budget(bagging=n)` for plans
  that do not swing on noise between channels the posterior cannot rank.
- `Loop`: the learning loop. Each cycle refreshes on new periods and experiment results,
  scores the last forecast, recommends the lift test worth the most to the plan, and adds
  a small budget-neutral exploration pulse between the channels the model can least tell
  apart. `pycausalsim.futures.compare_loops` grades it against known truth.

### Added: causal structure
- `discover_structure`: the PC algorithm with partial-correlation tests, background tiers and
  forbidden edges, after removing trend and season.
- `MarketingGraph.adjustment_set`, `identification` (back-door criterion on the declared graph)
  and `check_against_data` (declared relations against discovered ones).

### Added: production
- `MMM.save` / `MMM.load`: one zip of JSON, CSV and NumPy arrays; no pickles.
- `papilon.io`: `validate_frame`, `read_sql`, `from_snowflake`, `from_bigquery`, `long_to_wide`.
- `papilon.tracking.log_to_mlflow`.
- `monitor(new_data)`: coverage, standardized error and spend outside support on new periods.
- `papilon.report`: charts and a self-contained HTML report.
- AI tools `plan_stability` and `forecast`, also over the Model Context Protocol.
- CI on three operating systems, a documentation site, an API snapshot test, a deprecation
  policy, and MAINTAINERS.md with a continuity and succession plan.
- Optional extras: `mlflow`, `snowflake`, `bigquery`, `sql`, `all`.
- `make_synthetic_geo_mmm` for testing regional models.

### Changed
- `GateResult` gains `carryover`, `saturation`, `time_varying`, `method`, `modes`, `passed`.
- `posterior_curves()` adds `kernel` and `saturation`; `beta` is the effect as of the last period.
- `parameters()` returns more columns when the model uses the new families.
- Posterior draws differ from 0.5 for the same seed (the proposal is now a mixture); estimates
  agree within their ranges.

### Deprecated
- `infer_causal_structure` (pairwise screen) in favor of `discover_structure`; removal in 2.0.

## 0.5.0 (2026-11-11)

### Added
- `MarketingGraph`: a typed knowledge graph of the brand (channels with funnel stage,
  drivers, events, experiments; `drives`, `sets_spend`, `mediates`, `belongs_to`,
  `measured_by`). It compiles to model settings (`model_kwargs()`, `build_model()`,
  `carryover_candidates()`), audits the setup (`audit()` names every channel that
  depends on a declared driver with no data, and every mediated effect), answers
  "what else moves?" (`what_moves()`), holds unconfirmed edges apart (`propose()`,
  `confirm()`), and exports to plain data, JSON-LD and prose. The same graph feeds
  `pycausalsim.futures.worlds_from_graph`.
- S-shaped saturation: `MMM(s_curve=[...])` fits a Hill shape for those channels, and
  `compare_models()` chooses between concave and S-shaped curves by Bayes factor, the
  way `compare_carryover()` chooses carryover. `posterior_curves()` carries `shape`.
- `MMM.self_check()`: refits the same settings to simulated copies of the brand with
  known returns and reports interval coverage, bias and whether the default plan made
  or lost money. `confounded={channel: strength}` shows what an unmeasured driver
  would do to the estimates.
- `MMM.value_of_lift_test()`: ranks channels by the expected gain to the budget plan
  from testing each, by reweighting posterior draws on simulated test results.
- `MMM.backtest(holdout)`: refit without the last periods and forecast them, with a
  naive forecast for comparison.
- `papilon.ai`: `brief()` (facts and rule-based cautions as plain data), `ToolBox`
  (the model as tools with JSON schemas in the layouts chat model APIs accept, results
  carrying their caveats) and `SYSTEM_PROMPT`. No language model is called and no
  data leaves the machine.
- `python -m papilon.mcp_server`: the same tools over the Model Context Protocol
  (optional extra `papilon[mcp]`).
- `MMM.roas_draws()`: per-draw ROAS, marginal ROAS and share of sales.

### Changed
- The posterior is evaluated for whole batches of draws at once. Importance sampling,
  the Hessian, `get_roi()` and `contributions()` are vectorized; a two-family
  comparison on the bundled data takes about half the time. Log evidence matches
  0.4.1 to three decimals; Monte Carlo draws differ slightly.
- `GateResult` gains `s_curve`.
- Every source file carries the Apache 2.0 header, checked by a test. Added
  `TRADEMARKS.md`, `GOVERNANCE.md`, `CONTRIBUTING.md` (contributions under Apache 2.0
  with DCO sign-off), `CODE_OF_CONDUCT.md` and `CITATION.cff`.

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
  with an effective-sample-size gate.
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
