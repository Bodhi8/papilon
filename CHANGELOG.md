# Changelog

## 1.0.0 (unreleased)

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
