# RFC 002: A marketing mix model that checks itself

Author: Brian Curry, Vector1 Research
Status: implemented in Papilon 0.5.0 and PyCausalSim 0.5.0
Date: 2026-11-11

This note records the design of the methods added in 0.5.0, so that what each one
computes, and when it was first described, is on the record alongside the code.

## 1. Self-check on simulated copies of the brand

`MMM.self_check()` asks whether this model, with this data, could recover returns like
the ones it reports.

1. Take K joint posterior draws of the media parameters as K ground truths.
2. For each, compute the conditional posterior mean of the baseline coefficients and
   the noise scale in closed form (the baseline is conjugate), and generate sales from
   the brand's own spend, controls and calendar with fresh noise.
3. Refit the same settings to each simulated brand.
4. Score against the answer key: coverage of the 90% ROAS interval, bias, and the true
   gain of the default budget plan as a share of the best possible gain.

Optionally, add demand proportional to one channel's standardized spend that the
channel did not cause. The resulting bias is the size of error an unmeasured driver of
that strength would produce in this dataset.

The check holds the model's functional form fixed. It detects weak identification
(spend that moves with season, channels that move together) and priors that dominate
the data. It does not detect a wrong functional form; `compare_models()` and
PyCausalSim's S-curve worlds address that.

## 2. The value of a lift test, by posterior reweighting

`MMM.value_of_lift_test()` estimates, for each channel, the expected improvement in
the budget plan from running a lift test on it, without refitting.

For a test on channel c with standard error s, repeat for simulated outcomes m = 1..M:

1. Draw a posterior draw i as the truth and a measured return r = ROAS_i[c] + N(0, s).
2. Reweight every posterior draw k by the likelihood of that measurement,
   w_k ∝ exp(-(ROAS_k[c] - r)² / 2s²). The weighted draws are the posterior after the
   test.
3. Re-plan under the weighted posterior. Both plans maximize expected sales within the
   same limits, so the value of information is non-negative in expectation.
4. Score the new plan and today's plan under truth i.

The mean difference in step 4 is the expected value of the test, in sales per period.
Common random numbers (the same i and noise across channels) make the ranking stable
at small M. The weighted interval width gives the expected precision after the test.

## 3. The marketing knowledge graph

`MarketingGraph` is a small ontology: node types Outcome, Channel, Driver, Group and
Experiment; relations `drives`, `sets_spend`, `mediates`, `belongs_to`, `measured_by`;
every edge is `confirmed` or `proposed`.

Compile rules:

- media = channels; controls = columns of observed drivers of the outcome;
- lift tests on a channel pool by inverse variance into its ROAS prior;
- upper-funnel channels form the delayed-carryover candidate family.

Audit rules:

- a driver with `sets_spend` to a channel and `drives` to the outcome, with no column
  and not handled by the model, makes that channel's return unidentified from the
  data. Severity is high unless a lift test covers the channel;
- `A mediates B` means the model credits the mediated part of A's effect to B;
- observed driver relations with no linear association in the data are flagged as
  weak, as a screen only;
- proposed edges are reported and never used.

The same graph, exported as plain data, is the input to
`pycausalsim.futures.worlds_from_graph`, which turns each relation into a structural
equation (a `sets_spend` edge scales the channel's spend with the driver, `drives`
adds to baseline demand, `mediates` scales the downstream channel's spend with the
upstream channel's adstock) and returns worlds with an answer key, including each
channel's total return with mediated paths.

## 4. Saturation shape by Bayes factor

The saturation curve of a channel in `s_curve` is Hill with shape s,
x^s / (x^s + K^s), with log s ~ Normal(log 1.5, 0.4). The marginal likelihood from
the importance weights compares families with and without the shape parameter, with
the same gate as the carryover comparison.

## 5. Grounded tools for language models

`papilon.ai.ToolBox` exposes the fitted model as tools. Each result is computed by the
model and returned with its caveats, which are produced by fixed rules (gate verdict,
Pareto k above 0.7, interval wider than the estimate, overlapping top ranks, plan
outside sustained spend, probability of beating the current mix below 0.8, graph audit
findings of high or medium severity). The language model writes sentences; it does not
produce numbers.
