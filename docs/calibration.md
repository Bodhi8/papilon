# Calibrating with experiments

Observational data cannot always separate channels that move together or a channel
whose spend follows demand. An experiment can. Papilon takes experiments in two forms.

**Lift tests on the weeks they ran** (recommended):

```python
tests = [{"channel": "paid_search", "start": "2025-03-03", "end": "2025-04-21",
          "factor": 0.0,          # the test switched the channel off (or: spend_change=-42000)
          "lift": -61500.0,       # incremental sales measured over the test plus readout weeks
          "se": 9000.0}]
model = pp.MMM(df, "sales", media, "date", controls=controls, lift_tests=tests)
```

The model predicts the lift of exactly that change, in those weeks, starting from the
spend that actually ran, including carryover into the `readout` weeks after the test
(default `l_max - 1`), and the measurement enters the likelihood with its standard
error. This calibrates the curve at the spend level and season where the test ran,
which is where it was measured.

**Return priors** (`roi_priors={"paid_search": (1.2, 0.25)}`) summarize a test as a
return over the whole window. Use them when only a summary is available.

**Which test to run.** `model.value_of_lift_test()` simulates each possible test,
updates the posterior on each simulated result and re-plans, then reports the expected
gain to the plan. Test the channel at the top. The learning loop does this every cycle.

In a known-truth test (`tests/test_v100.py`), one eight-week holdout of paid search
measured with a 15% standard error narrowed the channel's 90% range by more than 30%.
