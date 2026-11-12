# Running in production

**Check the data first.**

```python
from papilon.io import validate_frame
validate_frame(df, "sales", media, "date", controls)   # errors first: missing columns, negatives, duplicates, gaps
```

**Load from a warehouse.** `papilon.io.read_sql(query, engine)` works with any SQLAlchemy
engine or DB-API connection; `from_snowflake` and `from_bigquery` wrap the official
clients (install `papilon[snowflake]` or `papilon[bigquery]`). `long_to_wide` pivots spend
stored one row per date and channel.

**Save and load.** `model.save("brand_2026q4.papilon")` writes one zip of JSON, CSV and
NumPy arrays: no pickles, so loading never runs code from the file, and the parts open
without Papilon. `pp.MMM.load(path)` restores the fitted model without refitting.

**Track refreshes.** `papilon.tracking.log_to_mlflow(model, experiment="mmm")` logs settings,
fit quality and every channel's return with its range, plus the saved model and returns
table as artifacts (install `papilon[mlflow]`).

**Monitor.** When new periods arrive, `model.monitor(new_data)` compares them with the
forecast: coverage of the 90% range, mean standardized error, spend outside anything
the model has seen, and `refit_recommended`.

**Report.** `papilon.report.report(model, "report.html", graph=g)` writes one HTML file
with charts, returns, the plan and its stability, the graph audit and every caution,
for people who will never run Python.

**Inference you can rely on.** Each fit reports a verdict. `ACCEPT` means importance
sampling passed its effective-sample-size gate; `SMC` means it did not and the tempered
sequential Monte Carlo fallback produced the draws (`escalate="smc"`, the default). Both
are usable; `GateResult.passed` says so. Multimodal posteriors get a mixture proposal
built from every distinct mode found.

**Versions and support.** See [API stability](api-stability.md) and
[MAINTAINERS.md](https://github.com/Bodhi8/papilon/blob/main/MAINTAINERS.md).
