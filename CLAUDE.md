# Working rules for AI assistants in this repository

- **Nothing that costs money without the owner's approval first.** This covers paid API
  calls, paid compute or CI minutes, subscriptions and any other spend. Free actions need
  no approval.
- **No competitor comparisons.** Do not add, run or publish scores for other
  organizations' libraries or AI models. Benchmarks here grade this project's own code.
- **Correctness is tested against known truth.** A change to an estimator comes with a
  test on simulated data whose true answer is known.
- **Every source file carries the Apache-2.0 header**; `tests/test_license_headers.py`
  checks it.
- **Report changes and pushes** to the owner, and work on a branch with a pull request,
  not on `main`.
- **Put each task on the right model.** The lead session (the strongest model) designs
  APIs and estimators, works out the math and priors, designs every known-truth test,
  makes release decisions and reviews all delegated work. Sonnet subagents may implement
  a change the lead has specified, write tests to the lead's spec, and fix docs, CI and
  packaging. Haiku subagents handle lookups and log reading. Skip delegation for small
  tasks. A delegated change is done only when the known-truth tests and the full suite
  pass and the lead has read the diff; anything that fails twice goes back to the lead.
