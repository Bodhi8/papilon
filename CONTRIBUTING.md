# Contributing to Papilon

Bug reports, questions and pull requests are welcome at
https://github.com/Bodhi8/papilon/issues.

## Code

1. Fork, branch from `main`, and install with `pip install -e ".[dev]"`.
2. Add or update tests in `tests/`. Run `pytest -m "not slow and not jax"` (a few minutes; needs no
   optional dependency). If you touch the model or the samplers, also run `pytest -m jax` after
   `pip install -e ".[dev,jax]"`.
3. A change to a model default needs a result from `benchmarks/recovery.py` in the
   pull request: defaults are set by how plans and estimates score against known truth.
4. Sign off your commits (`git commit -s`).

## Licensing of contributions

Papilon is licensed under the Apache License 2.0. By submitting a contribution you
license it on the same terms (Section 5 of the license) and certify, under the
Developer Certificate of Origin, that you have the right to submit it. Do not include
code or text copied from other projects unless its license allows it and you keep its
notice. See [GOVERNANCE.md](GOVERNANCE.md) and [TRADEMARKS.md](TRADEMARKS.md).
