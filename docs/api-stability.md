# API stability and versioning

Papilon follows [semantic versioning](https://semver.org) from 1.0.0.

**The public API** is every name in `papilon.__all__`, `papilon.mmm.__all__`,
`papilon.ai.__all__`, `papilon.io.__all__`, `papilon.discovery.__all__`,
`papilon.persist.__all__`, `papilon.report.__all__` and `papilon.tracking.__all__`, and
every documented parameter and return column of those names. Names starting with an
underscore are internal and may change at any time.

| Release | May contain |
|---|---|
| Patch (1.0.x) | Bug fixes. No new parameters, no changed defaults. |
| Minor (1.x.0) | New features and parameters. Defaults change only with a benchmark result in the changelog. Deprecations. |
| Major (x.0.0) | Removal of anything deprecated for at least one minor release. |

**Deprecation.** A deprecated name or parameter keeps working and raises a
`FutureWarning` that names the release that removes it and the replacement. The
changelog lists every deprecation.

**Numbers.** Posterior draws depend on the random seed and on the sampler. A minor
release may move estimates within their reported uncertainty; the changelog says
when a change to inference or defaults moves them by more.

**Saved models.** Files written by `MMM.save` carry a format version. Every 1.x
release reads every earlier 1.x file.

**Enforcement.** `tests/test_public_api.py` compares the public names and parameters
with a recorded snapshot and fails if one is removed.
