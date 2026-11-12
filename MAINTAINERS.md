# Maintainers and continuity

## Who maintains Papilon

| Role | Person | Contact |
|---|---|---|
| Lead maintainer, copyright and trademark holder | Brian Curry, Vector1 Research | brian@vector1.ai |

Papilon is maintained by one person today. Teams evaluating it for production
should weigh that, and this page says plainly what is promised and what protects you
if the maintainer is unavailable.

## What is promised

- **Stable API from 1.0.** Public names and parameters are not removed or changed in
  meaning within a major version. Deprecations warn for at least one minor release
  first. See [docs/api-stability.md](docs/api-stability.md).
- **Security fixes** for the latest minor release, per [SECURITY.md](SECURITY.md).
- **Bug reports** acknowledged within five business days.
- **Every release** is tested on Linux, Windows and macOS for Python 3.10 to 3.13,
  tagged, and described in [CHANGELOG.md](CHANGELOG.md).

## What protects users regardless

- **Apache License 2.0.** Anyone may fork, maintain and ship Papilon, including a
  company for its own use, with a patent grant from every contributor.
- **No hidden parts.** The model, its checks, the benchmark and its results are all
  in this repository. Nothing depends on a service run by the maintainer.
- **Readable saved models.** `MMM.save` writes plain JSON, CSV and NumPy arrays in a
  zip, so fitted models remain readable without Papilon.
- **Plain dependencies.** NumPy, SciPy, pandas, networkx and matplotlib; optional
  extras only for warehouses, MLflow and the Model Context Protocol.

## Succession

If the lead maintainer cannot continue, the repository and the PyPI project will be
transferred to a co-maintainer named in this file, or, if none is named, to a
maintainer chosen by the most active contributors of the preceding year, announced
in the repository. Co-maintainers are welcome: contributors with a sustained record
of reviewed contributions may be invited to join this table.

## Commercial support

Organizations that need guaranteed response times, help with a deployment, or a
review of their own model can contact brian@vector1.ai.
