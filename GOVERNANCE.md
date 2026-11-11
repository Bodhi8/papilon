# Governance

Papilon is maintained by Brian Curry (Vector1 Research), who holds the copyright and
the trademarks, makes final decisions on scope, design and releases, and is the only
person with release rights on PyPI.

## How decisions are made

- Bugs and small changes: a pull request, reviewed and merged by the maintainer.
- New methods and changes to defaults: an issue or a short RFC in `docs/design/` first,
  with the evidence (for marketing models, results on simulated worlds with known
  answers). Defaults change only with a benchmark result in the changelog.
- Releases follow semantic versioning. Every release is tagged and described in
  `CHANGELOG.md`, which is also the dated public record of when a method first shipped.

## Contributions and licensing

- Contributions are accepted under the project's license, the Apache License 2.0
  (Section 5 of the license: what you submit is licensed on the same terms, including
  its patent grant). There is no separate contributor agreement.
- Sign off each commit (`git commit -s`) to certify, under the Developer Certificate of
  Origin (https://developercertificate.org), that you wrote the change or have the
  right to submit it.
- Do not submit code copied from projects whose license is incompatible with Apache
  2.0, or text copied from other projects' documentation. Say so in the pull request if
  a change is adapted from elsewhere, and keep the original notice.
- New source files carry the license header. A test checks this.

## Names

Use of the project names is covered by [TRADEMARKS.md](TRADEMARKS.md).

## Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
