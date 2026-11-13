# Papilon documentation

Papilon is a Bayesian marketing mix model you can check. These guides cover the parts
added in 0.5; the [README](../README.md) covers the model itself.

| Guide | What it answers |
|---|---|
| [Quickstart](quickstart.md) | From a spreadsheet of weekly spend and sales to returns and a budget plan, in ten minutes |
| [The knowledge graph](knowledge-graph.md) | How to write down what you believe about the brand, and what the model does with it |
| [Checking the model](checking.md) | Backtest, self-check, lift-test value: which check answers which doubt |
| [Working with AI assistants](ai-assistants.md) | Giving an assistant the model's numbers and caveats, locally |
| [The marketing vocabulary](vocabulary/README.md) | The terms the knowledge graph uses, for sharing graphs between tools |
| [The leaderboard](../benchmarks/LEADERBOARD.md) | How Papilon and a reference baseline score against known truth, and [how to enter](../benchmarks/SUBMITTING.md) |
| [Inference: the gate and NUTS](inference.md) | Which sampler runs, what it reports, and how to read a failed fit |
| [RFC 001](../papilon-amortized-inference-design.md), [RFC 002](design/002-checkable-mmm.md), [RFC 003](design/003-learning-loop-and-geo.md), [RFC 004](design/004-nuts-backend.md) | Design records for the inference method, the checks, the learning loop and the NUTS backend |

Papilon and PyCausalSim are research software from Vector1 Research, free to use under
the Apache License 2.0. If they are useful in your work, please cite them
([CITATION.cff](../CITATION.cff)).
