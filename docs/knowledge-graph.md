# The knowledge graph

A marketing mix model depends on things the data cannot tell it: which channels feed
which, what else moves sales, and which of those drivers you can measure.
`MarketingGraph` is where you write that down. The model is then built from it, and the
graph tells you which estimates rest on something you have no data for.

## Building one

```python
import papilon as pp

g = pp.MarketingGraph("sales")                                   # the outcome column

g.channel("paid_search", stage="lower", group="search")          # funnel stage: upper, mid or lower
g.channel("paid_social", stage="mid")
g.channel("online_video", stage="upper")

g.driver("seasonality", ["paid_search", "sales"], modeled=True)  # the model handles it; no column needed
g.driver("price", ["sales"], column="price_index")               # measured: becomes a control
g.event("launch", ["paid_search", "paid_social", "sales"])       # no column: unobserved

g.mediates("online_video", "paid_search")                        # video lifts branded search
g.experiment("geo_q3", channel="paid_search", roas=1.2, sd=0.25) # a lift test you ran
```

A driver that sets a channel's spend and also drives sales is a confounder of that
channel: it makes the channel look better or worse than it is.

## What it does

| Call | Result |
|---|---|
| `g.audit(df)` | Findings, most serious first. See below. |
| `g.build_model(df, date_col="date")` | A `papilon.MMM` with media, controls and lift-test priors taken from the graph |
| `g.carryover_candidates()` | Families for `compare_carryover`: upper-funnel channels may peak after they run |
| `g.what_moves("online_video")` | Everything downstream and the chain of relations that connects it |
| `g.describe()` | The graph in sentences, for a person or a language model |
| `g.to_dict()`, `g.to_jsonld()` | Plain data, and JSON-LD in the [marketing vocabulary](vocabulary/README.md) |

## Reading the audit

| Finding | Meaning | What to do |
|---|---|---|
| `unobserved_confounder` | A driver moves this channel's spend and sales together and has no data. The channel's return cannot be learned from this dataset. | Run a lift test and add it with `experiment()`, or add a column for the driver. Severity drops to low once a test covers the channel. |
| `mediated` | Part of one channel's effect arrives through another, and the model credits that part to the second. | Read the first channel as understated. Cutting it also lowers what the second can earn. |
| `missing_column` | A declared column is not in the data. | Fix the name or add the data. |
| `weak_link` | A declared relation shows no linear association in the data. | A screen only. Reconsider the edge, or keep it if you have other reasons. |
| `proposed` | An edge awaiting your confirmation. | `confirm()` it or leave it; proposed edges are never used. |

## Edges suggested by a language model

```python
g.propose("paid_social", "mediates", "paid_search", by="assistant", note="social prompts branded searches")
g.confirm("paid_social", "mediates", "paid_search")     # only after you agree
```

A proposed edge appears in `describe()` and `audit()` marked as unconfirmed, and is
ignored everywhere else until confirmed.

## Simulating brands with the same structure

```python
from pycausalsim.futures import worlds_from_graph, leaderboard, papilon_estimator

worlds = worlds_from_graph(g, 40, seed=1)                   # every driver with a column is visible
blind = worlds_from_graph(g, 40, seed=1, hide=["price"])    # the same brands, price withheld
```

Each world has an answer key, so you can measure what not observing a driver costs.

## A caution

A graph makes assumptions explicit and checkable. It does not make them true. A wrong
edge produces a confidently wrong model, and the data can test only some edges.
