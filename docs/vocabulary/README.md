# The Vector1 marketing vocabulary

Version 0.1, November 2026. Namespace: `https://vector1.ai/ns/marketing#`

A small vocabulary for describing how a brand's marketing works, so that a description
written once can configure a model, generate simulated brands, and be read by other
tools. `papilon.MarketingGraph.to_jsonld()` writes it; the context file is
[marketing.jsonld](marketing.jsonld).

## Classes

| Term | Meaning |
|---|---|
| `Outcome` | The quantity being explained, usually sales or revenue per period |
| `Channel` | A paid media channel with spend per period |
| `Driver` | Anything else that moves the outcome or a channel's spend: seasonality, price, a launch, a competitor |
| `Group` | A set of channels that belong together, such as all search |
| `Experiment` | A lift test or geo experiment that measured a channel's return |

## Relations

| Term | From, to | Meaning |
|---|---|---|
| `drives` | Channel or Driver, Outcome | Changes in the source cause changes in the outcome |
| `sets_spend` | Driver, Channel | The brand spends more on the channel when the driver is high |
| `mediates` | Channel, Channel | Part of the first channel's effect on the outcome arrives through the second |
| `belongs_to` | Channel, Group | Membership; not causal |
| `measured_by` | Channel, Experiment | The experiment measured this channel; not causal |

A `Driver` with `sets_spend` to a channel and `drives` to the outcome is a confounder
of that channel.

## Properties

| Term | On | Meaning |
|---|---|---|
| `name` | any | Identifier, unique within a graph |
| `stage` | Channel | Place in the funnel: `upper`, `mid` or `lower` |
| `column` | Channel, Driver | Name of the data column that measures it; absent for an unobserved driver |
| `modeled` | Driver | True when the model accounts for it without a column (yearly seasonality, trend) |
| `kind` | Driver | `driver`, or `event` for something that starts and stops |
| `roas`, `sd` | Experiment | Measured return on spend and its standard error |
| `status` | any relation | `confirmed`, or `proposed` for an edge awaiting review. Proposed edges must not be used for modeling. |

## Versioning

Terms are added in minor versions and never change meaning. A term that needs a
different meaning gets a new name. The vocabulary is published under the Apache
License 2.0; "Vector1" is a trademark of Brian Curry (see
[TRADEMARKS.md](../../TRADEMARKS.md)).
