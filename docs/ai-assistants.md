# Working with AI assistants

A language model is good at explaining and bad at arithmetic it has not been given.
Papilon's approach: the model supplies every number and every caution, and the
assistant writes the sentences. Papilon itself calls no language model and sends no
data anywhere.

## Facts and cautions as data

```python
from papilon.ai import brief

b = brief(model, plan=plan, graph=g)
b["channels"]["paid_search"]     # roas, range, marginal roas, whether a lift test calibrated it
b["caveats"]                     # plain-language warnings
```

The caveats come from fixed rules, so the same model always produces the same
warnings: a fit that failed its gate, unreliable importance weights, a range wider than
the estimate, top channels whose ranges overlap, a plan that leaves the range of past
spend or is not clearly better than today's mix, and high or medium findings from the
graph audit.

## The model as tools

```python
from papilon.ai import ToolBox, SYSTEM_PROMPT

box = ToolBox(model, graph=g)
box.specs("tool_use")            # or "function", or "mcp": tool definitions with JSON schemas
box.call("optimize_budget", {"max_change": 0.25})
```

Pass `box.specs(...)` as the tool list of whichever chat model API you use, send
`SYSTEM_PROMPT` as the system message, and answer each tool request with
`box.call_json(name, arguments)`. Every result carries a `caveats` list. Errors come
back as `{"error": ...}` so the assistant can correct its call.

Tools: `brief`, `get_roi`, `response_curve`, `optimize_budget`, `evaluate_plan`,
`value_of_lift_test`, `backtest`, and `what_moves` when a graph is attached.

## Over the Model Context Protocol

```bash
pip install "papilon[mcp]"
python -m papilon.mcp_server --data sales.csv --target sales --date date \
    --media paid_search,paid_social,online_video,display --controls price_index,promo
```

Most assistants that support the protocol take a configuration like this:

```json
{
  "mcpServers": {
    "papilon": {
      "command": "python",
      "args": ["-m", "papilon.mcp_server", "--data", "/path/to/sales.csv", "--target", "sales",
               "--date", "date", "--media", "paid_search,paid_social,online_video,display"]
    }
  }
}
```

The model is fitted once when the server starts, on your machine. The assistant
receives tool results, not your data file. Add `--graph brand_graph.json`
(from `json.dump(g.to_dict(), ...)`) to give it the knowledge graph.

## What this does not do

It does not stop an assistant from misreading a result. Read the numbers yourself
before acting on a summary.
