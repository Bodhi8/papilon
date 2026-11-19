# Working with AI assistants

A language model is good at explaining and bad at arithmetic it has not been given.
Papilon's approach: the model supplies every number and every caution, and the
assistant writes the sentences. Papilon calls a language model only when you ask it to
(see Chat below); otherwise it sends nothing anywhere.

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

## Chat and agent tasks

```python
from papilon.ai import ToolBox
from papilon.chat import Chat

chat = Chat(ToolBox(model))                       # the built-in offline agent
reply = chat.ask("Which channel returns the most?")
reply.text                                        # the answer
reply.steps                                       # every tool call and its result
reply.grounding                                   # each number traced to a tool result

chat.run_task("plan_quarter")                     # a multi-step task
print(chat.transcript())                          # the conversation, tool calls included
```

The offline agent ships with Papilon. It needs no key and no network, and the same
question always gets the same answer. For each question it makes a short plan of tool
calls, runs them (later calls can use earlier results: the quarter plan forecasts under
the spend the optimizer chose), and writes the answer from the results. It understands a
fixed range of questions: returns by channel or for one channel, budget plans at today's
total or a percentage up or down, one channel at other spend levels, which lift test to
run first, forecasts, and the tasks below. Anything else gets the overview and a list of
what it can do.

| Task | Steps |
|---|---|
| `overview` | `brief` |
| `audit` | `brief`, `backtest` |
| `plan_quarter` | `brief`, `optimize_budget`, `plan_stability`, `value_of_lift_test`, `forecast` under the plan |

To let a language model drive the same tools:

```python
Chat(box, llm="anthropic")                        # ANTHROPIC_API_KEY; default model claude-sonnet-5-5
Chat(box, llm="anthropic:claude-haiku-5-5")
Chat(box, llm="openai:<model>")                   # OPENAI_API_KEY
Chat(box, llm="ollama:llama3.1")                  # a local Ollama server, nothing leaves the machine
```

Or pass any object with a `complete(system, history, tools)` method. Requests use the
Python standard library; there is nothing extra to install. When you use a hosted model,
the question and the tool results go to that provider. Tool results are summaries, but
some include figures from your data, such as backtest actuals.

### Grounding

`reply.grounding` lists every number in the answer with the tool it came from. A number
counts as traced when some tool result, tool input or the question rounds to it at the
precision written, as is or times 100 (a share written as a percent). Numbers it cannot
trace, such as a sum the language model worked out itself, are listed in
`grounding["unsupported"]`. Small counts (up to 12), interval levels like "90%" and years
are not checked. The offline agent's answers always pass; its test suite checks that.

### Approval

Tools that change the model are listed in `box.mutating`. The chat stops before running
one and returns a reply whose `pending` field names the tool and its input; call
`chat.approve()` or `chat.decline()` to carry on. Pass `confirm=lambda name, args: ...`
to decide in code instead. `max_steps` (default 8) caps the tool calls behind one answer.

### In a terminal

```bash
python -m papilon.chat --data sales.csv --target sales --date date \
    --media paid_search,paid_social,online_video,display --controls price_index,promo
python -m papilon.chat --task plan_quarter --llm anthropic --data sales.csv ...
```

## What this does not do

It does not stop an assistant from misreading a result. Read the numbers yourself
before acting on a summary.
