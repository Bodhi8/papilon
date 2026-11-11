#   Copyright 2024 - 2026 Brian Curry / Vector1 Research
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Serve a fitted Papilon model over the Model Context Protocol.

    python -m papilon.mcp_server --data sales.csv --target sales --date date \\
        --media paid_search,paid_social,online_video,display --controls price_index,promo \\
        [--delayed online_video] [--graph brand_graph.json]

The model is fitted once at start-up; every tool then answers from that fit
through :class:`papilon.ai.ToolBox`, so an assistant connected to this server
reads the same numbers and the same caveats as the Python API. The data never
leaves the machine the server runs on. Needs the optional ``mcp`` package
(``pip install "papilon[mcp]"``); both the 1.x and 2.x SDKs are supported.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from .ai import SYSTEM_PROMPT, ToolBox

__all__ = ["build_server", "main"]


def build_server(box: ToolBox, name: str = "papilon"):
    """An MCP server exposing ``box``'s tools. Returns the server object; call ``.run()``."""
    try:
        from mcp.server.mcpserver import MCPServer as Server      # mcp >= 2
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as Server      # mcp 1.x
        except ImportError as e:  # pragma: no cover
            raise ImportError('the MCP server needs the "mcp" package: pip install "papilon[mcp]"') from e
    server = Server(name, instructions=SYSTEM_PROMPT)
    desc = {t["name"]: t["description"] for t in box.specs("mcp")}

    def brief() -> str:
        return box.call_json("brief")

    def get_roi() -> str:
        return box.call_json("get_roi")

    def response_curve(channel: str, spend_levels: list[float]) -> str:
        return box.call_json("response_curve", {"channel": channel, "spend_levels": spend_levels})

    def optimize_budget(total: float | None = None, max_change: float | None = 0.5,
                        objective: str = "quantile") -> str:
        return box.call_json("optimize_budget", {"total": total, "max_change": max_change, "objective": objective})

    def evaluate_plan(spend: dict[str, float]) -> str:
        return box.call_json("evaluate_plan", {"spend": spend})

    def value_of_lift_test(sd: float = 0.25) -> str:
        return box.call_json("value_of_lift_test", {"sd": sd})

    def backtest(holdout: int = 13) -> str:
        return box.call_json("backtest", {"holdout": holdout})

    def what_moves(node: str) -> str:
        return box.call_json("what_moves", {"node": node})

    for fn in (brief, get_roi, response_curve, optimize_budget, evaluate_plan, value_of_lift_test, backtest, what_moves):
        if fn.__name__ in desc:
            server.add_tool(fn, name=fn.__name__, description=desc[fn.__name__])
    return server


def main(argv: list[str] | None = None) -> None:
    from .graph import MarketingGraph
    from .mmm import MMM

    ap = argparse.ArgumentParser(prog="python -m papilon.mcp_server", description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="CSV with one row per period")
    ap.add_argument("--target", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--media", required=True, help="comma-separated spend columns")
    ap.add_argument("--controls", default="")
    ap.add_argument("--delayed", default="")
    ap.add_argument("--graph", default=None, help="JSON from MarketingGraph.to_dict()")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    a = ap.parse_args(argv)
    split = lambda s: [x for x in s.split(",") if x]
    df = pd.read_csv(a.data, parse_dates=[a.date])
    model = MMM(df, a.target, split(a.media), a.date, controls=split(a.controls),
                delayed=split(a.delayed), seed=a.seed).fit()
    graph = MarketingGraph.from_dict(json.load(open(a.graph))) if a.graph else None
    build_server(ToolBox(model, graph=graph)).run(a.transport)


if __name__ == "__main__":  # pragma: no cover
    main()
