#   Copyright 2026 Brian Curry / Vector1 Research
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
"""Chat with a fitted model, and let an agent work through multi-step tasks on it.

``Chat(box)`` runs a tool-use loop over a :class:`ToolBox`: the language model
asks for tools, the toolbox computes, the model writes the answer from the
results. Every reply comes back with the steps it took and a grounding check
that traces each number in the answer to a tool result.

Language models are pluggable and optional:

``llm=None`` or ``"offline"``
    A deterministic agent that ships with the library. It picks tools from the
    question, chains several when a task needs it, and writes the answer from
    the results. No network, no key, no cost; the same question always gets the
    same answer. It understands a fixed range of questions (see ``TASKS``).
``"anthropic"`` or ``AnthropicLLM(...)``
    Claude, through the Messages API (``ANTHROPIC_API_KEY``).
``"openai:<model>"``, ``"ollama:<model>"`` or ``OpenAILLM(...)``
    Any chat completions API with function calling, local servers included.

Network calls use the standard library only. Actions that change the model
(tools listed in ``box.mutating``) wait for approval: pass ``confirm=`` to
decide in code, or leave it out and answer the reply's ``pending`` request
with :meth:`Chat.approve` or :meth:`Chat.decline`.

From a terminal, ``python -m <package>.chat --help`` starts a conversation with
a model built from a CSV file.

This module is shared, line for line, by Papilon and PyCausalSim.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from ._planner import TASKS, cli_toolbox, make_plan

__all__ = ["Chat", "Reply", "OfflineAgent", "AnthropicLLM", "OpenAILLM", "make_llm", "check_grounding", "TASKS"]


# --------------------------------------------------------------------- replies

@dataclass
class Reply:
    """One answer: the text, the tool steps behind it and the grounding check.

    ``pending`` is set when the agent stopped to ask for approval of an action
    that changes the model; nothing else runs until :meth:`Chat.approve` or
    :meth:`Chat.decline` is called.
    """
    text: str
    steps: list = field(default_factory=list)
    grounding: dict = field(default_factory=dict)
    backend: str = "offline"
    pending: dict | None = None

    def to_dict(self) -> dict:
        return {"text": self.text, "steps": self.steps, "grounding": self.grounding,
                "backend": self.backend, "pending": self.pending}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def __str__(self) -> str:
        return self.text


# ------------------------------------------------------------------- grounding

_NUM = re.compile(r"(?<![\w.\-/:])[-−+]?\$?\d[\d,]*(?:\.\d+)?%?")


def _numbers_in(text: str):
    """(shown text, value, decimals, is_percent) for each number written in ``text``."""
    out = []
    for m in _NUM.finditer(text or ""):
        s = m.group(0)
        raw = s.replace("−", "-").replace("$", "").replace(",", "")
        pct = raw.endswith("%")
        raw = raw.rstrip("%")
        if raw in ("", "-", "+"):
            continue
        try:
            v = float(raw)
        except ValueError:
            continue
        dec = len(raw.split(".")[1]) if "." in raw else 0
        out.append((s.rstrip(".,"), v, dec, pct))
    return out


def _flatten(x, label, out):
    if isinstance(x, bool) or x is None:
        return
    if isinstance(x, (int, float)):
        out.append((float(x), label))
    elif isinstance(x, str):
        out.extend((v, label) for _, v, _, _ in _numbers_in(x))
    elif isinstance(x, dict):
        for k, v in x.items():
            _flatten(str(k), label, out)
            _flatten(v, label, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _flatten(v, label, out)


def _exempt(v: float, dec: int, pct: bool) -> bool:
    """Numbers that are labels, not findings: small counts, interval levels, years."""
    if dec:
        return False
    a = abs(v)
    return a <= 12 or (pct and a in (50, 80, 90, 95, 100)) or (1900 <= a <= 2100 and not pct)


def check_grounding(text: str, sources) -> dict:
    """Trace every number in ``text`` to a number in ``sources``.

    ``sources`` is a list of ``(label, value)`` pairs, where ``value`` is any
    JSON-like structure (tool results, tool inputs, the question). A number is
    supported when some source value rounds to it at the precision shown, as is
    or times 100 (a share written as a percent), ignoring sign (a fall written
    as a positive amount). Integers up to 12, interval levels such as "90%" and
    years are not checked.

    Returns ``{"ok", "checked", "supported", "unsupported"}``; each item gives
    the number as written and, when supported, the label of its source.
    """
    pool = []
    for label, value in sources:
        _flatten(value, label, pool)
    supported, unsupported = [], []
    for shown, x, dec, pct in _numbers_in(text):
        if _exempt(x, dec, pct):
            continue
        tol = 0.5 * 10 ** (-dec) + 1e-9 * max(1.0, abs(x))
        hit = None
        for v, label in pool:
            for cand in (v, v * 100.0):
                if abs(abs(cand) - abs(x)) <= tol:
                    hit = label
                    break
            if hit:
                break
        (supported if hit else unsupported).append({"text": shown, "value": x, "source": hit})
    return {"ok": not unsupported, "checked": len(supported) + len(unsupported),
            "supported": supported, "unsupported": unsupported}


# ------------------------------------------------------------------- backends

def _post(url: str, body: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:  # pragma: no cover - needs a live API
        raise RuntimeError(f"{url} answered {e.code}: {e.read().decode(errors='replace')[:500]}") from e


def _result_text(ev) -> str:
    return json.dumps(ev["result"])


class AnthropicLLM:
    """Claude through the Messages API. The key comes from ``api_key`` or ``ANTHROPIC_API_KEY``."""
    name = "anthropic"
    tool_format = "tool_use"

    def __init__(self, model: str = "claude-sonnet-5-5", api_key: str | None = None, max_tokens: int = 1500,
                 base_url: str = "https://api.anthropic.com", timeout: float = 120.0) -> None:
        self.model, self.max_tokens, self.base_url, self.timeout = model, max_tokens, base_url.rstrip("/"), timeout
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("no API key: pass api_key= or set ANTHROPIC_API_KEY")

    @staticmethod
    def messages(history: list) -> list:
        out = []
        for ev in history:
            if ev["role"] == "user":
                out.append({"role": "user", "content": ev["text"]})
            elif ev["role"] == "assistant":
                content = [{"type": "text", "text": ev["text"]}] if ev.get("text") else []
                content += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["input"]}
                            for c in ev.get("tool_calls", [])]
                out.append({"role": "assistant", "content": content})
            else:
                block = {"type": "tool_result", "tool_use_id": ev["id"], "content": _result_text(ev)}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    def complete(self, system: str, history: list, tools: list) -> dict:
        data = _post(f"{self.base_url}/v1/messages",
                     {"model": self.model, "max_tokens": self.max_tokens, "system": system,
                      "tools": tools, "messages": self.messages(history)},
                     {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, self.timeout)
        content = data.get("content", [])
        return {"text": "".join(b.get("text", "") for b in content if b.get("type") == "text"),
                "tool_calls": [{"id": b["id"], "name": b["name"], "input": b.get("input") or {}}
                               for b in content if b.get("type") == "tool_use"]}


class OpenAILLM:
    """Any chat completions API with function calling: OpenAI, Azure, vLLM, Ollama, LM Studio.

    ``model`` is required. The key comes from ``api_key`` or ``OPENAI_API_KEY``;
    local servers usually need none.
    """
    name = "openai"
    tool_format = "function"

    def __init__(self, model: str, api_key: str | None = None, base_url: str = "https://api.openai.com/v1",
                 timeout: float = 120.0) -> None:
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")

    @staticmethod
    def messages(system: str, history: list) -> list:
        out = [{"role": "system", "content": system}]
        for ev in history:
            if ev["role"] == "user":
                out.append({"role": "user", "content": ev["text"]})
            elif ev["role"] == "assistant":
                m = {"role": "assistant", "content": ev.get("text") or None}
                if ev.get("tool_calls"):
                    m["tool_calls"] = [{"id": c["id"], "type": "function",
                                        "function": {"name": c["name"], "arguments": json.dumps(c["input"])}}
                                       for c in ev["tool_calls"]]
                out.append(m)
            else:
                out.append({"role": "tool", "tool_call_id": ev["id"], "content": _result_text(ev)})
        return out

    def complete(self, system: str, history: list, tools: list) -> dict:
        headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = _post(f"{self.base_url}/chat/completions",
                     {"model": self.model, "messages": self.messages(system, history), "tools": tools},
                     headers, self.timeout)
        msg = data["choices"][0]["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": c["id"], "name": c["function"]["name"], "input": args})
        return {"text": msg.get("content") or "", "tool_calls": calls}


class OfflineAgent:
    """The built-in agent: no language model, no network.

    For each question it makes a plan, a short list of tool calls in which later
    calls may use earlier results, runs the plan one call per turn of the loop,
    and writes the answer from the results. Questions outside its range get an
    overview and a list of what it can do.
    """
    name = "offline"
    tool_format = "tool_use"

    def __init__(self, box) -> None:
        self.box = box

    def complete(self, system: str, history: list, tools: list) -> dict:
        start = max(i for i, ev in enumerate(history) if ev["role"] == "user")
        question = history[start]["text"]
        done = [{"name": ev["name"], "input": ev["input"], "result": ev["result"]}
                for ev in history[start + 1:] if ev["role"] == "tool"]
        plan = make_plan(question, self.box)
        if len(done) < len(plan.steps):
            name, args = plan.steps[len(done)]
            args = args(done) if callable(args) else dict(args)
            return {"text": "", "tool_calls": [{"id": f"offline-{start}-{len(done)}", "name": name, "input": args}]}
        return {"text": plan.write(done), "tool_calls": []}


def make_llm(spec, box=None):
    """A backend from a short spec: "offline", "anthropic", "anthropic:<model>",
    "openai:<model>" or "ollama:<model>" (a local Ollama server)."""
    if spec is None or spec == "offline":
        return OfflineAgent(box)
    if not isinstance(spec, str):
        return spec
    kind, _, model = spec.partition(":")
    if kind == "anthropic":
        return AnthropicLLM(model) if model else AnthropicLLM()
    if kind == "openai":
        return OpenAILLM(model)
    if kind == "ollama":
        return OpenAILLM(model, api_key="", base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
    raise ValueError(f"unknown language model spec {spec!r}")


# ------------------------------------------------------------------------ chat

_DECLINED = "The user declined this action. Do not retry it; answer without it or suggest an alternative."

class Chat:
    """A conversation with a fitted model.

    >>> chat = Chat(ToolBox(model))                 # offline agent, no key needed
    >>> reply = chat.ask("Which channel returns the most?")
    >>> reply.text, reply.steps, reply.grounding["ok"]
    >>> chat.run_task("plan_quarter")              # a multi-step agent task
    >>> Chat(box, llm="anthropic").ask("...")       # the same tools, driven by Claude

    ``max_steps`` caps the tool calls behind one answer. ``confirm(name, input)``
    decides on actions that change the model; without it the reply's
    ``pending`` field asks the caller.
    """

    def __init__(self, box, llm=None, system: str | None = None, max_steps: int = 8, confirm=None) -> None:
        self.box = box
        self.llm = make_llm(llm, box)
        self.system = system or box.system_prompt
        self.max_steps = int(max_steps)
        self.confirm = confirm
        self.history: list = []
        self._pending: list | None = None
        self._queue: list = []
        self._start = 0

    # public -------------------------------------------------------------
    def ask(self, question: str) -> Reply:
        """Answer one question, calling as many tools as it needs."""
        self.start(question)
        return self._run()

    def start(self, question: str) -> None:
        """Begin a question without running anything; drive it with :meth:`step`.

        For interfaces that show each tool call as it happens::

            chat.start("Plan next quarter")
            while not isinstance(out := chat.step(), Reply):
                show(out)            # {"type": "plan", "calls": [...]} or {"type": "tool", ...}
        """
        if self._pending:
            raise RuntimeError("an action is waiting for approval: call approve() or decline() first")
        self.history.append({"role": "user", "text": str(question)})
        self._start = len(self.history) - 1
        self._queue = []

    def step(self):
        """Advance by one unit of work.

        Returns ``{"type": "plan", "calls": [...]}`` when the model chose its
        next tool calls, ``{"type": "tool", "tool", "input", "result"}`` after
        running one, or the final :class:`Reply` (which may be a request for
        approval, with ``pending`` set).
        """
        if self._pending:
            return self._pending_reply()
        if self._queue:
            c = self._queue.pop(0)
            mutating = getattr(self.box, "mutating", frozenset())
            if c["name"] in mutating and not c.get("approved"):
                if self.confirm is None:
                    self._pending = [c] + self._queue
                    self._queue = []
                    return self._pending_reply()
                if not self.confirm(c["name"], c["input"]):
                    self._record(c, {"error": _DECLINED})
                    return {"type": "tool", "tool": c["name"], "input": c["input"], "result": self.history[-1]["result"]}
            self._record(c, self.box.call(c["name"], c["input"]))
            return {"type": "tool", "tool": c["name"], "input": c["input"], "result": self.history[-1]["result"]}
        n_calls = sum(1 for ev in self.history[self._start:] if ev["role"] == "tool")
        fmt = getattr(self.llm, "tool_format", "tool_use")
        out = self.llm.complete(self.system, self.history, self.box.specs(fmt))
        calls = out.get("tool_calls") or []
        if calls and n_calls + len(calls) > self.max_steps:
            calls = []
            out["text"] = ((out.get("text") or "") + "\n\nI stopped after "
                           f"{self.max_steps} tool calls without finishing. Ask a narrower question.").strip()
        self.history.append({"role": "assistant", "text": out.get("text") or "", "tool_calls": calls})
        if not calls:
            return self._finish(out.get("text") or "")
        self._queue = list(calls)
        return {"type": "plan", "calls": [{"tool": c["name"], "input": c["input"]} for c in calls],
                "text": out.get("text") or ""}

    def run_task(self, name: str) -> Reply:
        """Run one of the agent tasks in ``TASKS`` by name."""
        if name not in TASKS:
            raise KeyError(f"unknown task {name!r}; tasks are {sorted(TASKS)}")
        return self.ask(TASKS[name])

    def approve(self, run: bool = True):
        """Run the action the last reply asked about, then carry on to the answer.
        With ``run=False`` nothing runs yet; continue with :meth:`step`."""
        return self._resolve(True, run)

    def decline(self, run: bool = True):
        """Refuse the action the last reply asked about, then carry on without it.
        With ``run=False`` nothing runs yet; continue with :meth:`step`."""
        return self._resolve(False, run)

    def reset(self) -> None:
        self.history, self._pending, self._queue = [], None, []

    def transcript(self) -> str:
        """The conversation as Markdown, tool calls included: a record of how each answer was reached."""
        lines = []
        for ev in self.history:
            if ev["role"] == "user":
                lines += [f"**You:** {ev['text']}", ""]
            elif ev["role"] == "assistant":
                for c in ev.get("tool_calls", []):
                    lines.append(f"- tool `{c['name']}` {json.dumps(c['input'])}")
                if ev.get("text"):
                    lines += ["", f"**Assistant:** {ev['text']}", ""]
        return "\n".join(lines).strip() + "\n"

    # loop ---------------------------------------------------------------
    def _steps(self) -> list:
        return [{"tool": ev["name"], "input": ev["input"], "result": ev["result"]}
                for ev in self.history[self._start:] if ev["role"] == "tool"]

    def _record(self, call, result):
        self.history.append({"role": "tool", "id": call["id"], "name": call["name"],
                             "input": call["input"], "result": result})

    def _resolve(self, approved: bool, run: bool = True):
        if not self._pending:
            raise RuntimeError("nothing is waiting for approval")
        calls, self._pending = self._pending, None
        first = dict(calls[0])
        if approved:
            first["approved"] = True
            self._queue = [first] + calls[1:]
        else:
            self._record(first, {"error": _DECLINED})
            self._queue = calls[1:]
        return self._run() if run else None

    def _pending_reply(self) -> Reply:
        c = self._pending[0]
        return Reply(text=f"I want to run `{c['name']}`, which changes the model. Approve or decline?",
                     steps=self._steps(), backend=self.llm.name,
                     pending={"tool": c["name"], "input": c["input"]})

    def _run(self) -> Reply:
        while True:
            out = self.step()
            if isinstance(out, Reply):
                return out

    def _finish(self, text: str) -> Reply:
        steps = self._steps()
        sources = [("question", self.history[self._start]["text"])]
        for s in steps:
            sources += [(s["tool"], s["result"]), (s["tool"], s["input"])]
        return Reply(text=text, steps=steps, grounding=check_grounding(text, sources), backend=self.llm.name)


# ------------------------------------------------------------------------- cli

def main(argv: list | None = None) -> None:
    """Chat in a terminal. Options other than --llm and --task build the model (see --help)."""
    import argparse
    import sys
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--llm", default="offline", help='"offline" (default), "anthropic[:model]", '
                                                     '"openai:<model>" or "ollama:<model>"')
    ap.add_argument("--task", default=None, choices=sorted(TASKS), help="run one agent task and exit")
    ap.add_argument("-h", "--help", action="store_true")
    a, rest = ap.parse_known_args(argv)
    if a.help:
        ap.print_help()
        print()
    box = cli_toolbox(rest + (["--help"] if a.help else []))
    chat = Chat(box, llm=a.llm)

    def show(reply):
        for s in reply.steps:
            print(f"  [tool] {s['tool']} {json.dumps(s['input'])}")
        print("\n" + reply.text + "\n")
        g = reply.grounding
        if g:
            print(f"  [grounding] {g['checked'] - len(g['unsupported'])} of {g['checked']} numbers traced to a "
                  "tool result" + (": not traced " + ", ".join(u["text"] for u in g["unsupported"])
                                   if g["unsupported"] else ""))
        while reply.pending:
            ok = input(f"Approve {reply.pending['tool']} {json.dumps(reply.pending['input'])}? [y/N] ")
            reply = chat.approve() if ok.strip().lower().startswith("y") else chat.decline()
            show(reply)
            return

    if a.task:
        show(chat.run_task(a.task))
        return
    print(f"Chatting with the model ({chat.llm.name}). Tasks: {', '.join(sorted(TASKS))} "
          "(type the name to run one). Empty line to quit.", file=sys.stderr)
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break
        if not q:
            break
        show(chat.run_task(q) if q in TASKS else chat.ask(q))


if __name__ == "__main__":  # pragma: no cover
    main()
