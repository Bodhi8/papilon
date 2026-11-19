"""Chat and agent layer: the offline agent, grounding, approval, message formats, the terminal chat."""
import json

import pytest

import papilon as pp
from papilon.ai import ToolBox
from papilon.chat import (TASKS, AnthropicLLM, Chat, OfflineAgent, OpenAILLM, Reply, check_grounding,
                          main, make_llm)

MEDIA = ["paid_search", "paid_social", "online_video", "display"]


@pytest.fixture(scope="module")
def box():
    df, _ = pp.make_synthetic_mmm()
    m = pp.MMM(df, "sales", MEDIA, "date", controls=["price_index", "promo"],
               delayed=["online_video"], seed=11).fit(n_starts=2)
    return ToolBox(m)


# ----------------------------------------------------------------- shared core

class FakeBox:
    system_prompt = "fake"
    mutating = frozenset({"change"})

    def __init__(self):
        self.calls = []

    def specs(self, fmt="tool_use"):
        return [{"name": "read", "description": "", "input_schema": {"type": "object"}},
                {"name": "change", "description": "", "input_schema": {"type": "object"}}]

    def call(self, name, args=None):
        self.calls.append(name)
        return {"tool": name, "result": {"value": 4.25, "share": 0.317}, "caveats": []}


class Scripted:
    name, tool_format = "scripted", "tool_use"

    def __init__(self, *outs):
        self.outs, self.seen = list(outs), []

    def complete(self, system, history, tools):
        self.seen.append(list(history))
        return self.outs.pop(0)


def call(name, i=0, **args):
    return {"text": "", "tool_calls": [{"id": f"c{i}", "name": name, "input": args}]}


def test_grounding_traces_numbers_and_flags_invented_ones():
    src = [("t", {"roas": 1.2345, "share": 0.0937, "change": -12.3, "note": "Pareto k 0.41"})]
    g = check_grounding("ROAS 1.23 (about 1.2), 9.4% of sales, cut 12.3, k 0.41.", src)
    assert g["ok"] and g["checked"] == 5
    g = check_grounding("ROAS 1.23, or about 2.47 if doubled.", src)
    assert not g["ok"] and [u["text"] for u in g["unsupported"]] == ["2.47"]
    # labels are not findings: small counts, interval levels, years, dates
    assert check_grounding("The 90% range over 3 channels in 2025-12-29.", []) == \
        {"ok": True, "checked": 0, "supported": [], "unsupported": []}


def test_loop_runs_tools_then_grounds_the_answer():
    box = FakeBox()
    llm = Scripted(call("read"), {"text": "Value 4.25, share 31.7%, and 7.77 made up.", "tool_calls": []})
    r = Chat(box, llm=llm).ask("go")
    assert box.calls == ["read"] and [s["tool"] for s in r.steps] == ["read"]
    assert [u["text"] for u in r.grounding["unsupported"]] == ["7.77"] and r.backend == "scripted"
    assert llm.seen[1][-1]["role"] == "tool"          # the model saw the tool result
    assert isinstance(r, Reply) and json.loads(r.to_json())["text"] == r.text


def test_actions_that_change_the_model_wait_for_approval():
    box = FakeBox()
    chat = Chat(box, llm=Scripted(call("change", x=1), {"text": "done", "tool_calls": []}))
    r = chat.ask("change it")
    assert r.pending == {"tool": "change", "input": {"x": 1}} and box.calls == []
    with pytest.raises(RuntimeError):
        chat.ask("something else")
    r = chat.approve()
    assert box.calls == ["change"] and r.text == "done" and r.pending is None

    box = FakeBox()
    chat = Chat(box, llm=Scripted(call("change"), {"text": "skipped", "tool_calls": []}))
    chat.ask("change it")
    r = chat.decline()
    assert box.calls == [] and "declined" in r.steps[0]["result"]["error"]

    box = FakeBox()
    r = Chat(box, llm=Scripted(call("change"), {"text": "no", "tool_calls": []}),
             confirm=lambda name, args: False).ask("change it")
    assert box.calls == [] and r.pending is None and r.text == "no"


def test_step_cap_stops_a_runaway_agent():
    box = FakeBox()
    llm = Scripted(*[call("read", i) for i in range(5)])
    r = Chat(box, llm=llm, max_steps=3).ask("loop")
    assert len(box.calls) == 3 and "stopped after 3 tool calls" in r.text


def test_message_formats_for_each_api():
    hist = [{"role": "user", "text": "q"},
            {"role": "assistant", "text": "", "tool_calls": [{"id": "a", "name": "read", "input": {}},
                                                             {"id": "b", "name": "read", "input": {"k": 1}}]},
            {"role": "tool", "id": "a", "name": "read", "input": {}, "result": {"x": 1}},
            {"role": "tool", "id": "b", "name": "read", "input": {"k": 1}, "result": {"x": 2}}]
    am = AnthropicLLM.messages(hist)
    assert [m["role"] for m in am] == ["user", "assistant", "user"]
    assert [b["type"] for b in am[2]["content"]] == ["tool_result", "tool_result"]
    om = OpenAILLM.messages("sys", hist)
    assert [m["role"] for m in om] == ["system", "user", "assistant", "tool", "tool"]
    assert json.loads(om[2]["tool_calls"][1]["function"]["arguments"]) == {"k": 1}


def test_backend_specs(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError):
        make_llm("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert make_llm("anthropic:claude-haiku-5-5").model == "claude-haiku-5-5"
    assert make_llm("ollama:llama3.1").base_url.endswith(":11434/v1")
    assert isinstance(make_llm(None, FakeBox()), OfflineAgent)
    with pytest.raises(ValueError):
        make_llm("nope")


# ------------------------------------------------------- papilon's offline agent

QUESTIONS = {
    "Which channel returns the most?": ["get_roi"],
    "How is paid search doing?": ["get_roi"],
    "Plan the budget with 10% more": ["optimize_budget", "plan_stability"],
    "What if I spend 60 on paid social?": ["response_curve"],
    "Which lift test should I run first?": ["value_of_lift_test"],
    "Forecast the next 8 weeks": ["forecast"],
    TASKS["overview"]: ["brief"],
    TASKS["audit"]: ["brief", "backtest"],
    TASKS["plan_quarter"]: ["brief", "optimize_budget", "plan_stability", "value_of_lift_test", "forecast"],
}


@pytest.mark.parametrize("question", list(QUESTIONS))
def test_offline_agent_plans_and_every_number_is_grounded(box, question):
    r = Chat(box).ask(question)
    assert [s["tool"] for s in r.steps] == QUESTIONS[question]
    assert r.grounding["ok"], r.grounding["unsupported"]
    assert r.grounding["checked"] > 0 and r.text and r.backend == "offline"
    assert "—" not in r.text and "–" not in r.text


def test_offline_agent_reads_the_question(box):
    chat = Chat(box)
    r = chat.ask("Plan the budget with 10% less")
    assert r.steps[0]["input"]["total"] < box.model.spend_raw.mean(axis=0).sum()
    assert "10% less" in r.text
    r = chat.ask("Plan next quarter's budget")
    assert "total" not in r.steps[1]["input"]                 # "13 weeks" is not a budget
    r = chat.ask("What if I double display?")
    lv = r.steps[0]["input"]["spend_levels"]
    assert r.steps[0]["input"]["channel"] == "display" and lv[1] == pytest.approx(2 * lv[0], rel=1e-3)
    r = chat.ask("tell me a joke")
    assert r.text.startswith("I am the built-in offline agent")
    assert "plan_quarter" in TASKS and chat.run_task("overview").steps[0]["tool"] == "brief"
    assert "**You:**" in chat.transcript() and "tool `brief`" in chat.transcript()


def test_terminal_chat(box, tmp_path, monkeypatch, capsys):
    df, _ = pp.make_synthetic_mmm()
    path = tmp_path / "sales.csv"
    df.to_csv(path, index=False)
    main(["--task", "overview", "--data", str(path), "--target", "sales", "--date", "date",
          "--media", ",".join(MEDIA), "--controls", "price_index,promo"])
    out = capsys.readouterr().out
    assert "[tool] brief" in out and "numbers traced to a tool result" in out
