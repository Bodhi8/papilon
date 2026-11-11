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
"""A typed knowledge graph of a brand's marketing: what drives what, and what is known.

A marketing mix model rests on assumptions that usually live in someone's head:
which channels feed which, what else moves sales, which of those drivers the
data can see. :class:`MarketingGraph` writes them down as a small ontology and
then puts them to work:

* it compiles into model settings (controls, carryover candidates, lift-test
  priors) for :class:`papilon.MMM`;
* it audits the setup, naming every channel whose estimate depends on a driver
  that was declared but has no data behind it;
* it exports the same structure for ``pycausalsim.futures.worlds_from_graph``,
  which simulates brands with that structure and a known answer key;
* it describes itself in plain text and JSON-LD, so a language model answering
  questions about the brand reads declared relations instead of guessing them.

Node types: ``Outcome``, ``Channel``, ``Driver`` (seasonality, price, launches,
competitor moves), ``Group`` and ``Experiment``. Relations: ``drives``
(driver or channel -> outcome), ``sets_spend`` (driver -> channel: the brand
spends more when this driver is high), ``mediates`` (channel -> channel: part of
the first channel's effect arrives through the second), ``belongs_to`` and
``measured_by``. A driver with both ``sets_spend`` and ``drives`` edges is a
confounder of that channel.

A graph makes assumptions explicit and checkable. It does not make them true.
"""

from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["MarketingGraph"]

NAMESPACE = "https://vector1.ai/ns/marketing#"
RELATIONS = ("drives", "sets_spend", "mediates", "belongs_to", "measured_by")
_STAGES = ("upper", "mid", "lower")


class MarketingGraph:
    """Declared structure of a brand's marketing. See the module docstring."""

    def __init__(self, outcome: str = "sales") -> None:
        self.g = nx.MultiDiGraph()
        self.outcome = outcome
        self.g.add_node(outcome, type="Outcome")

    # ------------------------------------------------------------- building
    def channel(self, name: str, stage: str = "lower", group: str | None = None,
                column: str | None = None) -> "MarketingGraph":
        """A paid channel. ``stage`` is its place in the funnel: upper, mid or lower."""
        if stage not in _STAGES:
            raise ValueError(f"stage must be one of {_STAGES}")
        self.g.add_node(name, type="Channel", stage=stage, column=column or name)
        self._edge(name, "drives", self.outcome)
        if group:
            self.g.add_node(group, type="Group")
            self._edge(name, "belongs_to", group)
        return self

    def driver(self, name: str, affects: list[str], column: str | None = None,
               modeled: bool = False, kind: str = "driver") -> "MarketingGraph":
        """A non-media driver and what it affects (channels and/or the outcome).

        ``column`` is the data column that measures it. ``modeled=True`` marks a
        driver the model handles without a column (yearly seasonality and trend).
        A driver with neither is unobserved.
        """
        self.g.add_node(name, type="Driver", column=column, modeled=bool(modeled), kind=kind)
        for a in affects:
            if a != self.outcome and self.g.nodes.get(a, {}).get("type") != "Channel":
                raise ValueError(f"{a!r} is not the outcome or a declared channel")
            self._edge(name, "drives" if a == self.outcome else "sets_spend", a)
        return self

    def event(self, name: str, affects: list[str], column: str | None = None) -> "MarketingGraph":
        """A dated event such as a launch or a competitor promotion (a driver of kind ``event``)."""
        return self.driver(name, affects, column=column, kind="event")

    def mediates(self, source: str, through: str) -> "MarketingGraph":
        """Part of ``source``'s effect arrives through ``through`` (video lifting branded search)."""
        for c in (source, through):
            self._need(c, "Channel")
        self._edge(source, "mediates", through)
        return self

    def experiment(self, name: str, channel: str, roas: float, sd: float) -> "MarketingGraph":
        """A lift test or geo experiment: measured return on spend and its standard error."""
        self._need(channel, "Channel")
        self.g.add_node(name, type="Experiment", roas=float(roas), sd=float(sd))
        self._edge(channel, "measured_by", name)
        return self

    def propose(self, source: str, relation: str, target: str, by: str = "model", note: str = "") -> "MarketingGraph":
        """Add an edge as a proposal (for example from a language model). It is
        listed and described but ignored by :meth:`model_kwargs` and :meth:`audit`
        until :meth:`confirm` is called."""
        if relation not in RELATIONS:
            raise ValueError(f"relation must be one of {RELATIONS}")
        for n in (source, target):
            if n not in self.g:
                raise ValueError(f"unknown node {n!r}")
        self._edge(source, relation, target, status="proposed", by=by, note=note)
        return self

    def confirm(self, source: str, relation: str, target: str) -> "MarketingGraph":
        self.g.edges[source, target, relation]["status"] = "confirmed"
        return self

    def _edge(self, a, rel, b, status="confirmed", **attrs):
        self.g.add_edge(a, b, key=rel, relation=rel, status=status, **attrs)

    def _need(self, name, typ):
        if self.g.nodes.get(name, {}).get("type") != typ:
            raise ValueError(f"{name!r} is not a declared {typ.lower()}")

    # -------------------------------------------------------------- reading
    def _nodes(self, typ):
        return [n for n, d in self.g.nodes(data=True) if d.get("type") == typ]

    @property
    def channels(self) -> list[str]:
        return self._nodes("Channel")

    def _out(self, node, rel, confirmed=True):
        return [b for _, b, k, d in self.g.out_edges(node, keys=True, data=True)
                if k == rel and (d["status"] == "confirmed" or not confirmed)]

    def _in(self, node, rel, confirmed=True):
        return [a for a, _, k, d in self.g.in_edges(node, keys=True, data=True)
                if k == rel and (d["status"] == "confirmed" or not confirmed)]

    def confounders(self, channel: str) -> list[str]:
        """Drivers that both set this channel's spend and drive the outcome."""
        return [d for d in self._in(channel, "sets_spend") if self.outcome in self._out(d, "drives")]

    def edges(self) -> pd.DataFrame:
        """Every relation, one per row, with its status (confirmed or proposed)."""
        return pd.DataFrame([{"source": a, "relation": k, "target": b, "status": d["status"],
                              "by": d.get("by", ""), "note": d.get("note", "")}
                             for a, b, k, d in self.g.edges(keys=True, data=True)],
                            columns=["source", "relation", "target", "status", "by", "note"])

    def what_moves(self, node: str) -> pd.DataFrame:
        """Everything downstream of ``node`` and the chain of relations that connects it.

        The answer to "what else changes if this changes?", read from the declared
        relations (``belongs_to`` and ``measured_by`` are not causal and are skipped).
        """
        causal = nx.DiGraph()
        for a, b, k, d in self.g.edges(keys=True, data=True):
            if k in ("drives", "sets_spend", "mediates") and d["status"] == "confirmed":
                causal.add_edge(a, b, relation=k)
        rows = []
        if node in causal:
            for target in nx.descendants(causal, node):
                for path in nx.all_simple_paths(causal, node, target):
                    rows.append({"affected": target, "steps": len(path) - 1, "path": " -> ".join(
                        f"{path[i]} [{causal.edges[path[i], path[i + 1]]['relation']}]" for i in range(len(path) - 1)
                    ) + f" -> {target}"})
        return pd.DataFrame(rows, columns=["affected", "steps", "path"]).sort_values(["steps", "affected"],
                                                                                    ignore_index=True)

    # ------------------------------------------------------------ compiling
    def model_kwargs(self, df: pd.DataFrame | None = None) -> dict:
        """Settings for :class:`papilon.MMM` implied by the graph.

        Media are the channels; controls are the columns of every observed driver
        of the outcome; lift tests become ``roi_priors`` (several tests on one
        channel are pooled by inverse variance). With ``df``, missing columns
        raise instead of failing later.
        """
        media = [self.g.nodes[c]["column"] for c in self.channels]
        controls = []
        for d in self._nodes("Driver"):
            col = self.g.nodes[d]["column"]
            if col and self.outcome in self._out(d, "drives") and col not in controls:
                controls.append(col)
        priors = {}
        for c in self.channels:
            tests = [self.g.nodes[e] for e in self._out(c, "measured_by")]
            if tests:
                w = np.array([1 / t["sd"] ** 2 for t in tests])
                priors[self.g.nodes[c]["column"]] = (float(np.sum(w * [t["roas"] for t in tests]) / w.sum()),
                                                     float(np.sqrt(1 / w.sum())))
        if df is not None:
            missing = [c for c in media + controls + [self.outcome] if c not in df.columns]
            if missing:
                raise ValueError(f"columns declared in the graph but missing from the data: {missing}")
        return {"target": self.outcome, "media": media, "controls": controls, "roi_priors": priors}

    def carryover_candidates(self) -> dict[str, list[str]]:
        """Families for ``MMM.compare_carryover``: upper-funnel channels may peak after they run."""
        upper = [self.g.nodes[c]["column"] for c in self.channels if self.g.nodes[c]["stage"] == "upper"]
        out = {"all_geometric": []}
        if upper:
            out["upper_funnel_delayed"] = upper
        return out

    def build_model(self, df: pd.DataFrame, date_col: str, **kwargs):
        """A :class:`papilon.MMM` configured from the graph. ``kwargs`` override or add settings."""
        from .mmm import MMM
        kw = self.model_kwargs(df) | kwargs
        return MMM(df, date_col=date_col, **kw)

    # -------------------------------------------------------------- auditing
    def audit(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """What the declared structure means for trusting each estimate.

        One row per finding, most serious first:

        ``unobserved_confounder``  a driver sets the channel's spend and drives the
            outcome, and there is no data for it. The channel's return is not
            identified from this data; the row says whether a lift test covers it.
        ``mediated``  part of a channel's effect arrives through another channel,
            so the model credits that part to the second channel.
        ``missing_column``  a declared column is not in ``df``.
        ``weak_link``  (with ``df``) a declared, observed ``sets_spend`` or
            ``drives`` relation shows no linear association in the data.
        ``proposed``  an edge awaiting confirmation; it is not used.
        """
        rows = []
        add = lambda sev, issue, subject, detail: rows.append(
            {"severity": sev, "issue": issue, "subject": subject, "detail": detail})
        for c in self.channels:
            tested = bool(self._out(c, "measured_by"))
            for d in self.confounders(c):
                n = self.g.nodes[d]
                if not n["column"] and not n["modeled"]:
                    add("high" if not tested else "low", "unobserved_confounder", c,
                        f"{d} raises {c} spend and {self.outcome} together and has no data. "
                        + ("A lift test on this channel calibrates it." if tested else
                           f"{c}'s return is not identified from this data: run a lift test and add it "
                           f"with experiment(), or add a column for {d}."))
            for m in self._out(c, "mediates"):
                add("medium", "mediated", c,
                    f"part of {c}'s effect arrives through {m}; the model credits that part to {m}, "
                    f"so {c} is understated and {m} overstated. Cutting {c} also lowers what {m} can earn.")
        for a, b, k, d in self.g.edges(keys=True, data=True):
            if d["status"] == "proposed":
                add("info", "proposed", f"{a} -> {b}", f"{k} proposed by {d.get('by') or 'unknown'}: "
                    f"{d.get('note') or 'no note'}. Not used until confirmed.")
        if df is not None:
            for n, d in self.g.nodes(data=True):
                col = d.get("column") if d.get("type") in ("Channel", "Driver") else None
                if col and col not in df.columns:
                    add("high", "missing_column", n, f"column {col!r} is not in the data")
            for dn in self._nodes("Driver"):
                col = self.g.nodes[dn]["column"]
                if not col or col not in df.columns:
                    continue
                for tgt in self._out(dn, "sets_spend") + self._out(dn, "drives"):
                    tcol = self.outcome if tgt == self.outcome else self.g.nodes[tgt]["column"]
                    if tcol not in df.columns or df[col].std() == 0 or df[tcol].std() == 0:
                        continue
                    r, p = stats.pearsonr(df[col].astype(float), df[tcol].astype(float))
                    if p > 0.10:
                        add("info", "weak_link", f"{dn} -> {tgt}",
                            f"declared, but the data shows no linear association (r = {r:+.2f}, p = {p:.2f}). "
                            "This is a screen: a real effect can hide behind other drivers.")
        order = {"high": 0, "medium": 1, "low": 2, "info": 3}
        out = pd.DataFrame(rows, columns=["severity", "issue", "subject", "detail"])
        return out.sort_values("severity", key=lambda s: s.map(order), kind="stable", ignore_index=True)

    # ------------------------------------------------------------ exporting
    def to_dict(self) -> dict:
        """Plain-data form, also the input to ``pycausalsim.futures.worlds_from_graph``."""
        return {"outcome": self.outcome,
                "nodes": [{"id": n, **d} for n, d in self.g.nodes(data=True)],
                "edges": [{"source": a, "target": b, **d} for a, b, _, d in self.g.edges(keys=True, data=True)]}

    @classmethod
    def from_dict(cls, data: dict) -> "MarketingGraph":
        out = cls(data["outcome"])
        for n in data["nodes"]:
            n = dict(n)
            out.g.add_node(n.pop("id"), **n)
        for e in data["edges"]:
            e = dict(e)
            a, b = e.pop("source"), e.pop("target")
            out.g.add_edge(a, b, key=e["relation"], **e)
        return out

    def to_jsonld(self) -> str:
        """JSON-LD using the Vector1 marketing vocabulary, for use beside a wider ontology."""
        ctx = {"@vocab": NAMESPACE, "id": "@id", "type": "@type"}
        items = []
        for n, d in self.g.nodes(data=True):
            item = {"id": f"_:{n}", "type": d["type"], "name": n}
            item |= {k: v for k, v in d.items() if k != "type" and v is not None}
            for _, b, k, ed in self.g.out_edges(n, keys=True, data=True):
                item.setdefault(k, []).append({"id": f"_:{b}", "status": ed["status"]})
            items.append(item)
        return json.dumps({"@context": ctx, "@graph": items}, indent=2)

    def describe(self) -> str:
        """The graph in plain sentences: grounding text for a person or a language model."""
        lines = [f"Outcome: {self.outcome}."]
        for c in self.channels:
            n = self.g.nodes[c]
            grp = self._out(c, "belongs_to")
            s = f"Channel {c}: {n['stage']}-funnel" + (f", in group {grp[0]}" if grp else "") + "."
            conf = self.confounders(c)
            if conf:
                s += " Confounded by " + ", ".join(
                    f"{d} ({'observed' if self.g.nodes[d]['column'] else 'modeled' if self.g.nodes[d]['modeled'] else 'UNOBSERVED'})"
                    for d in conf) + "."
            for m in self._out(c, "mediates"):
                s += f" Part of its effect arrives through {m}."
            for e in self._out(c, "measured_by"):
                t = self.g.nodes[e]
                s += f" Lift test {e}: ROAS {t['roas']:.2f} (se {t['sd']:.2f})."
            lines.append(s)
        for d in self._nodes("Driver"):
            n = self.g.nodes[d]
            state = f"column {n['column']}" if n["column"] else "modeled" if n["modeled"] else "no data"
            lines.append(f"{'Event' if n['kind'] == 'event' else 'Driver'} {d} ({state}) affects "
                         + ", ".join(self._out(d, "sets_spend") + self._out(d, "drives")) + ".")
        prop = self.edges().query("status == 'proposed'")
        for _, r in prop.iterrows():
            lines.append(f"PROPOSED, not confirmed: {r.source} {r.relation} {r.target}.")
        return "\n".join(lines)
