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
"""Causal discovery from observational data: the PC algorithm with partial correlations.

``discover_structure`` finds which variables are directly dependent after
conditioning on others (the skeleton), orients what the data can orient (colliders,
then Meek's rules), and leaves the rest undirected. Background knowledge enters as
``tiers``: a variable in a later tier cannot cause one in an earlier tier, and
``forbidden`` edges are never added.

Weekly marketing data violates the independence the tests assume: series trend and
repeat with the season. ``preprocess="calendar"`` (the default) tests what is left of
each series after removing a trend and yearly seasonality; "difference" tests
period-to-period changes; "none" tests levels. Keep ``alpha`` small.

Linear tests are good at finding how spend and other drivers move together (budgets
that follow demand, channels planned together). They are poor at finding media
effects on sales, which are carried over, saturating and small next to noise; that is
the marketing mix model's job. Treat the result as evidence to set against the
analyst's graph (``MarketingGraph.check_against_data``), not as the graph.
"""
from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["discover_structure", "partial_correlation_test"]


def partial_correlation_test(data: np.ndarray, i: int, j: int, cond: tuple, n: int) -> float:
    """p-value of the Fisher z test that columns i and j are independent given ``cond``."""
    idx = [i, j] + list(cond)
    C = np.corrcoef(data[:, idx], rowvar=False)
    try:
        P = np.linalg.pinv(C)
    except np.linalg.LinAlgError:
        return 1.0
    r = -P[0, 1] / np.sqrt(abs(P[0, 0] * P[1, 1]))
    r = float(np.clip(r, -0.999999, 0.999999))
    dof = n - len(cond) - 3
    if dof <= 0:
        return 1.0
    z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(dof)
    return float(2 * stats.norm.sf(abs(z)))


def discover_structure(df: pd.DataFrame, columns: list[str] | None = None, alpha: float = 0.01,
                       max_cond: int = 3, tiers: dict[str, int] | None = None,
                       forbidden: list[tuple[str, str]] | None = None, preprocess: str = "calendar",
                       period: float = 52.18) -> nx.DiGraph:
    """Run the PC algorithm. Returns a DiGraph: a directed edge a -> b, or both a -> b and
    b -> a with ``undirected=True`` when the data cannot say which way it runs. Each edge
    carries ``p_value`` (of the unconditional test) and the graph carries ``sepsets``."""
    cols = list(columns or df.select_dtypes("number").columns)
    if preprocess not in ("calendar", "difference", "none"):
        raise ValueError('preprocess must be "calendar", "difference" or "none"')
    X = df[cols].astype(float).reset_index(drop=True)
    if preprocess == "difference":
        X = X.diff().dropna()
    elif preprocess == "calendar":
        t = np.arange(len(X)) / period
        D = np.column_stack([np.ones(len(X)), t] + [f(2 * np.pi * k * t) for k in (1, 2) for f in (np.sin, np.cos)])
        coef, *_ = np.linalg.lstsq(D, X.to_numpy(), rcond=None)
        X = pd.DataFrame(X.to_numpy() - D @ coef, columns=cols)
    X = X.loc[:, X.std() > 0]
    cols = list(X.columns)
    data, n, k = X.to_numpy(), len(X), len(cols)
    tiers = dict(tiers or {})
    forbid = {tuple(e) for e in (forbidden or [])}
    adj = {a: set(range(k)) - {a} for a in range(k)}
    sepset, pval = {}, {}
    for a, b in combinations(range(k), 2):
        pval[(a, b)] = pval[(b, a)] = partial_correlation_test(data, a, b, (), n)
    for size in range(max_cond + 1):
        for a in range(k):
            for b in sorted(adj[a]):
                if b < a or b not in adj[a]:
                    continue
                others = (adj[a] | adj[b]) - {a, b}
                for cond in combinations(sorted(others), size):
                    if partial_correlation_test(data, a, b, cond, n) > alpha:
                        adj[a].discard(b); adj[b].discard(a)
                        sepset[(a, b)] = sepset[(b, a)] = set(cond)
                        break
    # start undirected, then apply background knowledge
    directed = set()
    undirected = {frozenset((a, b)) for a in range(k) for b in adj[a]}

    def orient(a, b):
        e = frozenset((a, b))
        if e in undirected:
            undirected.discard(e); directed.add((a, b)); return True
        return False
    for e in list(undirected):
        a, b = tuple(e)
        ta, tb = tiers.get(cols[a]), tiers.get(cols[b])
        if ta is not None and tb is not None and ta != tb:
            orient(a, b) if ta < tb else orient(b, a)
        elif (cols[a], cols[b]) in forbid:
            orient(b, a)
        elif (cols[b], cols[a]) in forbid:
            orient(a, b)
    # colliders: a - c - b with a, b not adjacent and c not in their separating set
    for c in range(k):
        nb = [x for x in range(k) if frozenset((x, c)) in undirected or (x, c) in directed or (c, x) in directed]
        for a, b in combinations(nb, 2):
            if b in adj[a] or c in sepset.get((a, b), set()):
                continue
            if (c, a) not in directed:
                orient(a, c)
            if (c, b) not in directed:
                orient(b, c)
    # Meek's rules until nothing changes
    changed = True
    while changed:
        changed = False
        for e in list(undirected):
            for a, b in (tuple(e), tuple(e)[::-1]):
                if frozenset((a, b)) not in undirected:
                    continue
                # R1: c -> a - b, c and b not adjacent  =>  a -> b
                if any((c, a) in directed and b not in adj[c] and c != b for c in range(k)):
                    changed |= orient(a, b); continue
                # R2: a -> c -> b and a - b  =>  a -> b
                if any((a, c) in directed and (c, b) in directed for c in range(k)):
                    changed |= orient(a, b); continue
                # R3: a - c -> b, a - d -> b, c and d not adjacent  =>  a -> b
                cs = [c for c in range(k) if frozenset((a, c)) in undirected and (c, b) in directed]
                if any(d not in adj[c] for c, d in combinations(cs, 2)):
                    changed |= orient(a, b)
    G = nx.DiGraph()
    G.add_nodes_from(cols)
    for a, b in directed:
        G.add_edge(cols[a], cols[b], undirected=False, p_value=pval[(a, b)])
    for e in undirected:
        a, b = tuple(e)
        G.add_edge(cols[a], cols[b], undirected=True, p_value=pval[(a, b)])
        G.add_edge(cols[b], cols[a], undirected=True, p_value=pval[(a, b)])
    G.graph["sepsets"] = {(cols[a], cols[b]): sorted(cols[x] for x in s) for (a, b), s in sepset.items() if a < b}
    G.graph["alpha"], G.graph["preprocess"], G.graph["n"] = alpha, preprocess, n
    return G
