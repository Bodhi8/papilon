"""Fails when a public name, method or parameter recorded in api_snapshot.json disappears.

Additions are fine. To record new public API after adding it, run
``python -c "import json, tests._api as a; json.dump(a.collect(), open('tests/api_snapshot.json','w'), indent=1, sort_keys=True)"``.
"""
import json
from pathlib import Path

from tests import _api

SNAP = json.loads((Path(__file__).parent / "api_snapshot.json").read_text())


def test_no_public_api_removed():
    now = _api.collect()
    problems = []
    for name, rec in SNAP.items():
        if name not in now:
            problems.append(f"{name} removed")
            continue
        cur = now[name]
        if rec["params"] is not None and cur["params"] is not None:
            gone = [p for p in rec["params"] if p not in cur["params"]]
            if gone:
                problems.append(f"{name}: parameters removed {gone}")
        for meth, ps in (rec.get("methods") or {}).items():
            if meth not in (cur.get("methods") or {}):
                problems.append(f"{name}.{meth} removed")
            elif ps is not None and cur["methods"][meth] is not None:
                gone = [p for p in ps if p not in cur["methods"][meth]]
                if gone:
                    problems.append(f"{name}.{meth}: parameters removed {gone}")
    assert not problems, "\n".join(problems)
