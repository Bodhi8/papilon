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
"""Grade a candidate entry without touching the published leaderboard.

A candidate is an adapter for another library that is still being verified. Its scores
are written to ``benchmarks/candidates/results/<name>.json`` and are not added to
``LEADERBOARD.md``. An entry moves to the leaderboard only after the full run reproduces
and the library's maintainers have been invited to review its configuration.

    python benchmarks/run_candidate.py pymc_marketing --worlds 16 --jobs 3

Brands and tracks are the leaderboard's own: the first ``--worlds`` of
``pycausalsim.futures.marketing_worlds(60, seed=7, ...)`` per track.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from leaderboard import TRACKS, summarize  # noqa: E402

# name -> (module in benchmarks/candidates, function, description)
CANDIDATES = {
    "pymc_marketing": ("candidates.pymc_marketing", "pymc_marketing_default",
                       "PyMC-Marketing MMM, default priors, geometric adstock, logistic saturation"),
}


def _one(job):
    name, track, i = job
    from pycausalsim.futures import grade, marketing_worlds
    module, func, _ = CANDIDATES[name]
    entry = getattr(importlib.import_module(module), func)
    worlds = marketing_worlds(60, seed=7, **TRACKS[track])
    t = time.time()
    try:
        recs = grade(entry, worlds, ids=[i]).to_dict("records")
        err = None
    except Exception as exc:  # one failed brand must not lose the rest of the run
        recs, err = [], f"{type(exc).__name__}: {exc}"
    return track, i, recs, err, round(time.time() - t, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("name", choices=sorted(CANDIDATES))
    ap.add_argument("--worlds", type=int, default=16)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--tracks", nargs="+", default=list(TRACKS), choices=list(TRACKS))
    ap.add_argument("--quiet", action="store_true", help="do not print scores, only progress")
    a = ap.parse_args()

    jobs = [(a.name, t, i) for t in a.tracks for i in range(a.worlds)]
    rows: dict = {t: [] for t in a.tracks}
    failures, seconds = [], []
    with Pool(a.jobs) as pool:
        for n, (track, i, recs, err, secs) in enumerate(pool.imap_unordered(_one, jobs), 1):
            rows[track].extend(recs)
            seconds.append(secs)
            if err:
                failures.append({"track": track, "world": i, "error": err})
            print(f"[{n}/{len(jobs)}] {track} world {i}: {'FAILED' if err else 'ok'} in {secs}s", flush=True)

    import pycausalsim
    versions = {"pycausalsim": pycausalsim.__version__}
    for lib in ("pymc_marketing", "pymc", "papilon"):
        try:
            versions[lib] = importlib.import_module(lib).__version__
        except Exception:
            pass
    result = {"candidate": a.name, "description": CANDIDATES[a.name][2], "status": "unverified",
              "versions": versions, "seed": 7, "brands_per_track": a.worlds,
              "seconds_per_brand_mean": round(sum(seconds) / max(len(seconds), 1), 1),
              "failures": failures,
              "tracks": {t: summarize(pd.DataFrame(r)) for t, r in rows.items() if r},
              "rows": {t: r for t, r in rows.items()}}
    out = HERE / "candidates" / "results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.name}.json"
    path.write_text(json.dumps(result, indent=2, default=float))
    print(f"wrote {path.relative_to(HERE.parent)}; {len(failures)} failed brand(s)")
    if not a.quiet:
        print(json.dumps(result["tracks"], indent=2))


if __name__ == "__main__":
    main()
