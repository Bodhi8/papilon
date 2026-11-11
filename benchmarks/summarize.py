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
"""Summarize recovery.py output: ROAS calibration and plan outcomes by fit and plan rule.

Usage: python benchmarks/summarize.py std.jsonl [hard.jsonl ...]
"""
import json
import sys

import numpy as np

CH = ["paid_search", "paid_social", "online_video", "display"]


def load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p)]
    return rows


def summarize(rows, label):
    print(f"\n=== {label}: {len(rows)} brands ===")
    fits = list(rows[0]["fits"])
    for f in fits:
        cov = {c: np.mean([r["fits"][f]["roas"][c]["p05"] <= r["fits"][f]["roas"][c]["true"] <= r["fits"][f]["roas"][c]["p95"]
                           for r in rows]) for c in CH}
        mae = {c: np.mean([abs(r["fits"][f]["roas"][c]["est"] - r["fits"][f]["roas"][c]["true"]) for r in rows]) for c in CH}
        bias = {c: np.mean([r["fits"][f]["roas"][c]["est"] - r["fits"][f]["roas"][c]["true"] for r in rows]) for c in CH}
        print(f"\n[{f}] ROAS 90% coverage " + "  ".join(f"{c.split('_')[-1]} {cov[c]:.0%}" for c in CH)
              + f"  | all {np.mean(list(cov.values())):.0%}")
        print("      ROAS abs error " + "  ".join(f"{c.split('_')[-1]} {mae[c]:.2f}" for c in CH)
              + "  | bias " + "  ".join(f"{bias[c]:+.2f}" for c in CH))
        print(f"      {'plan':15s} {'promised':>8s} {'true':>7s} {'capture':>8s} {'med cap':>8s} {'P(loss)':>8s} "
              f"{'in 90%':>7s} {'over-promise':>12s} {'moved':>6s} {'q13 true':>8s} {'q13 in90':>8s}")
        for p in rows[0]["fits"][f]["plans"]:
            pr = np.array([r["fits"][f]["plans"][p]["promised"] for r in rows])
            tg = np.array([r["fits"][f]["plans"][p]["true_gain"] for r in rows])
            best = np.array([r["best_gain"] for r in rows])
            cap = tg / np.maximum(best, 1e-9)
            inr = np.mean([r["fits"][f]["plans"][p]["p05"] <= r["fits"][f]["plans"][p]["true_gain"] <= r["fits"][f]["plans"][p]["p95"]
                           for r in rows])
            mv = np.mean([r["fits"][f]["plans"][p]["moved"] for r in rows])
            q13 = np.mean([r["fits"][f]["plans"][p]["q13_true"] for r in rows])
            q13in = np.mean([r["fits"][f]["plans"][p]["q13_p05"] <= r["fits"][f]["plans"][p]["q13_true"] <= r["fits"][f]["plans"][p]["q13_p95"]
                             for r in rows])
            print(f"      {p:15s} {pr.mean():8.2f} {tg.mean():7.2f} {tg.sum() / best.sum():8.0%} {np.median(cap):8.0%} "
                  f"{np.mean(tg < 0):8.0%} {inr:7.0%} {np.mean(pr - tg):12.2f} {mv:6.1f} {q13:8.1f} {q13in:8.0%}")
    print(f"\n      best possible gain: mean {np.mean([r['best_gain'] for r in rows]):.2f} per week")
    f0 = fits[0]
    print(f"      video delayed chosen ({f0}): {np.mean([r['fits'][f0]['video_delayed_prob'] > 0.5 for r in rows]):.0%}"
          f"   pareto k > 0.7: {np.mean([max(r['fits'][f0]['pareto_k'].values()) > 0.7 for r in rows]):.0%}")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        summarize(load([path]), path)
