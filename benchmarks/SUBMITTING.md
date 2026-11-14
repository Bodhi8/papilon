# Entering the leaderboard

The leaderboard is open to any marketing mix model, from any library or none. Its
purpose is to learn which modeling choices produce estimates and budget plans that
hold up, on brands where the truth is known.

## What an entry is

A Python function:

```python
def my_model(df, info):
    # df: one row per week with date, spend per channel, controls, sales
    # info: media, target, date_col, controls, current (weekly spend by channel), total
    ...
    return {
        "roas": {channel: (estimate, p05, p95) for channel in info["media"]},
        "plan": spend_per_week_by_channel,     # same total as info["total"]; or None
    }
```

## Rules

1. The function sees only `df` and `info`. It must not read the answer key, the
   generator's parameters or the world's index to look anything up.
2. One configuration for every brand in a track. No per-brand tuning.
3. Plans keep the total budget and are non-negative. Choosing how far to move is part
   of what is being scored.
4. Fix the random seed so the result can be reproduced.
5. State the library version and settings in the pull request.

## How to submit

1. Add the function and an `ENTRIES` line to `benchmarks/leaderboard.py`.
2. Run `python benchmarks/leaderboard.py` and commit the regenerated `LEADERBOARD.md`
   and `leaderboard.json`.
3. Open a pull request. Entries are merged when the run reproduces.

Entries are described by what they do, in the submitter's words. If you maintain a
library and want an entry corrected or removed, open an issue.

## What the score means

Simulated brands are kinder than real data. A good score shows a model can recover
known returns under these conditions; it does not show it is right about your brand.

## Candidates

An adapter for another library starts as a candidate, not as a leaderboard entry. It lives
in `benchmarks/candidates/` and is graded with

```bash
python benchmarks/run_candidate.py pymc_marketing --worlds 16 --jobs 3
```

which writes its scores to `benchmarks/candidates/results/` (not committed) and leaves
`LEADERBOARD.md` untouched. A candidate becomes an entry when three things are true:

1. The full run completes on every brand of every track it enters, at the settings stated
   in the adapter.
2. The adapter uses the library as its own documentation recommends, with one
   configuration for every brand.
3. The library's maintainers have been sent the adapter and its scores and given time to
   correct the configuration.

Until then a candidate's scores are not published.
