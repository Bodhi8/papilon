import numpy as np
import pandas as pd
from itertools import product

def grid_search_optimize(df, feature_ranges, objective_fn, constraints=None):
    keys, values = zip(*feature_ranges.items())
    all_configs = [dict(zip(keys, v)) for v in product(*values)]

    if constraints:
        all_configs = list(filter(constraints, all_configs))

    results = []
    for config in all_configs:
        score = objective_fn(config)
        results.append({**config, 'score': score})

    return pd.DataFrame(results).sort_values('score', ascending=False)
