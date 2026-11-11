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
