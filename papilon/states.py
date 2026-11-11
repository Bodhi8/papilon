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
from collections import Counter
import numpy as np

def count_microstates(data):
    return dict(Counter(map(tuple, data)))

def macrostate_entropy(microstates):
    total = sum(microstates.values())
    probs = [v / total for v in microstates.values()]
    return -sum(p * np.log2(p) for p in probs)
