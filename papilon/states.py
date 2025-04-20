from collections import Counter
import numpy as np

def count_microstates(data):
    return dict(Counter(map(tuple, data)))

def macrostate_entropy(microstates):
    total = sum(microstates.values())
    probs = [v / total for v in microstates.values()]
    return -sum(p * np.log2(p) for p in probs)
