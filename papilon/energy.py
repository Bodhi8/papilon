import numpy as np

def energy_score(probabilities):
    probabilities = np.clip(probabilities, 1e-10, 1.0)
    return -np.sum(np.log(probabilities))
