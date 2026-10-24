import numpy as np

def shannon_entropy(data):
    _, counts = np.unique(data, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

def joint_entropy(x, y):
    joint = list(zip(x, y))
    return shannon_entropy(joint)
