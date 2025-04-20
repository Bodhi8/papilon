import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

def simulate_kde_scenarios(df, columns, n_samples=1000, bandwidth=1.0):
    data = df[columns].dropna().values
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(data)
    simulated = kde.sample(n_samples)
    return pd.DataFrame(simulated, columns=columns)
