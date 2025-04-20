import pandas as pd
import networkx as nx
from sklearn.feature_selection import f_regression

def infer_causal_structure(df, threshold=0.01, visualize=True):
    numeric_df = df.select_dtypes(include='number').dropna()
    features = numeric_df.columns.tolist()
    G = nx.DiGraph()

    for target in features:
        for predictor in [col for col in features if col != target]:
            X = numeric_df[[predictor]]
            y = numeric_df[target]
            _, pval = f_regression(X, y)
            if pval[0] < threshold:
                G.add_edge(predictor, target, weight=1 - pval[0])

    if visualize:
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=1500, font_size=10, edge_color='gray')
        plt.title('Inferred Causal Graph')
        plt.show()

    return G
