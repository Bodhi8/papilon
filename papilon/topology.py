import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

def create_topology_graph(edges_df, source_col="source", target_col="target", weight_col=None):
    """
    Create a NetworkX graph from a DataFrame representing edges in a complex ecosystem.
    """
    if weight_col:
        G = nx.from_pandas_edgelist(edges_df, source=source_col, target=target_col, edge_attr=weight_col)
    else:
        G = nx.from_pandas_edgelist(edges_df, source=source_col, target=target_col)
    return G

def visualize_topology_graph(G, title="Topology Graph", node_size=500, with_labels=True):
    """
    Visualize a NetworkX graph with matplotlib.
    """
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=with_labels, node_size=node_size, node_color='skyblue', edge_color='gray')
    plt.title(title)
    plt.show()
