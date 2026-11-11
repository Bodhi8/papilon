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
