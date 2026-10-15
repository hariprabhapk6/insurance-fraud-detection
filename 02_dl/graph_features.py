"""
graph_features.py — Level: DL (Graph Neural Network component)
=================================================================
This environment has no internet access, so PyTorch / PyTorch-Geometric
cannot be installed. Rather than skip the GNN requirement, we implement
the graph-representation-learning IDEA with what's available: networkx
graph-structural features (degree, clustering coefficient, PageRank,
connected-component size) computed over the policyholder network. These
are the same signals a GNN's message-passing layers would learn to encode
(local connectivity + neighborhood influence), just computed in closed
form instead of learned end-to-end. This is documented explicitly as a
substitute, not presented as a full GNN.
"""
import pandas as pd
import networkx as nx
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
EDGES = BASE / "data" / "raw" / "policyholder_network.csv"
OUT = BASE / "02_dl" / "policyholder_graph_features.csv"


def main():
    edges = pd.read_csv(EDGES)
    G = nx.from_pandas_edgelist(edges, "source", "target", edge_attr=True)

    degree = dict(G.degree())
    pagerank = nx.pagerank(G, alpha=0.85) if G.number_of_edges() > 0 else {}
    clustering = nx.clustering(G)
    components = {n: len(c) for c in nx.connected_components(G) for n in c}

    rows = []
    for node in G.nodes():
        rows.append({
            "policyholder_id": node,
            "graph_degree": degree.get(node, 0),
            "graph_pagerank": pagerank.get(node, 0.0),
            "graph_clustering_coef": clustering.get(node, 0.0),
            "graph_component_size": components.get(node, 1),
        })
    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(OUT, index=False)
    print(f"Graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
    print(f"Saved graph features for {len(feat_df)} policyholders -> {OUT}")
    print(feat_df.sort_values('graph_pagerank', ascending=False).head(5))


if __name__ == "__main__":
    main()
