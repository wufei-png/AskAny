#!/usr/bin/env python3
"""Visualize LightRAG knowledge graph from GraphML file."""

import sys

import networkx as nx


def visualize_graphml(graphml_path: str, max_nodes: int = 200):
    """Load and display graph info."""
    G = nx.read_graphml(graphml_path)

    print("=== GraphML Analysis ===")
    print(f"File: {graphml_path}")
    print(f"Total nodes: {len(G.nodes())}")
    print(f"Total edges: {len(G.edges())}")

    # Node statistics
    node_attrs = set()
    for node in G.nodes(data=True):
        node_attrs.update(node[1].keys())
    print(f"Node attributes: {node_attrs}")

    # Edge statistics
    edge_attrs = set()
    for edge in G.edges(data=True):
        edge_attrs.update(edge[2].keys())
    print(f"Edge attributes: {edge_attrs}")

    # Sample nodes
    print("\n=== Sample Nodes (first 10) ===")
    for i, (node, data) in enumerate(G.nodes(data=True)):
        if i >= 10:
            break
        print(f"  {node}: {data}")

    # Sample edges
    print("\n=== Sample Edges (first 10) ===")
    for i, (src, tgt, _data) in enumerate(G.edges(data=True)):
        if i >= 10:
            break
        print(f"  {src} -> {tgt}")

    # Find fpach related nodes
    print("\n=== FPACH Related Nodes ===")
    fpach_nodes = [n for n in G.nodes() if "fpach" in n.lower()]
    print(f"Found {len(fpach_nodes)} nodes containing 'fpach'")
    for n in fpach_nodes[:10]:
        neighbors = list(G.neighbors(n))
        print(f"  {n}: {neighbors[:5]}")


if __name__ == "__main__":
    graphml_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "lightrag_data/graph_chunk_entity_relation.graphml"
    )
    max_nodes = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    visualize_graphml(graphml_file, max_nodes)
