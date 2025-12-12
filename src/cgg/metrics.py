"""
Graph evaluation metrics for generated vs. reference code graphs.

Metrics:
- correctness: 1 if code is functionally correct else 0 (caller decides).
- node label accuracy
- edge precision, recall, F1
- tree edit distance (TED) using zss; expects tree-shaped graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Tuple

import networkx as nx
from zss import Node, simple_distance


@dataclass
class GraphEvalResult:
    correctness: float
    node_label_accuracy: float
    edge_precision: float
    edge_recall: float
    edge_f1: float
    tree_edit_distance: float | None

    def as_dict(self) -> Dict[str, float | None]:
        return {
            "correctness": self.correctness,
            "node_label_accuracy": self.node_label_accuracy,
            "edge_precision": self.edge_precision,
            "edge_recall": self.edge_recall,
            "edge_f1": self.edge_f1,
            "tree_edit_distance": self.tree_edit_distance,
        }


def _edge_set(graph: nx.Graph) -> set[Tuple[Hashable, Hashable]]:
    if graph.is_directed():
        return {(u, v) for u, v in graph.edges()}
    return {tuple(sorted((u, v))) for u, v in graph.edges()}


def node_label_accuracy(
    reference: nx.Graph,
    generated: nx.Graph,
    node_label_attr: str = "label",
) -> float:
    ref_nodes = set(reference.nodes())
    gen_nodes = set(generated.nodes())
    matched = ref_nodes & gen_nodes
    if not ref_nodes:
        return 1.0 if not gen_nodes else 0.0
    correct = 0
    for node in matched:
        ref_label = reference.nodes[node].get(node_label_attr)
        gen_label = generated.nodes[node].get(node_label_attr)
        correct += 1 if ref_label == gen_label else 0
    # penalize missing nodes by normalizing over reference size
    return correct / len(ref_nodes)


def edge_prf(
    reference: nx.Graph,
    generated: nx.Graph,
) -> Tuple[float, float, float]:
    ref_edges = _edge_set(reference)
    gen_edges = _edge_set(generated)

    if not ref_edges and not gen_edges:
        return 1.0, 1.0, 1.0

    tp = len(ref_edges & gen_edges)
    precision = tp / len(gen_edges) if gen_edges else 0.0
    recall = tp / len(ref_edges) if ref_edges else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )
    return precision, recall, f1


def _to_zss_tree(graph: nx.DiGraph, node_label_attr: str) -> Node:
    # Identify root: node with in-degree 0
    roots = [n for n, deg in graph.in_degree() if deg == 0]
    if len(roots) != 1:
        raise ValueError(
            "Graph must be a rooted tree (exactly one node with in-degree 0)."
        )
    root = roots[0]

    def build(node: Hashable) -> Node:
        label = graph.nodes[node].get(node_label_attr, str(node))
        zss_children = [build(child) for child in graph.successors(node)]
        return Node(label, children=zss_children)

    return build(root)


def tree_edit_distance(
    reference: nx.DiGraph,
    generated: nx.DiGraph,
    node_label_attr: str = "label",
) -> float:
    ref_tree = _to_zss_tree(reference, node_label_attr)
    gen_tree = _to_zss_tree(generated, node_label_attr)
    return simple_distance(
        ref_tree,
        gen_tree,
        label_dist=lambda a, b: 0 if a == b else 1,
    )


def evaluate_graphs(
    reference: nx.Graph,
    generated: nx.Graph,
    *,
    code_correct: bool | None = None,
    node_label_attr: str = "label",
    ted_if_tree: bool = True,
) -> GraphEvalResult:
    correctness = 1.0 if code_correct else 0.0 if code_correct is not None else 0.0

    node_acc = node_label_accuracy(
        reference, generated, node_label_attr=node_label_attr
    )
    edge_p, edge_r, edge_f1 = edge_prf(reference, generated)

    ted: float | None = None
    if ted_if_tree:
        try:
            # Use directed views to respect parent->child
            ref_d = reference if reference.is_directed() else reference.to_directed()
            gen_d = generated if generated.is_directed() else generated.to_directed()
            ted = tree_edit_distance(ref_d, gen_d, node_label_attr=node_label_attr)
        except Exception:
            ted = None

    return GraphEvalResult(
        correctness=correctness,
        node_label_accuracy=node_acc,
        edge_precision=edge_p,
        edge_recall=edge_r,
        edge_f1=edge_f1,
        tree_edit_distance=ted,
    )
