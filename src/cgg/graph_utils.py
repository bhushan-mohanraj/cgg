"""Utilities to convert model outputs (node_logits, edge_logits) into NetworkX graphs
for visualization and comparison.

Functions
- batch_logits_to_nx: convert batched predictions into a list of networkx.DiGraph
"""

from __future__ import annotations

from typing import List, Dict

import networkx as nx
import torch


def _build_inverse_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    inv = {v: k for k, v in vocab.items()}
    # ensure PAD/UNK exist
    inv.setdefault(0, "<PAD>")
    inv.setdefault(1, "<UNK>")
    return inv


def logits_to_nx(
    node_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    vocab: Dict[str, int],
    mask: torch.Tensor | None = None,
    edge_thresh: float = 0.5,
) -> List[nx.DiGraph]:
    """Convert batched node and edge logits to NetworkX graphs.

    Args:
        node_logits: Tensor of shape (B, N, V) where V is vocab_size (logits over node tokens)
        edge_logits: Tensor of shape (B, N, N) (logits for binary edge presence)
        vocab: dict mapping token->index
        mask: optional boolean Tensor (B, N) indicating real nodes
        edge_thresh: threshold applied to sigmoid(edge_logits) to decide edges

    Returns:
        list of `networkx.DiGraph` of length B.
    """
    inv_vocab = _build_inverse_vocab(vocab)
    device = node_logits.device

    B, N, V = node_logits.shape
    if mask is None:
        mask = torch.ones((B, N), dtype=torch.bool, device=device)
    else:
        mask = mask.bool().to(device)

    graphs: List[nx.DiGraph] = []
    # compute node token ids
    node_ids = node_logits.argmax(dim=-1)  # (B, N)
    # edge probabilities
    edge_probs = torch.sigmoid(edge_logits)

    for b in range(B):
        g = nx.DiGraph()
        for i in range(N):
            if not mask[b, i]:
                continue
            tok_idx = int(node_ids[b, i].item())
            tok = inv_vocab.get(tok_idx, "<UNK>")
            # split token into label/token heuristics: keep both
            g.add_node(i, label=tok, token=tok)

        # add edges above threshold between valid nodes
        for i in range(N):
            if not mask[b, i]:
                continue
            for j in range(N):
                if not mask[b, j]:
                    continue
                p = float(edge_probs[b, i, j].item())
                if p >= edge_thresh:
                    g.add_edge(i, j, weight=p)

        graphs.append(g)

    return graphs


def single_logits_to_nx(
    node_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    vocab: Dict[str, int],
    mask: torch.Tensor | None = None,
    edge_thresh: float = 0.5,
) -> nx.DiGraph:
    """Convenience wrapper for single graph outputs (no batch dim)."""
    if node_logits.dim() == 2:
        node_logits = node_logits.unsqueeze(0)
    if edge_logits.dim() == 2:
        edge_logits = edge_logits.unsqueeze(0)
    if mask is not None and mask.dim() == 1:
        mask = mask.unsqueeze(0)
    return logits_to_nx(
        node_logits, edge_logits, vocab, mask=mask, edge_thresh=edge_thresh
    )[0]
