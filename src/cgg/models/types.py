from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GraphBatch:
    """
    Minimal batch representation for code graphs.

    node_features: (batch, nodes, feat_dim)
    adjacency: (batch, nodes, nodes) binary or weighted adjacency
    mask: (batch, nodes) boolean mask indicating which nodes are real
    """

    node_features: torch.Tensor
    adjacency: torch.Tensor
    mask: torch.Tensor

    def to(self, device: torch.device | str) -> "GraphBatch":
        return GraphBatch(
            node_features=self.node_features.to(device),
            adjacency=self.adjacency.to(device),
            mask=self.mask.to(device),
        )

    @property
    def num_nodes(self) -> int:
        return int(self.mask.sum(dim=1).max().item())


def compute_edge_loss(
    edge_logits: torch.Tensor,
    edge_targets: torch.Tensor,
    mask: torch.Tensor,
    clamp_range: tuple[float, float] = (-10.0, 10.0),
    neg_edge_penalty: float = 5.0,
) -> torch.Tensor:
    """
    Compute binary cross-entropy loss for edge prediction with class balancing.

    This is a shared utility used by all graph generation models to ensure
    consistent edge loss computation.

    Args:
        edge_logits: (batch, nodes, nodes) predicted edge logits
        edge_targets: (batch, nodes, nodes) ground-truth binary adjacency
        mask: (batch, nodes) boolean mask indicating valid nodes
        clamp_range: (min, max) to clamp logits and prevent NaN from -inf values
        neg_edge_penalty: Extra penalty multiplier for predicting edges where
                          there shouldn't be any (false positives). Higher values
                          encourage sparser graphs.

    Returns:
        Scalar edge loss tensor (0.0 if no valid edges)
    """
    device = edge_logits.device
    mask = mask.bool()
    edge_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)

    if not edge_mask.any():
        return torch.tensor(0.0, device=device)

    # Clamp logits to prevent -inf from causing NaN in BCE
    edge_logits_clamped = torch.clamp(
        edge_logits, min=clamp_range[0], max=clamp_range[1]
    )

    # Extract valid edge predictions and targets
    pred = edge_logits_clamped[edge_mask]
    target = edge_targets[edge_mask]

    # Compute class weights to handle imbalance (edges are typically sparse)
    pos = float(target.sum().item())
    tot = float(target.numel())
    neg = max(0.0, tot - pos)

    # Base BCE loss with positive weighting
    if pos > 0.0:
        # Weight positive examples more heavily since edges are sparse
        pos_weight = torch.tensor(neg / pos, device=device, dtype=torch.float32)
        base_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    else:
        # No positive examples in batch -> use unweighted BCE
        base_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    per_edge_loss = base_loss_fn(pred, target)

    # Apply extra penalty for false positives (predicting edge where target=0)
    # This encourages sparser graphs
    if neg_edge_penalty > 1.0:
        # False positive: sigmoid(pred) > 0.5 but target = 0
        pred_probs = torch.sigmoid(pred)
        is_neg_target = target < 0.5
        is_pos_pred = pred_probs > 0.5
        false_positives = is_neg_target & is_pos_pred

        # Apply penalty
        penalty_weight = torch.ones_like(per_edge_loss)
        penalty_weight[false_positives] = neg_edge_penalty
        per_edge_loss = per_edge_loss * penalty_weight

    return per_edge_loss.mean()


def compute_edge_metrics(
    edge_logits: torch.Tensor,
    edge_targets: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute precision, recall, and F1 for edge prediction.

    Args:
        edge_logits: (batch, nodes, nodes) predicted edge logits
        edge_targets: (batch, nodes, nodes) ground-truth binary adjacency
        mask: (batch, nodes) boolean mask indicating valid nodes
        threshold: probability threshold for predicting an edge

    Returns:
        Dictionary with 'precision', 'recall', 'f1' (all in [0, 1])
    """
    mask = mask.bool()
    edge_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)

    if not edge_mask.any():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Get predictions and targets for valid edges only
    pred_probs = torch.sigmoid(edge_logits[edge_mask])
    pred_edges = (pred_probs > threshold).float()
    true_edges = edge_targets[edge_mask].float()

    # True positives, false positives, false negatives
    tp = (pred_edges * true_edges).sum().item()
    fp = (pred_edges * (1 - true_edges)).sum().item()
    fn = ((1 - pred_edges) * true_edges).sum().item()

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}
