from dataclasses import dataclass

import torch


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

