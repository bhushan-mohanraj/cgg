"""
Shared Graph Encoder using PyTorch Geometric.

This module provides a unified GCN-based encoder that all generator models use.
The encoder takes a graph (node features + adjacency) and produces a latent
embedding that generators can use for autoregressive generation.

Architecture:
    Input Graph → GCNConv layers → Global Mean Pooling → Latent z

All models (VAE, Diffusion, Autoregressive) share this encoder so that they
operate in the same latent space. This enables:
1. Train encoder + VAE generator together first
2. Freeze encoder, train other generators on the same latent space
3. Unified alignment from text embeddings → graph latent space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import dense_to_sparse

from .types import GraphBatch


class SharedGraphEncoder(nn.Module):
    """
    Shared GCN-based encoder using PyTorch Geometric.

    This encoder is used by all generator models during training.
    At inference time, it can be replaced by the alignment adapter
    that maps text embeddings to the same latent space.

    Uses:
        - GCNConv for message passing
        - BatchNorm for stable training
        - global_mean_pool for graph-level readout
        - Optional variational mode for VAE training
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 3,
        variational: bool = False,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Vocabulary size (dimension of one-hot node features)
            hidden_dim: Hidden dimension for GCN layers
            latent_dim: Output embedding dimension
            num_layers: Number of GCN layers
            variational: If True, output mean and logvar for VAE-style training
            dropout: Dropout probability between layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.variational = variational
        self.dropout = dropout

        # GCN layers using PyG
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Batch normalization for stable training
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        )

        # Output projection
        if variational:
            self.mean_proj = nn.Linear(hidden_dim, latent_dim)
            self.logvar_proj = nn.Linear(hidden_dim, latent_dim)
        else:
            self.out_proj = nn.Sequential(
                nn.Linear(hidden_dim, latent_dim),
                nn.Tanh(),
            )

    def _dense_to_pyg(self, batch: GraphBatch):
        """
        Convert dense batch format to PyG sparse format.

        Args:
            batch: GraphBatch with node_features (B, N, D), adjacency (B, N, N), mask (B, N)

        Returns:
            x: Node features (total_nodes, D)
            edge_index: Edge indices (2, total_edges)
            batch_idx: Batch assignment (total_nodes,)
        """
        B, N, D = batch.node_features.shape
        device = batch.node_features.device

        x_list = []
        edge_index_list = []
        batch_idx_list = []
        node_offset = 0

        for b in range(B):
            mask = batch.mask[b].bool()
            num_nodes = mask.sum().item()

            if num_nodes == 0:
                continue

            # Get valid node features
            x_list.append(batch.node_features[b, mask])

            # Get edges for this graph (only within valid nodes)
            adj = batch.adjacency[b, :num_nodes, :num_nodes]
            ei, _ = dense_to_sparse(adj)

            # Offset edge indices
            if ei.numel() > 0:
                edge_index_list.append(ei + node_offset)

            # Batch assignment
            batch_idx_list.append(
                torch.full((num_nodes,), b, dtype=torch.long, device=device)
            )

            node_offset += num_nodes

        if not x_list:
            # Empty batch fallback
            return (
                torch.zeros(1, D, device=device),
                torch.zeros(2, 0, dtype=torch.long, device=device),
                torch.zeros(1, dtype=torch.long, device=device),
            )

        x = torch.cat(x_list, dim=0)
        batch_idx = torch.cat(batch_idx_list, dim=0)

        if edge_index_list:
            edge_index = torch.cat(edge_index_list, dim=1)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)

        return x, edge_index, batch_idx

    def forward(self, batch: GraphBatch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode a batch of graphs to latent embeddings.

        Args:
            batch: GraphBatch with node_features, adjacency, mask

        Returns:
            If variational: (mean, logvar) each of shape (B, latent_dim)
            If not variational: (z, None) where z is (B, latent_dim)
        """
        B = batch.node_features.shape[0]
        device = batch.node_features.device

        # Convert to PyG sparse format
        x, edge_index, batch_idx = self._dense_to_pyg(batch)

        # Apply GCN layers with residual connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)

            # Residual connection (skip first layer due to dim change)
            if i > 0:
                x_new = x_new + x

            x = F.dropout(x_new, p=self.dropout, training=self.training)

        # Global pooling to get graph-level embeddings
        pooled = global_mean_pool(x, batch_idx)  # (num_graphs, hidden_dim)

        # Ensure we have embeddings for all batch elements
        num_graphs = batch_idx.max().item() + 1 if batch_idx.numel() > 0 else 0
        if num_graphs < B:
            full_pooled = torch.zeros(B, self.hidden_dim, device=device)
            for b in range(num_graphs):
                full_pooled[b] = pooled[b]
            pooled = full_pooled

        # Project to latent space
        if self.variational:
            mean = self.mean_proj(pooled)
            logvar = self.logvar_proj(pooled)
            logvar = torch.clamp(logvar, min=-20, max=20)
            return mean, logvar
        else:
            z = self.out_proj(pooled)
            return z, None

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
