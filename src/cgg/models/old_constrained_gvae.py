import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import GraphBatch, compute_edge_loss

# Require PyG for GCN layers
try:
    from torch_geometric.nn import GCNConv  # type: ignore
    from torch_geometric.utils import dense_to_sparse  # type: ignore
except Exception as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "torch_geometric is required. Install it: https://pytorch-geometric.readthedocs.io/"
    ) from exc


class ConstrainedGraphVAE(nn.Module):
    """
    Constrained graph VAE for code graphs.

    - Encoder: GCN -> latent mean/logvar
    - Decoder: generates node logits and edge logits with degree masking
    - Constraint: max node degree enforced via edge mask during decoding
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        max_degree: int = 6,
    ):
        super().__init__()
        self.max_degree = max_degree

        # Use PyG GCNConv layers for encoding.
        # Note: GCNConv expects node features shaped (num_nodes_total, in_channels)
        # and an edge_index for the (possibly batched) graph.
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.enc_mean = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.dec_node = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.dec_edge = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, batch: GraphBatch) -> tuple[torch.Tensor, torch.Tensor]:
        x, adj, mask = batch.node_features, batch.adjacency, batch.mask

        # Convert batched dense adjacency to a single flattened edge_index
        # and flatten node features to (B*N, feat) so we can run PyG convs.
        B, N, feat_dim = x.shape
        device = x.device
        x_flat = x.reshape(B * N, feat_dim)

        edge_idx_list = []
        for b in range(B):
            dense = adj[b].to(device)
            ei, _ = dense_to_sparse(dense)
            if ei.numel() == 0:
                continue
            # offset indices by b*N
            ei = ei + (b * N)
            edge_idx_list.append(ei)

        if len(edge_idx_list) == 0:
            # no edges at all -> create empty edge_index
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            edge_index = torch.cat(edge_idx_list, dim=1).to(device)

        out1 = F.relu(self.conv1(x_flat, edge_index))
        out2 = F.relu(self.conv2(out1, edge_index))

        h = out2.reshape(B, N, -1)
        h = h * mask.unsqueeze(-1)
        pooled = h.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return self.enc_mean(pooled), self.enc_logvar(pooled)

    def _dense_gcn(
        self, linear: nn.Linear, x: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        raise RuntimeError(
            "_dense_gcn removed; ConstrainedGraphVAE now requires torch_geometric GCNConv"
        )

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Clamp logvar to avoid numerical overflow in exp()
        safe_logvar = torch.clamp(logvar, max=20.0)
        std = torch.exp(0.5 * safe_logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(
        self, z: torch.Tensor, batch: GraphBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_nodes = batch.mask.shape[1]

        # Node reconstruction
        node_logits = (
            self.dec_node(z).unsqueeze(1).expand(-1, num_nodes, -1)
        )  # (batch, nodes, input_dim)

        # Edge reconstruction with degree masking
        z_nodes = z.unsqueeze(1).expand(-1, num_nodes, -1)  # (batch, nodes, latent)
        z_i = z_nodes.unsqueeze(2).expand(-1, num_nodes, num_nodes, -1)
        z_j = z_nodes.unsqueeze(1).expand(-1, num_nodes, num_nodes, -1)
        pair = torch.cat([z_i, z_j], dim=-1)
        raw_edge_logits = self.dec_edge(pair).squeeze(-1)  # (batch, nodes, nodes)

        degree_mask = self._degree_mask(batch)
        edge_logits = raw_edge_logits.masked_fill(~degree_mask, float("-inf"))
        return node_logits, edge_logits

    def forward(self, batch: GraphBatch):
        mean, logvar = self.encode(batch)
        z = self.reparameterize(mean, logvar)
        node_logits, edge_logits = self.decode(z, batch)
        return node_logits, edge_logits, mean, logvar

    def loss(
        self,
        batch: GraphBatch,
        node_logits: torch.Tensor,
        edge_logits: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Robust loss computation that avoids NaNs when masks are empty
        device = node_logits.device

        node_mask = batch.mask.bool()
        if node_mask.any():
            node_loss = F.mse_loss(
                node_logits[node_mask], batch.node_features[node_mask]
            )
        else:
            node_loss = torch.tensor(0.0, device=device)

        # Use shared edge loss computation
        edge_loss = compute_edge_loss(edge_logits, batch.adjacency, batch.mask)

        # Prevent numerical overflow from exp(logvar) by clamping logvar
        safe_logvar = torch.clamp(logvar, max=20.0)
        kl = -0.5 * torch.mean(1 + safe_logvar - mean.pow(2) - torch.exp(safe_logvar))

        total = node_loss + edge_loss + kl
        return {"loss": total, "node_loss": node_loss, "edge_loss": edge_loss, "kl": kl}

    def _degree_mask(self, batch: GraphBatch) -> torch.Tensor:
        # Prevent nodes from exceeding max_degree during decoding
        current_degree = batch.adjacency.sum(dim=-1)
        allowed = current_degree < self.max_degree
        # broadcast to edge matrix
        mask_i = allowed.unsqueeze(-1)
        mask_j = allowed.unsqueeze(-2)
        return mask_i & mask_j & batch.mask.unsqueeze(-1) & batch.mask.unsqueeze(-2)
