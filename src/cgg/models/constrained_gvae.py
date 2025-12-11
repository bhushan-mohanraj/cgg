import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import GraphBatch


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Simple GCN: A_hat X W
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        norm_adj = adj / deg
        return self.linear(norm_adj @ x)


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

        self.enc1 = GCNLayer(input_dim, hidden_dim)
        self.enc2 = GCNLayer(hidden_dim, hidden_dim)
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
        h = F.relu(self.enc1(x, adj))
        h = F.relu(self.enc2(h, adj))
        h = h * mask.unsqueeze(-1)
        pooled = h.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return self.enc_mean(pooled), self.enc_logvar(pooled)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(
        self, z: torch.Tensor, batch: GraphBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_nodes = batch.mask.shape[1]

        # Node reconstruction
        node_logits = self.dec_node(z).unsqueeze(1).expand(
            -1, num_nodes, -1
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
        node_mask = batch.mask.bool()
        node_loss = F.mse_loss(
            node_logits[node_mask], batch.node_features[node_mask]
        )

        edge_targets = batch.adjacency
        edge_mask = (
            batch.mask.unsqueeze(-1).bool() & batch.mask.unsqueeze(-2).bool()
        )
        edge_loss = F.binary_cross_entropy_with_logits(
            edge_logits[edge_mask], edge_targets[edge_mask], reduction="mean"
        )
        kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
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

