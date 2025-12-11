import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import GraphBatch


class DiffusionBackbone(nn.Module):
    def __init__(self, dim: int, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 2
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, nodes, dim)
        seq = x.transpose(0, 1)
        out = self.encoder(seq)
        return out.transpose(0, 1)


class DiscreteDiffusionModel(nn.Module):
    """
    Discrete diffusion for code graphs.

    - Diffuses categorical node tokens and binary edges.
    - Uses a transformer backbone to predict clean tokens at step t.
    """

    def __init__(
        self,
        input_dim: int,
        num_node_tokens: int = 256,
        num_steps: int = 1000,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.num_node_tokens = num_node_tokens

        self.node_embed = nn.Embedding(num_node_tokens, hidden_dim)
        self.time_embed = nn.Embedding(num_steps, hidden_dim)
        self.backbone = DiffusionBackbone(hidden_dim)
        self.node_head = nn.Linear(hidden_dim, num_node_tokens)
        self.edge_head = nn.Linear(hidden_dim, 1)

        # Precompute linear beta schedule
        beta_start, beta_end = 1e-4, 0.02
        self.register_buffer(
            "betas", torch.linspace(beta_start, beta_end, steps=num_steps)
        )
        alphas = 1.0 - self.betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Add discrete noise by random replacement.
        """
        noise = torch.randint(
            0, self.num_node_tokens, x_start.shape, device=x_start.device
        )
        keep_prob = self.alphas_cumprod[t].view(-1, 1, 1)
        keep_mask = torch.bernoulli(keep_prob).bool()
        return torch.where(keep_mask, x_start, noise)

    def forward(self, batch: GraphBatch, t: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Predict clean nodes/edges from noisy observations at timestep t.
        t: (batch,) timestep indices
        """
        node_tokens = batch.node_features.argmax(dim=-1)  # assume one-hot inputs
        x_noisy = self.q_sample(node_tokens, t)

        time_emb = self.time_embed(t).unsqueeze(1)  # (batch, 1, dim)
        node_emb = self.node_embed(x_noisy) + time_emb
        node_ctx = self.backbone(node_emb)
        node_logits = self.node_head(node_ctx)
        node_logits = node_logits.masked_fill(~batch.mask.unsqueeze(-1).bool(), float("-inf"))

        # Edge prediction conditioned on node context
        node_pair = node_ctx.unsqueeze(2) + node_ctx.unsqueeze(1)
        edge_logits = self.edge_head(node_pair).squeeze(-1)
        edge_logits = edge_logits.masked_fill(
            ~(batch.mask.unsqueeze(-1) & batch.mask.unsqueeze(-2)),
            float("-inf"),
        )

        return {"node_logits": node_logits, "edge_logits": edge_logits, "x_noisy": x_noisy}

    def p_losses(
        self, batch: GraphBatch, t: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        outputs = self.forward(batch, t)
        target_tokens = batch.node_features.argmax(dim=-1)
        node_loss = F.cross_entropy(
            outputs["node_logits"][batch.mask.bool()],
            target_tokens[batch.mask.bool()],
            reduction="mean",
        )

        edge_targets = batch.adjacency
        edge_mask = batch.mask.unsqueeze(-1) & batch.mask.unsqueeze(-2)
        edge_loss = F.binary_cross_entropy_with_logits(
            outputs["edge_logits"][edge_mask], edge_targets[edge_mask], reduction="mean"
        )

        return {"loss": node_loss + edge_loss, "node_loss": node_loss, "edge_loss": edge_loss}

