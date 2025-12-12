import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import GraphBatch, compute_edge_loss


class DiffusionBackbone(nn.Module):
    def __init__(self, dim: int, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        # Use batch_first=True so the encoder expects (batch, seq, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: expected (batch, nodes, dim)
        assert x.dim() == 3, f"DiffusionBackbone expected 3D tensor, got {x.dim()}-D"
        out = self.encoder(x)
        return out


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
        # Ensure keep_mask has the same (batch, nodes) shape as x_start to avoid
        # adding extra singleton dimensions during broadcasting. `t` is (batch,).
        x_start.shape[0]
        n_nodes = x_start.shape[1]
        keep_prob = self.alphas_cumprod[t].view(-1, 1)  # (batch, 1)
        # Expand to per-node probabilities and sample a mask of shape (batch, nodes)
        keep_mask = torch.bernoulli(keep_prob.expand(-1, n_nodes)).bool()
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
        node_logits = node_logits.masked_fill(
            ~batch.mask.unsqueeze(-1).bool(), float("-inf")
        )

        # Edge prediction conditioned on node context
        node_pair = node_ctx.unsqueeze(2) + node_ctx.unsqueeze(1)
        edge_logits = self.edge_head(node_pair).squeeze(-1)
        edge_logits = edge_logits.masked_fill(
            ~(batch.mask.unsqueeze(-1) & batch.mask.unsqueeze(-2)),
            float("-inf"),
        )

        return {
            "node_logits": node_logits,
            "edge_logits": edge_logits,
            "x_noisy": x_noisy,
        }

    def p_losses(self, batch: GraphBatch, t: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.forward(batch, t)
        target_tokens = batch.node_features.argmax(dim=-1)
        node_loss = F.cross_entropy(
            outputs["node_logits"][batch.mask.bool()],
            target_tokens[batch.mask.bool()],
            reduction="mean",
        )

        # Use shared edge loss computation
        edge_loss = compute_edge_loss(
            outputs["edge_logits"], batch.adjacency, batch.mask
        )

        return {
            "loss": node_loss + edge_loss,
            "node_loss": node_loss,
            "edge_loss": edge_loss,
        }
