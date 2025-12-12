"""
Diffusion-based Graph Generator using PyTorch Geometric.

This generator uses discrete diffusion to generate graphs from latent embeddings.
It conditions the denoising process on the latent z and uses GNN layers for
graph-aware denoising.

Key features:
- Discrete diffusion on node tokens
- GNN-based denoising network
- Latent-conditioned generation
- Iterative refinement from noise to clean graph
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from torch_geometric.nn import TransformerConv
from torch_geometric.utils import dense_to_sparse

from .types import compute_edge_loss


class GNNDenoisingBlock(nn.Module):
    """
    GNN block for denoising node representations.
    Uses TransformerConv for attention-based message passing.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.conv = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Attention-based message passing
        x_new = self.conv(x, edge_index)
        x = self.norm(x + x_new)

        # FFN
        x = self.norm2(x + self.ffn(x))
        return x


class DiffusionGenerator(nn.Module):
    """
    Discrete diffusion generator using PyTorch Geometric.

    Generation process:
    1. Take latent z from encoder (or alignment adapter)
    2. Predict graph size from z
    3. Start with random/noisy node tokens
    4. Iteratively denoise conditioned on z using GNN
    5. Predict edges from final node representations
    """

    PAD_TOKEN = 0

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_node_tokens: int = 256,
        max_nodes: int = 64,
        num_steps: int = 100,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_node_tokens = num_node_tokens
        self.max_nodes = max_nodes
        self.num_steps = num_steps

        # Latent conditioning
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Embeddings
        self.token_embed = nn.Embedding(num_node_tokens, hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Embedding(num_steps, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.pos_embed = nn.Embedding(max_nodes, hidden_dim)

        # GNN denoising layers
        self.gnn_layers = nn.ModuleList(
            [
                GNNDenoisingBlock(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        # For early layers without edges, use self-attention via dense layer
        self.initial_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(2)
            ]
        )

        # Output heads
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_node_tokens),
        )

        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.size_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_nodes),
        )

        # Diffusion schedule
        beta_start, beta_end = 1e-4, 0.02
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: add noise to tokens."""
        noise = torch.randint(
            0, self.num_node_tokens, x_start.shape, device=x_start.device
        )

        # Keep probability based on schedule
        keep_prob = self.alphas_cumprod[t].view(-1, 1)
        keep_mask = torch.bernoulli(keep_prob.expand(-1, x_start.shape[1])).bool()

        return torch.where(keep_mask, x_start, noise)

    def forward(
        self,
        z: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            z: Latent embedding (B, latent_dim)
            target_tokens: Ground-truth node tokens (B, N)
            target_mask: Valid node mask (B, N)
            t: Diffusion timestep (B,)
        """
        B = z.shape[0]
        device = z.device

        if target_tokens is None:
            return self._generate_dict(z)

        N = target_tokens.shape[1]

        # Random timestep
        if t is None:
            t = torch.randint(0, self.num_steps, (B,), device=device)

        # Add noise
        noisy_tokens = self.q_sample(target_tokens, t)

        # Embed
        token_emb = self.token_embed(noisy_tokens)  # (B, N, H)
        time_emb = self.time_embed(t).unsqueeze(1)  # (B, 1, H)
        pos = torch.arange(N, device=device).unsqueeze(0).clamp(max=self.max_nodes - 1)
        pos_emb = self.pos_embed(pos)  # (1, N, H)
        latent_emb = self.latent_proj(z).unsqueeze(1)  # (B, 1, H)

        # Combine embeddings
        h = token_emb + time_emb + pos_emb + latent_emb  # (B, N, H)

        # Initial self-attention layers (no graph structure yet)
        attn_mask = None
        if target_mask is not None:
            attn_mask = ~target_mask.bool()

        for layer in self.initial_layers:
            h = layer(h, src_key_padding_mask=attn_mask)

        # GNN layers using predicted edges
        # First predict initial edges from current representations
        edge_logits_init = self._compute_edge_logits(h, target_mask)
        edge_probs = torch.sigmoid(edge_logits_init)

        # Use edges for GNN message passing (per graph in batch)
        h_refined = self._apply_gnn_layers(h, edge_probs, target_mask)

        # Final predictions
        node_logits = self.node_head(h_refined)
        edge_logits = self._compute_edge_logits(h_refined, target_mask)
        size_logits = self.size_head(self.latent_proj(z))

        return {
            "node_logits": node_logits,
            "edge_logits": edge_logits,
            "size_logits": size_logits,
            "noisy_tokens": noisy_tokens,
            "t": t,
        }

    def _compute_edge_logits(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute pairwise edge logits."""
        B, N, H = h.shape

        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)
        pairs = torch.cat([h_i, h_j], dim=-1)

        edge_logits = self.edge_head(pairs).squeeze(-1)

        if mask is not None:
            mask_bool = mask.bool()
            edge_mask = mask_bool.unsqueeze(-1) & mask_bool.unsqueeze(-2)
            edge_logits = edge_logits.masked_fill(~edge_mask, float("-inf"))

        return edge_logits

    def _apply_gnn_layers(
        self,
        h: torch.Tensor,
        edge_probs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply GNN layers using predicted edge probabilities."""
        B, N, H = h.shape
        device = h.device

        # Threshold edges for message passing
        edge_adj = (edge_probs > 0.3).float()

        # Process each graph in batch
        h_out = h.clone()

        for b in range(B):
            if mask is not None:
                num_nodes = mask[b].sum().int().item()
            else:
                num_nodes = N

            if num_nodes < 2:
                continue

            # Get edges
            adj = edge_adj[b, :num_nodes, :num_nodes]
            edge_index, _ = dense_to_sparse(adj)

            if edge_index.shape[1] == 0:
                # No edges - add self-loops
                edge_index = torch.stack(
                    [
                        torch.arange(num_nodes, device=device),
                        torch.arange(num_nodes, device=device),
                    ],
                    dim=0,
                )

            # Apply GNN
            x = h[b, :num_nodes]
            for gnn_layer in self.gnn_layers:
                x = gnn_layer(x, edge_index)

            h_out[b, :num_nodes] = x

        return h_out

    def _generate_dict(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Placeholder - use generate() for actual generation."""
        B = z.shape[0]
        device = z.device
        return {
            "node_logits": torch.zeros(
                B, self.max_nodes, self.num_node_tokens, device=device
            ),
            "edge_logits": torch.zeros(
                B, self.max_nodes, self.max_nodes, device=device
            ),
            "size_logits": torch.zeros(B, self.max_nodes, device=device),
        }

    @torch.no_grad()
    def generate(
        self,
        z: torch.Tensor,
        max_nodes: Optional[int] = None,
        num_steps: Optional[int] = None,
        temperature: float = 1.0,
    ) -> Tuple[List[List[int]], List[List[Tuple[int, int]]]]:
        """
        Generate graphs through iterative denoising.

        Args:
            z: Latent embedding (B, latent_dim)
            max_nodes: Maximum nodes
            num_steps: Denoising steps
        """
        if max_nodes is None:
            max_nodes = self.max_nodes
        if num_steps is None:
            num_steps = min(50, self.num_steps)

        B = z.shape[0]
        device = z.device

        # Predict graph size
        size_logits = self.size_head(self.latent_proj(z))
        predicted_sizes = size_logits.argmax(dim=-1).clamp(min=3, max=max_nodes)

        # Create mask
        positions = torch.arange(max_nodes, device=device).unsqueeze(0)
        mask = positions < predicted_sizes.unsqueeze(1)

        # Start with noise
        x = torch.randint(0, self.num_node_tokens, (B, max_nodes), device=device)

        # Iterative denoising
        step_size = max(1, self.num_steps // num_steps)

        for i in range(num_steps - 1, -1, -1):
            t = torch.full(
                (B,),
                min(i * step_size, self.num_steps - 1),
                dtype=torch.long,
                device=device,
            )

            # Get embeddings
            token_emb = self.token_embed(x)
            time_emb = self.time_embed(t).unsqueeze(1)
            pos_emb = self.pos_embed(positions.clamp(max=self.max_nodes - 1))
            latent_emb = self.latent_proj(z).unsqueeze(1)

            h = token_emb + time_emb + pos_emb + latent_emb

            # Self-attention
            for layer in self.initial_layers:
                h = layer(h, src_key_padding_mask=~mask)

            # Predict edges and apply GNN
            edge_probs = torch.sigmoid(self._compute_edge_logits(h, mask.float()))
            h = self._apply_gnn_layers(h, edge_probs, mask.float())

            # Predict clean tokens
            node_logits = self.node_head(h)

            # Sample with decreasing temperature
            temp = max(0.1, temperature * (i / num_steps))
            probs = F.softmax(node_logits / temp, dim=-1)
            new_tokens = torch.multinomial(probs.view(-1, self.num_node_tokens), 1)
            new_tokens = new_tokens.view(B, max_nodes)

            x = torch.where(mask, new_tokens, x)

        # Final edge prediction
        token_emb = self.token_embed(x)
        latent_emb = self.latent_proj(z).unsqueeze(1)
        h = token_emb + latent_emb

        for layer in self.initial_layers:
            h = layer(h, src_key_padding_mask=~mask)

        edge_logits = self._compute_edge_logits(h, mask.float())
        edge_probs = torch.sigmoid(edge_logits)

        # Convert to output format
        all_tokens = []
        all_edges = []

        for b in range(B):
            size = predicted_sizes[b].item()
            tokens = x[b, :size].tolist()
            all_tokens.append(tokens)

            edges = []
            for i in range(size):
                for j in range(size):
                    if i != j and edge_probs[b, i, j] > 0.5:
                        edges.append((i, j))
            all_edges.append(edges)

        return all_tokens, all_edges

    def loss(
        self,
        node_logits: torch.Tensor,
        edge_logits: torch.Tensor,
        size_logits: torch.Tensor,
        target_tokens: torch.Tensor,
        target_edges: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss."""
        device = node_logits.device

        # Node loss
        if mask.any():
            node_loss = F.cross_entropy(
                node_logits[mask.bool()],
                target_tokens[mask.bool()],
                reduction="mean",
            )
        else:
            node_loss = torch.tensor(0.0, device=device)

        # Edge loss
        edge_loss = compute_edge_loss(edge_logits, target_edges, mask)

        # Size loss
        target_sizes = mask.sum(dim=1).long().clamp(max=self.max_nodes - 1)
        size_loss = F.cross_entropy(size_logits, target_sizes, reduction="mean")

        total = node_loss + edge_loss + 0.1 * size_loss

        return {
            "loss": total,
            "node_loss": node_loss,
            "edge_loss": edge_loss,
            "size_loss": size_loss,
        }
