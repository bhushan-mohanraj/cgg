"""
Autoregressive Graph Generator using PyTorch Geometric.

This is a pure autoregressive generator that creates graphs node-by-node.
It uses a transformer-based architecture with GNN refinement at each step.

Key features:
- Transformer decoder for sequential node generation
- GCN layers to incorporate graph structure as it's built
- Attention-based edge prediction
- Clean, modular design
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse

from .types import compute_edge_loss


class AutoregressiveGraphGenerator(nn.Module):
    """
    Pure autoregressive graph generator using PyTorch Geometric.

    Generation process:
    1. Take latent z from encoder (or alignment adapter)
    2. Use z as memory for transformer decoder
    3. Generate nodes autoregressively
    4. After each node, predict edges to all previous nodes
    5. Use GCN to incorporate graph structure for better predictions
    """

    PAD_TOKEN = 0
    STOP_TOKEN = 1
    START_TOKEN = 2

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_node_tokens: int = 256,
        max_nodes: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_node_tokens = num_node_tokens
        self.max_nodes = max_nodes

        # Latent projection to transformer memory
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Token and position embeddings
        self.token_embed = nn.Embedding(num_node_tokens, hidden_dim)
        self.pos_embed = nn.Embedding(max_nodes + 1, hidden_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # GCN for graph-aware refinement
        self.gnn_refine = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(2)]
        )
        self.gnn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])

        # Output heads
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_node_tokens),
        )

        # Edge prediction with bilinear attention
        self.edge_query = nn.Linear(hidden_dim, hidden_dim)
        self.edge_key = nn.Linear(hidden_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Stop prediction
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        target_adjacency: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with teacher forcing.

        Args:
            z: Latent embedding (B, latent_dim)
            target_tokens: Ground-truth node tokens (B, N)
            target_adjacency: Ground-truth edges (B, N, N)
            target_mask: Valid node mask (B, N)
        """
        B = z.shape[0]
        device = z.device

        if target_tokens is None:
            return self._generate_dict(z)

        N = target_tokens.shape[1]

        # Memory from latent
        memory = self.latent_proj(z).unsqueeze(1)  # (B, 1, H)

        # Input: shifted tokens (teacher forcing)
        input_tokens = torch.zeros_like(target_tokens)
        input_tokens[:, 1:] = target_tokens[:, :-1]

        # Embed tokens + positions
        token_emb = self.token_embed(input_tokens)
        positions = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(positions)

        seq = token_emb + pos_emb  # (B, N, H)

        # Causal mask for autoregressive decoding
        causal_mask = nn.Transformer.generate_square_subsequent_mask(N, device=device)

        # Decode
        hidden = self.decoder(seq, memory, tgt_mask=causal_mask)  # (B, N, H)

        # Refine with GCN using target graph structure
        if target_adjacency is not None:
            hidden = self._refine_with_gnn(hidden, target_adjacency, target_mask)

        # Predictions
        node_logits = self.node_head(hidden)
        edge_logits = self._compute_edge_logits(hidden, target_mask)
        stop_logits = self.stop_head(hidden).squeeze(-1)

        return {
            "node_logits": node_logits,
            "edge_logits": edge_logits,
            "stop_logits": stop_logits,
            "hidden": hidden,
        }

    def _refine_with_gnn(
        self,
        hidden: torch.Tensor,
        adjacency: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Refine hidden states using GCN on graph structure."""
        B, N, H = hidden.shape

        refined = hidden.clone()

        for b in range(B):
            if mask is not None:
                num_nodes = mask[b].sum().int().item()
            else:
                num_nodes = N

            if num_nodes < 2:
                continue

            # Get subgraph
            x = hidden[b, :num_nodes]
            adj = adjacency[b, :num_nodes, :num_nodes]
            edge_index, _ = dense_to_sparse(adj)

            if edge_index.shape[1] == 0:
                continue

            # Apply GCN with residual
            h = x
            for conv, norm in zip(self.gnn_refine, self.gnn_norms):
                h_new = conv(h, edge_index)
                h_new = norm(h_new)
                h = h + F.gelu(h_new)

            refined[b, :num_nodes] = h

        return refined

    def _compute_edge_logits(
        self,
        hidden: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute edge logits using query-key attention."""
        B, N, H = hidden.shape

        queries = self.edge_query(hidden)  # (B, N, H)
        keys = self.edge_key(hidden)  # (B, N, H)

        # Pairwise
        q_i = queries.unsqueeze(2).expand(-1, -1, N, -1)
        k_j = keys.unsqueeze(1).expand(-1, N, -1, -1)
        pairs = torch.cat([q_i, k_j], dim=-1)

        edge_logits = self.edge_mlp(pairs).squeeze(-1)

        if mask is not None:
            edge_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            edge_logits = edge_logits.masked_fill(~edge_mask.bool(), float("-inf"))

        return edge_logits

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
            "stop_logits": torch.zeros(B, self.max_nodes, device=device),
        }

    @torch.no_grad()
    def generate(
        self,
        z: torch.Tensor,
        max_nodes: Optional[int] = None,
        temperature: float = 1.0,
    ) -> Tuple[List[List[int]], List[List[Tuple[int, int]]]]:
        """
        Autoregressively generate graphs.

        Args:
            z: Latent embedding (B, latent_dim)
            max_nodes: Maximum nodes to generate
            temperature: Sampling temperature
        """
        if max_nodes is None:
            max_nodes = self.max_nodes

        B = z.shape[0]
        device = z.device

        memory = self.latent_proj(z).unsqueeze(1)

        all_tokens: List[List[int]] = [[] for _ in range(B)]
        all_edges: List[List[Tuple[int, int]]] = [[] for _ in range(B)]

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        # Build sequence incrementally
        generated = torch.full(
            (B, 1), self.START_TOKEN, dtype=torch.long, device=device
        )
        adjacency = torch.zeros(B, max_nodes, max_nodes, device=device)

        for t in range(max_nodes):
            seq_len = generated.shape[1]

            # Embed current sequence
            token_emb = self.token_embed(generated)
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
            pos_emb = self.pos_embed(positions)

            seq = token_emb + pos_emb

            # Decode
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=device
            )
            hidden = self.decoder(seq, memory, tgt_mask=causal_mask)

            # Get last position
            last_hidden = hidden[:, -1, :]  # (B, H)

            # Predict stop
            stop_logits = self.stop_head(last_hidden).squeeze(-1)
            should_stop = (torch.sigmoid(stop_logits) > 0.5) | finished

            # Predict next token
            node_logits = self.node_head(last_hidden)
            if temperature != 1.0:
                node_logits = node_logits / temperature
            probs = F.softmax(node_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)  # (B, 1)

            # Store tokens
            for b in range(B):
                if not finished[b] and not should_stop[b]:
                    all_tokens[b].append(next_token[b, 0].item())

            # Predict edges to previous nodes
            if t > 0:
                prev_hidden = hidden[:, :-1, :]  # (B, t, H)
                curr_hidden = last_hidden.unsqueeze(1)  # (B, 1, H)

                queries = self.edge_query(curr_hidden)  # (B, 1, H)
                keys = self.edge_key(prev_hidden)  # (B, t, H)

                pairs = torch.cat(
                    [
                        queries.expand(-1, t, -1),
                        keys,
                    ],
                    dim=-1,
                )

                edge_logits = self.edge_mlp(pairs).squeeze(-1)  # (B, t)
                edge_probs = torch.sigmoid(edge_logits)
                edges = torch.bernoulli(edge_probs).bool()

                for b in range(B):
                    if not finished[b] and not should_stop[b]:
                        for prev_t in range(t):
                            if edges[b, prev_t]:
                                all_edges[b].append((prev_t, t))
                                adjacency[b, prev_t, t] = 1

            # Update
            finished = finished | should_stop
            generated = torch.cat([generated, next_token], dim=1)

            if finished.all():
                break

        return all_tokens, all_edges

    def loss(
        self,
        node_logits: torch.Tensor,
        edge_logits: torch.Tensor,
        stop_logits: torch.Tensor,
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

        # Stop loss
        B, N = mask.shape
        node_counts = mask.sum(dim=1).long()
        stop_targets = torch.zeros_like(mask, dtype=torch.float)
        for b in range(B):
            if node_counts[b] > 0:
                stop_targets[b, node_counts[b] - 1] = 1.0

        if mask.any():
            stop_loss = F.binary_cross_entropy_with_logits(
                stop_logits[mask.bool()],
                stop_targets[mask.bool()],
                reduction="mean",
            )
        else:
            stop_loss = torch.tensor(0.0, device=device)

        total = node_loss + edge_loss + 0.1 * stop_loss

        return {
            "loss": total,
            "node_loss": node_loss,
            "edge_loss": edge_loss,
            "stop_loss": stop_loss,
        }
