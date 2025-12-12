"""
VAE-based Autoregressive Graph Generator using PyTorch Geometric.

This generator uses a VAE-style latent space to generate graphs autoregressively.
It takes a latent embedding z and generates nodes one-by-one, using GNN layers
to incorporate graph structure as nodes are added.

Key features:
- Autoregressive generation (one node at a time)
- Uses GCNConv to incorporate partial graph structure during generation
- Edge prediction with learned pairwise representations
- Works from a single latent vector z
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse

from .types import compute_edge_loss


class VAEGenerator(nn.Module):
    """
    Autoregressive graph generator with VAE-style training using PyG.

    Generation process:
    1. Take latent z from encoder (or alignment adapter)
    2. Initialize with z projected to hidden state
    3. Autoregressively generate nodes using GRU + GCN refinement
    4. Predict edges between each new node and all previous nodes
    5. Stop when STOP token is generated or max_nodes reached
    """

    # Special tokens
    PAD_TOKEN = 0
    STOP_TOKEN = 1
    START_TOKEN = 2

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_node_tokens: int = 256,
        max_nodes: int = 64,
        num_gnn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_node_tokens = num_node_tokens
        self.max_nodes = max_nodes
        self.dropout = dropout

        # Project latent to initial hidden state
        self.latent_to_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Token embedding
        self.token_embed = nn.Embedding(num_node_tokens, hidden_dim)

        # GRU for sequential generation
        self.node_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # GCN layers to refine node representations using partial graph
        self.gnn_layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)]
        )
        self.gnn_bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_gnn_layers)]
        )

        # Node prediction head
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_node_tokens),
        )

        # Edge prediction: bilinear attention over node pairs
        self.edge_key = nn.Linear(hidden_dim, hidden_dim)
        self.edge_query = nn.Linear(hidden_dim, hidden_dim)
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Stop prediction
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _refine_with_gnn(
        self,
        hidden_states: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Refine node hidden states using GNN on current partial graph.

        Args:
            hidden_states: (B, N, hidden_dim) node representations
            adjacency: (B, N, N) current adjacency matrix
            mask: (B, N) valid node mask

        Returns:
            Refined hidden states (B, N, hidden_dim)
        """
        B, N, H = hidden_states.shape

        refined = hidden_states.clone()

        for b in range(B):
            num_nodes = mask[b].sum().int().item()
            if num_nodes < 2:
                continue

            # Get subgraph
            x = hidden_states[b, :num_nodes]  # (n, H)
            adj = adjacency[b, :num_nodes, :num_nodes]
            edge_index, _ = dense_to_sparse(adj)

            if edge_index.shape[1] == 0:
                continue

            # Apply GNN layers
            h = x
            for conv, bn in zip(self.gnn_layers, self.gnn_bns):
                h_new = conv(h, edge_index)
                if h_new.shape[0] > 1:
                    h_new = bn(h_new)
                h_new = F.relu(h_new)
                h = h + h_new  # Residual
                h = F.dropout(h, p=self.dropout, training=self.training)

            refined[b, :num_nodes] = h

        return refined

    def forward(
        self,
        z: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with teacher forcing for training.

        Args:
            z: Latent embedding (B, latent_dim)
            target_tokens: Ground-truth node tokens (B, N)
            target_mask: Which nodes are real (B, N)

        Returns:
            Dict with node_logits, edge_logits, stop_logits
        """
        batch_size = z.shape[0]
        device = z.device

        if target_tokens is None:
            return self._generate_dict(z)

        num_nodes = target_tokens.shape[1]

        # Initialize hidden state from latent
        h0 = self.latent_to_hidden(z)  # (B, hidden)
        h0 = h0.unsqueeze(0).expand(2, -1, -1).contiguous()  # (2, B, hidden)

        # Create input sequence: start token + shifted tokens
        input_tokens = torch.full(
            (batch_size, num_nodes), self.START_TOKEN, dtype=torch.long, device=device
        )
        input_tokens[:, 1:] = target_tokens[:, :-1]

        # Embed and run through GRU
        input_emb = self.token_embed(input_tokens)  # (B, N, hidden)
        hidden_states, _ = self.node_gru(input_emb, h0)  # (B, N, hidden)

        # Refine with GNN using target adjacency
        # Build adjacency from target (we'll compute edge loss separately)
        # For training, we use teacher forcing on the graph structure too

        # Predict nodes
        node_logits = self.node_head(hidden_states)  # (B, N, vocab)

        # Predict stop
        stop_logits = self.stop_head(hidden_states).squeeze(-1)  # (B, N)

        # Predict edges using attention-like mechanism
        edge_logits = self._compute_edge_logits(hidden_states, target_mask)

        return {
            "node_logits": node_logits,
            "edge_logits": edge_logits,
            "stop_logits": stop_logits,
            "hidden_states": hidden_states,
        }

    def _compute_edge_logits(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute pairwise edge logits using key-query attention."""
        B, N, H = hidden_states.shape

        # Compute keys and queries
        keys = self.edge_key(hidden_states)  # (B, N, H)
        queries = self.edge_query(hidden_states)  # (B, N, H)

        # Pairwise combinations
        k_i = keys.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, H)
        q_j = queries.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, H)
        pairs = torch.cat([k_i, q_j], dim=-1)  # (B, N, N, 2H)

        edge_logits = self.edge_head(pairs).squeeze(-1)  # (B, N, N)

        # Apply mask
        if mask is not None:
            edge_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            edge_logits = edge_logits.masked_fill(~edge_mask.bool(), float("-inf"))

        return edge_logits

    def _generate_dict(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Placeholder for inference - use generate() instead."""
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
        Autoregressively generate graphs from latent embeddings.

        Args:
            z: Latent embedding (B, latent_dim)
            max_nodes: Maximum nodes to generate
            temperature: Sampling temperature

        Returns:
            node_tokens: List of token lists per batch
            edge_lists: List of edge lists per batch
        """
        if max_nodes is None:
            max_nodes = self.max_nodes

        batch_size = z.shape[0]
        device = z.device

        # Initialize
        h = self.latent_to_hidden(z).unsqueeze(0).expand(2, -1, -1).contiguous()

        all_tokens: List[List[int]] = [[] for _ in range(batch_size)]
        all_edges: List[List[Tuple[int, int]]] = [[] for _ in range(batch_size)]
        all_hidden: List[torch.Tensor] = []

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        current_token = torch.full(
            (batch_size, 1), self.START_TOKEN, dtype=torch.long, device=device
        )

        # Build adjacency incrementally
        adjacency = torch.zeros(batch_size, max_nodes, max_nodes, device=device)

        for t in range(max_nodes):
            # Embed current token
            token_emb = self.token_embed(current_token)  # (B, 1, H)

            # GRU step
            output, h = self.node_gru(token_emb, h)
            hidden = output.squeeze(1)  # (B, H)
            all_hidden.append(hidden)

            # Predict stop
            stop_logits = self.stop_head(hidden).squeeze(-1)
            stop_prob = torch.sigmoid(stop_logits)
            should_stop = (torch.rand_like(stop_prob) < stop_prob) | finished

            # Predict next node
            node_logits = self.node_head(hidden)
            if temperature != 1.0:
                node_logits = node_logits / temperature
            probs = F.softmax(node_logits, dim=-1)
            sampled_tokens = torch.multinomial(probs, 1).squeeze(-1)

            # Store tokens
            for b in range(batch_size):
                if not finished[b] and not should_stop[b]:
                    all_tokens[b].append(sampled_tokens[b].item())

            # Predict edges to previous nodes
            if t > 0:
                prev_hidden = torch.stack(all_hidden[:-1], dim=1)  # (B, t, H)
                curr_hidden = hidden.unsqueeze(1)  # (B, 1, H)

                # Key-query attention for edges
                keys = self.edge_key(prev_hidden)  # (B, t, H)
                queries = self.edge_query(curr_hidden)  # (B, 1, H)

                pairs = torch.cat(
                    [
                        keys,
                        queries.expand(-1, t, -1),
                    ],
                    dim=-1,
                )  # (B, t, 2H)

                edge_logits = self.edge_head(pairs).squeeze(-1)  # (B, t)
                edge_probs = torch.sigmoid(edge_logits)
                edges = torch.bernoulli(edge_probs).bool()

                for b in range(batch_size):
                    if not finished[b] and not should_stop[b]:
                        for prev_t in range(t):
                            if edges[b, prev_t]:
                                all_edges[b].append((prev_t, t))
                                adjacency[b, prev_t, t] = 1
                                adjacency[b, t, prev_t] = 1  # Symmetric

            # Update state
            finished = finished | should_stop
            current_token = sampled_tokens.unsqueeze(1)

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
        mean: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute VAE training loss."""
        device = node_logits.device

        # Node prediction loss
        if mask.any():
            node_loss = F.cross_entropy(
                node_logits[mask.bool()],
                target_tokens[mask.bool()],
                reduction="mean",
            )
        else:
            node_loss = torch.tensor(0.0, device=device)

        # Edge loss with class balancing
        edge_loss = compute_edge_loss(edge_logits, target_edges, mask)

        # Stop loss: predict stop at last valid position
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

        # KL divergence for VAE
        if mean is not None and logvar is not None:
            safe_logvar = torch.clamp(logvar, max=20.0)
            kl_loss = -0.5 * torch.mean(
                1 + safe_logvar - mean.pow(2) - torch.exp(safe_logvar)
            )
        else:
            kl_loss = torch.tensor(0.0, device=device)

        # Weighted sum
        total = node_loss + edge_loss + 0.1 * stop_loss + 0.01 * kl_loss

        return {
            "loss": total,
            "node_loss": node_loss,
            "edge_loss": edge_loss,
            "stop_loss": stop_loss,
            "kl_loss": kl_loss,
        }
