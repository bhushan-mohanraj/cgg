import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import GraphBatch


class HierarchicalAutoregressiveModel(nn.Module):
    """
    Two-level autoregressive generator for code graphs.

    - Planner produces coarse segment codes for each node.
    - Decoder conditions on segment codes to autoregressively generate node tokens.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_segments: int = 8,
        num_node_tokens: int = 256,
        num_layers: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        self.num_segments = num_segments
        self.num_node_tokens = num_node_tokens

        self.planner = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_segments),
        )

        self.token_embed = nn.Embedding(num_node_tokens, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(hidden_dim, num_node_tokens)

    def plan_segments(self, batch: GraphBatch) -> torch.Tensor:
        """
        Returns logits over segments for each node. Shape: (batch, nodes, num_segments)
        """
        logits = self.planner(batch.node_features)
        logits = logits.masked_fill(~batch.mask.unsqueeze(-1).bool(), float("-inf"))
        return logits

    def decode_nodes(
        self,
        segment_logits: torch.Tensor,
        batch: GraphBatch,
        teacher_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Autoregressively decode node tokens conditioned on segments.
        teacher_tokens: (batch, nodes) token ids for teacher forcing.
        Returns logits: (batch, nodes, num_node_tokens)
        """
        batch_size, num_nodes, _ = segment_logits.shape
        device = segment_logits.device

        # Derive segment embeddings
        segment_ids = segment_logits.argmax(dim=-1)
        segment_emb = self.token_embed(segment_ids)

        if teacher_tokens is None:
            # Start with all zeros (BOS) and decode step-by-step (simplified)
            tokens = torch.zeros(batch_size, num_nodes, dtype=torch.long, device=device)
        else:
            tokens = teacher_tokens

        token_emb = self.token_embed(tokens) + segment_emb

        # Transformer expects (seq, batch, dim); flatten batches together per graph
        seq = token_emb.transpose(0, 1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(num_nodes).to(device)
        decoded = self.decoder(seq, seq, tgt_mask=tgt_mask)
        decoded = decoded.transpose(0, 1)  # (batch, nodes, hidden)
        logits = self.output_head(decoded)
        logits = logits.masked_fill(~batch.mask.unsqueeze(-1).bool(), float("-inf"))
        return logits

    def forward(
        self,
        batch: GraphBatch,
        teacher_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        segment_logits = self.plan_segments(batch)
        node_logits = self.decode_nodes(segment_logits, batch, teacher_tokens)
        return segment_logits, node_logits

    def loss(
        self,
        segment_logits: torch.Tensor,
        node_logits: torch.Tensor,
        target_segments: torch.Tensor,
        target_tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mask = mask.bool()
        seg_loss = F.cross_entropy(
            segment_logits[mask], target_segments[mask], reduction="mean"
        )
        tok_loss = F.cross_entropy(
            node_logits[mask], target_tokens[mask], reduction="mean"
        )
        return {"loss": seg_loss + tok_loss, "segment_loss": seg_loss, "token_loss": tok_loss}

