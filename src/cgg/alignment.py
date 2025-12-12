"""
Alignment utilities between LLM text embeddings and model-specific inputs.

Workflow:
1) Fetch an embedding from OpenAI for a user-provided algorithm description.
2) Map that embedding into seeds/priors for downstream graph generators.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "openai package is required. Install with `uv add openai`."
    ) from exc


DEFAULT_EMBED_MODEL = "text-embedding-3-small"


def fetch_openai_embedding(
    text: str,
    model: str = DEFAULT_EMBED_MODEL,
    client: Optional[OpenAI] = None,
    api_key: Optional[str] = None,
) -> torch.Tensor:
    """
    Fetch an embedding vector from OpenAI and return it as a torch tensor.
    """
    client = client or OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(input=text, model=model)
    vector = response.data[0].embedding
    return torch.tensor(vector, dtype=torch.float32)


@dataclass
class AlignmentConfig:
    embedding_dim: int
    cgvae_latent_dim: int
    hier_num_segments: int
    token_vocab: int


class AlignmentAdapter(nn.Module):
    """
    Simple MLP adapters that map a text embedding into seeds/priors for each model.
    """

    def __init__(self, cfg: AlignmentConfig):
        super().__init__()
        self.cfg = cfg
        self.to_cgvae_latent = nn.Sequential(
            nn.Linear(cfg.embedding_dim, cfg.cgvae_latent_dim),
            nn.ReLU(),
            nn.Linear(cfg.cgvae_latent_dim, cfg.cgvae_latent_dim),
        )
        self.to_segments = nn.Linear(cfg.embedding_dim, cfg.hier_num_segments)
        self.to_tokens = nn.Linear(cfg.embedding_dim, cfg.token_vocab)

    def forward(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        embedding: (embedding_dim,) or (batch, embedding_dim)
        Returns a dict with keys for each downstream model.
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        cgvae_latent = self.to_cgvae_latent(embedding)  # (batch, latent_dim)
        segment_logits = self.to_segments(embedding)  # (batch, num_segments)
        token_logits = self.to_tokens(embedding)  # (batch, vocab)

        return {
            "cgvae": {"latent": cgvae_latent},
            "hierarchical": {
                "segment_logits": segment_logits,
                "token_logits": token_logits,
            },
            "diffusion": {"token_logits": token_logits},
        }


def embed_and_align(
    text: str,
    adapter: AlignmentAdapter,
    embed_model: str = DEFAULT_EMBED_MODEL,
    device: str | torch.device = "cpu",
    client: Optional[OpenAI] = None,
) -> Dict[str, torch.Tensor]:
    """
    Convenience helper: fetch OpenAI embedding, map to model-specific seeds.
    """
    with torch.no_grad():
        emb = fetch_openai_embedding(text=text, model=embed_model, client=client).to(
            device
        )
        aligned = adapter(emb)
        return {
            k: {kk: vv.to(device) for kk, vv in v.items()} for k, v in aligned.items()
        }
