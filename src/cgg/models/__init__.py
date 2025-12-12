"""
Model implementations for code graph generation.

This package provides:
1. Shared encoder for creating graph embeddings (uses PyG GCNConv)
2. Multiple generator architectures that all work from embeddings
3. Original models (for backwards compatibility, prefixed with old_)

All models use PyTorch Geometric (PyG) for graph neural network layers.
"""

# Shared encoder (used by all generators)
from .shared_encoder import SharedGraphEncoder

# New autoregressive generators (work from embeddings)
from .vae_generator import VAEGenerator
from .diffusion_generator import DiffusionGenerator
from .autoregressive_generator import AutoregressiveGraphGenerator

# Original models (for backwards compatibility)
from .old_constrained_gvae import ConstrainedGraphVAE
from .old_hierarchical_autoregressive import HierarchicalAutoregressiveModel
from .old_discrete_diffusion import DiscreteDiffusionModel

# Types and utilities
from .types import GraphBatch, compute_edge_loss

__all__ = [
    # Shared encoder
    "SharedGraphEncoder",
    # New generators
    "VAEGenerator",
    "DiffusionGenerator",
    "AutoregressiveGraphGenerator",
    # Original models
    "ConstrainedGraphVAE",
    "HierarchicalAutoregressiveModel",
    "DiscreteDiffusionModel",
    # Types
    "GraphBatch",
    "compute_edge_loss",
]
