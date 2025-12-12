"""Package-level training utilities for `cgg` models.

Run with: `python -m cgg.train --model vae_gen ...`

This module supports both:
1. Original models (cgvae, hier, diff) - for backwards compatibility
2. New encoder+generator models (vae_gen, diff_gen, autoreg) - shared encoder architecture
"""

from __future__ import annotations

import argparse
import pathlib
from datetime import datetime
from typing import Iterable, List

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset

from . import data as cgg_data
from .models import (
    # Original models
    ConstrainedGraphVAE,
    HierarchicalAutoregressiveModel,
    DiscreteDiffusionModel,
    # New shared encoder + generators
    SharedGraphEncoder,
    VAEGenerator,
    DiffusionGenerator,
    AutoregressiveGraphGenerator,
)
from .models.types import GraphBatch, compute_edge_metrics


def save_loss_plot(
    losses: List[float],
    model_name: str,
    out_dir: str = "results",
    y_min: float = 0.0,
    y_max: float = 10.0,
    metrics: dict = None,
):
    """Save a loss plot with datetime label, optionally including edge metrics."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = pathlib.Path(out_dir) / f"loss_{model_name}_{timestamp}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine layout based on whether we have metrics
    if metrics and any(len(v) > 0 for v in metrics.values()):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = None

    # Plot loss
    ax1.plot(range(len(losses)), losses, "b-", linewidth=2, marker="o", markersize=4)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title(f"Training Loss: {model_name}", fontsize=14)
    ax1.set_ylim(y_min, y_max)
    ax1.grid(True, alpha=0.3)

    # Plot metrics if available
    if ax2 is not None and metrics:
        colors = {"precision": "green", "recall": "blue", "f1": "red"}
        for name, values in metrics.items():
            if len(values) > 0:
                ax2.plot(
                    range(len(values)),
                    values,
                    color=colors.get(name, "gray"),
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    label=name.capitalize(),
                )
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Score", fontsize=12)
        ax2.set_title(f"Edge Metrics: {model_name}", fontsize=14)
        ax2.set_ylim(0.0, 1.0)
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss plot to {plot_path}")


class GraphFunctionDataset(Dataset):
    """Dataset of NetworkX graphs created from top-level function ASTs."""

    def __init__(
        self,
        paths: Iterable[pathlib.Path],
        max_nodes: int = 128,
        max_samples: int = None,
    ):
        self.funcs = []
        for p in paths:
            try:
                self.funcs.extend(cgg_data.load_file_function_defs(p))
            except Exception:
                # skip files that can't be parsed
                continue

        # Limit number of samples
        if max_samples is not None and len(self.funcs) > max_samples:
            self.funcs = self.funcs[:max_samples]

        # Convert to networkx graphs
        self.graphs = [cgg_data.functiondef_to_nx(f) for f in self.funcs]
        self.max_nodes = max_nodes
        # Build vocabulary from functions
        self.vocab = cgg_data.build_vocab_from_functions(self.funcs)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        return self.graphs[idx]


def collate_graphs(
    batch: List, vocab: dict, max_nodes: int | None = None
) -> GraphBatch:
    return cgg_data.graphs_to_batch(batch, vocab=vocab, max_nodes=max_nodes)


def train_cgvae(dataloader: DataLoader, device: torch.device, args):
    vocab_size = len(dataloader.dataset.vocab)
    model = ConstrainedGraphVAE(input_dim=vocab_size, hidden_dim=128, latent_dim=64).to(
        device
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        for i, graphs in enumerate(dataloader):
            # DataLoader may now yield a GraphBatch directly (preferred), but
            # keep compatibility if it yields a raw list of graphs.
            if isinstance(graphs, GraphBatch):
                batch = graphs
            else:
                batch = collate_graphs(
                    graphs, dataloader.dataset.vocab, max_nodes=args.max_nodes
                )

            batch = batch.to(device)
            node_logits, edge_logits, mean, logvar = model(batch)
            loss_dict = model.loss(batch, node_logits, edge_logits, mean, logvar)
            loss = loss_dict["loss"]

            # Skip batch if loss is non-finite
            if not torch.isfinite(loss):
                continue

            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"[cgvae] epoch {epoch} loss={loss_dict['loss'].item():.4f}")
    torch.save(model.state_dict(), args.out)


def train_hier(dataloader: DataLoader, device: torch.device, args):
    vocab_size = len(dataloader.dataset.vocab)
    model = HierarchicalAutoregressiveModel(
        input_dim=vocab_size, hidden_dim=128, num_segments=8, num_node_tokens=vocab_size
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for graphs in dataloader:
            if isinstance(graphs, GraphBatch):
                batch = graphs
            else:
                batch = collate_graphs(
                    graphs, dataloader.dataset.vocab, max_nodes=args.max_nodes
                )
            batch = batch.to(device)
            # target tokens (from one-hot)
            target_tokens = batch.node_features.argmax(dim=-1)
            # derive pseudo segments by hashing token id
            target_segments = (target_tokens % model.num_segments).long()

            seg_logits, node_logits, edge_logits = model(
                batch, teacher_tokens=target_tokens
            )
            loss_dict = model.loss(
                seg_logits,
                node_logits,
                edge_logits,
                target_segments,
                target_tokens,
                batch.adjacency,
                batch.mask,
            )
            opt.zero_grad()
            loss_dict["loss"].backward()
            opt.step()
        print(f"[hier] epoch {epoch} loss={loss_dict['loss'].item():.4f}")
    torch.save(model.state_dict(), args.out)


def train_diffusion(dataloader: DataLoader, device: torch.device, args):
    vocab_size = len(dataloader.dataset.vocab)
    model = DiscreteDiffusionModel(
        input_dim=vocab_size, num_node_tokens=vocab_size, num_steps=100, hidden_dim=128
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for graphs in dataloader:
            # Accept either a pre-collated GraphBatch or a list of raw graphs
            if isinstance(graphs, GraphBatch):
                batch = graphs
            else:
                batch = collate_graphs(
                    graphs, dataloader.dataset.vocab, max_nodes=args.max_nodes
                )
            batch = batch.to(device)
            bsize = batch.node_features.shape[0]
            t = torch.randint(0, model.num_steps, (bsize,), device=device)
            loss_dict = model.p_losses(batch, t)
            opt.zero_grad()
            loss_dict["loss"].backward()
            opt.step()
        print(f"[diff] epoch {epoch} loss={loss_dict['loss'].item():.4f}")
    torch.save(model.state_dict(), args.out)


# =============================================================================
# New encoder + generator training functions
# =============================================================================


def train_vae_gen(
    dataloader: DataLoader, device: torch.device, args, test_indices=None
):
    """Train SharedEncoder + VAEGenerator."""
    vocab_size = len(dataloader.dataset.vocab)
    latent_dim = 128
    hidden_dim = 256

    encoder = SharedGraphEncoder(
        input_dim=vocab_size,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        variational=True,
    ).to(device)

    generator = VAEGenerator(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_node_tokens=vocab_size,
        max_nodes=args.max_nodes,
    ).to(device)

    params = list(encoder.parameters()) + list(generator.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    epoch_losses = []
    epoch_metrics = {"precision": [], "recall": [], "f1": []}

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_prec, epoch_rec, epoch_f1 = 0.0, 0.0, 0.0
        num_batches = 0

        for graphs in dataloader:
            if isinstance(graphs, GraphBatch):
                batch = graphs
            else:
                batch = collate_graphs(
                    graphs, dataloader.dataset.vocab, max_nodes=args.max_nodes
                )
            batch = batch.to(device)

            # Encode with variational encoder
            mean, logvar = encoder(batch)
            z = encoder.reparameterize(mean, logvar)

            # Get targets
            target_tokens = batch.node_features.argmax(dim=-1)
            target_edges = batch.adjacency
            mask = batch.mask

            # Forward
            outputs = generator(z, target_tokens=target_tokens, target_mask=mask)

            # Loss
            loss_dict = generator.loss(
                outputs["node_logits"],
                outputs["edge_logits"],
                outputs["stop_logits"],
                target_tokens,
                target_edges,
                mask,
                mean=mean,
                logvar=logvar,
            )
            loss = loss_dict["loss"]

            if not torch.isfinite(loss):
                continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            # Track metrics
            with torch.no_grad():
                metrics = compute_edge_metrics(
                    outputs["edge_logits"], target_edges, mask
                )
                epoch_prec += metrics["precision"]
                epoch_rec += metrics["recall"]
                epoch_f1 += metrics["f1"]

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_prec = epoch_prec / max(num_batches, 1)
        avg_rec = epoch_rec / max(num_batches, 1)
        avg_f1 = epoch_f1 / max(num_batches, 1)

        epoch_losses.append(avg_loss)
        epoch_metrics["precision"].append(avg_prec)
        epoch_metrics["recall"].append(avg_rec)
        epoch_metrics["f1"].append(avg_f1)

        print(
            f"[vae_gen] epoch {epoch} loss={avg_loss:.4f} "
            f"prec={avg_prec:.3f} rec={avg_rec:.3f} f1={avg_f1:.3f}"
        )

    _save_encoder_generator(
        encoder,
        generator,
        vocab_size,
        latent_dim,
        hidden_dim,
        args,
        vocab=dataloader.dataset.vocab,
        test_indices=test_indices,
        losses=epoch_losses,
        metrics=epoch_metrics,
    )


def train_diff_gen(
    dataloader: DataLoader, device: torch.device, args, test_indices=None
):
    """Train SharedEncoder + DiffusionGenerator."""
    vocab_size = len(dataloader.dataset.vocab)
    latent_dim = 128
    hidden_dim = 256

    encoder = SharedGraphEncoder(
        input_dim=vocab_size,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        variational=False,
    ).to(device)

    generator = DiffusionGenerator(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_node_tokens=vocab_size,
        max_nodes=args.max_nodes,
        num_steps=100,
    ).to(device)

    params = list(encoder.parameters()) + list(generator.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    epoch_losses = []
    epoch_metrics = {"precision": [], "recall": [], "f1": []}

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_prec, epoch_rec, epoch_f1 = 0.0, 0.0, 0.0
        num_batches = 0

        for graphs in dataloader:
            if isinstance(graphs, GraphBatch):
                batch = graphs
            else:
                batch = collate_graphs(
                    graphs, dataloader.dataset.vocab, max_nodes=args.max_nodes
                )
            batch = batch.to(device)

            # Encode
            z, _ = encoder(batch)

            # Get targets
            target_tokens = batch.node_features.argmax(dim=-1)
            target_edges = batch.adjacency
            mask = batch.mask

            # Forward (includes noise sampling)
            outputs = generator(z, target_tokens=target_tokens, target_mask=mask)

            # Loss
            loss_dict = generator.loss(
                outputs["node_logits"],
                outputs["edge_logits"],
                outputs["size_logits"],
                target_tokens,
                target_edges,
                mask,
            )
            loss = loss_dict["loss"]

            if not torch.isfinite(loss):
                continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            # Track metrics
            with torch.no_grad():
                metrics = compute_edge_metrics(
                    outputs["edge_logits"], target_edges, mask
                )
                epoch_prec += metrics["precision"]
                epoch_rec += metrics["recall"]
                epoch_f1 += metrics["f1"]

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_prec = epoch_prec / max(num_batches, 1)
        avg_rec = epoch_rec / max(num_batches, 1)
        avg_f1 = epoch_f1 / max(num_batches, 1)

        epoch_losses.append(avg_loss)
        epoch_metrics["precision"].append(avg_prec)
        epoch_metrics["recall"].append(avg_rec)
        epoch_metrics["f1"].append(avg_f1)

        print(
            f"[diff_gen] epoch {epoch} loss={avg_loss:.4f} "
            f"prec={avg_prec:.3f} rec={avg_rec:.3f} f1={avg_f1:.3f}"
        )

    _save_encoder_generator(
        encoder,
        generator,
        vocab_size,
        latent_dim,
        hidden_dim,
        args,
        vocab=dataloader.dataset.vocab,
        test_indices=test_indices,
        losses=epoch_losses,
        metrics=epoch_metrics,
    )


def _save_encoder_generator(
    encoder,
    generator,
    vocab_size,
    latent_dim,
    hidden_dim,
    args,
    vocab=None,
    test_indices=None,
    losses=None,
    metrics=None,
):
    """Save encoder + generator checkpoint and loss plot."""
    checkpoint = {
        "encoder": encoder.state_dict(),
        "generator": generator.state_dict(),
        "vocab_size": vocab_size,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "max_nodes": args.max_nodes,
        "generator_type": type(generator).__name__,
    }
    if vocab is not None:
        checkpoint["vocab"] = vocab
    if test_indices is not None:
        checkpoint["test_indices"] = test_indices
    if losses is not None:
        checkpoint["losses"] = losses
    if metrics is not None:
        checkpoint["metrics"] = metrics
    torch.save(checkpoint, args.out)
    print(f"Saved encoder + {type(generator).__name__} to {args.out}")

    # Save loss plot
    if losses is not None:
        model_name = type(generator).__name__
        save_loss_plot(losses, model_name, out_dir="results", metrics=metrics)


def train_autoreg(
    dataloader: DataLoader, device: torch.device, args, test_indices=None
):
    """Train the autoregressive graph generator.

    This trains an encoder-decoder setup:
    - Encoder: Creates latent embedding from input graph
    - Generator: Autoregressively generates graph from embedding

    At inference time, the encoder is replaced by the alignment adapter
    that maps text embeddings to the same latent space.
    """
    vocab_size = len(dataloader.dataset.vocab)
    latent_dim = 128
    hidden_dim = 256

    encoder = SharedGraphEncoder(
        input_dim=vocab_size,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        variational=False,
    ).to(device)

    generator = AutoregressiveGraphGenerator(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_node_tokens=vocab_size,
        max_nodes=args.max_nodes,
    ).to(device)

    params = list(encoder.parameters()) + list(generator.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    epoch_losses = []
    epoch_metrics = {"precision": [], "recall": [], "f1": []}

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_prec, epoch_rec, epoch_f1 = 0.0, 0.0, 0.0
        num_batches = 0

        for graphs in dataloader:
            if isinstance(graphs, GraphBatch):
                batch = graphs
            else:
                batch = collate_graphs(
                    graphs, dataloader.dataset.vocab, max_nodes=args.max_nodes
                )
            batch = batch.to(device)

            # Encode graphs to latent embeddings
            z, _ = encoder(batch)  # [B, latent_dim], None

            # Get target node tokens and edges
            target_tokens = batch.node_features.argmax(dim=-1)  # [B, N]
            target_edges = batch.adjacency  # [B, N, N]
            mask = batch.mask  # [B, N]

            # Forward pass with teacher forcing
            outputs = generator(
                z,
                target_tokens=target_tokens,
                target_adjacency=target_edges,
                target_mask=mask,
            )

            # Compute loss
            loss_dict = generator.loss(
                outputs["node_logits"],
                outputs["edge_logits"],
                outputs["stop_logits"],
                target_tokens,
                target_edges,
                mask,
            )
            loss = loss_dict["loss"]

            if not torch.isfinite(loss):
                continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            # Track metrics
            with torch.no_grad():
                metrics = compute_edge_metrics(
                    outputs["edge_logits"], target_edges, mask
                )
                epoch_prec += metrics["precision"]
                epoch_rec += metrics["recall"]
                epoch_f1 += metrics["f1"]

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_prec = epoch_prec / max(num_batches, 1)
        avg_rec = epoch_rec / max(num_batches, 1)
        avg_f1 = epoch_f1 / max(num_batches, 1)

        epoch_losses.append(avg_loss)
        epoch_metrics["precision"].append(avg_prec)
        epoch_metrics["recall"].append(avg_rec)
        epoch_metrics["f1"].append(avg_f1)

        print(
            f"[autoreg] epoch {epoch} loss={avg_loss:.4f} "
            f"prec={avg_prec:.3f} rec={avg_rec:.3f} f1={avg_f1:.3f}"
        )

    _save_encoder_generator(
        encoder,
        generator,
        vocab_size,
        latent_dim,
        hidden_dim,
        args,
        vocab=dataloader.dataset.vocab,
        test_indices=test_indices,
        losses=epoch_losses,
        metrics=epoch_metrics,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        choices=[
            # Original models (backwards compatibility)
            "cgvae",
            "hier",
            "diff",
            # New encoder+generator models
            "vae_gen",
            "diff_gen",
            # Pure autoregressive
            "autoreg",
        ],
        required=True,
        help="Model to train. Use *_gen variants for shared encoder architecture.",
    )
    p.add_argument("--data-path", default="data")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-nodes", type=int, default=128)
    p.add_argument(
        "--max-samples", type=int, default=500, help="Max samples to use (before split)"
    )
    p.add_argument(
        "--test-split",
        type=float,
        default=0.5,
        help="Fraction of data to hold out for testing",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="model_checkpoint.pt")
    args = p.parse_args()

    device = torch.device(args.device)

    # discover files
    paths = cgg_data.load_algorithm_file_paths()
    if len(paths) == 0:
        # try the path the user provided
        paths = list(pathlib.Path(args.data_path).glob("**/*.py"))

    # Create full dataset with max_samples limit
    full_dataset = GraphFunctionDataset(
        paths, max_nodes=args.max_nodes, max_samples=args.max_samples
    )
    if len(full_dataset) == 0:
        raise SystemExit(
            "No functions found in dataset path; please populate `data/` or provide --data-path"
        )

    # Train-test split
    n_total = len(full_dataset)
    n_test = int(n_total * args.test_split)
    n_train = n_total - n_test

    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_test], generator=torch.Generator().manual_seed(42)
    )

    # Copy vocab to split datasets for access
    train_dataset.vocab = full_dataset.vocab
    test_dataset.vocab = full_dataset.vocab

    print(f"Dataset: {n_total} total, {n_train} train, {n_test} test")

    # Provide a deterministic collate function that converts a list of graphs
    # into a single `GraphBatch`. This avoids accidental nested batches and
    # ensures each yielded element is a `GraphBatch`.
    def _collate_to_graphbatch(batch_list):
        return collate_graphs(batch_list, full_dataset.vocab, max_nodes=args.max_nodes)

    train_dl = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate_to_graphbatch,
    )

    # Save test indices for later use in generate_sample
    test_indices = list(test_dataset.indices)

    # Route to appropriate training function
    if args.model == "cgvae":
        train_cgvae(train_dl, device, args)
    elif args.model == "hier":
        train_hier(train_dl, device, args)
    elif args.model == "diff":
        train_diffusion(train_dl, device, args)
    elif args.model == "vae_gen":
        train_vae_gen(train_dl, device, args, test_indices=test_indices)
    elif args.model == "diff_gen":
        train_diff_gen(train_dl, device, args, test_indices=test_indices)
    elif args.model == "autoreg":
        train_autoreg(train_dl, device, args, test_indices=test_indices)
    else:
        raise ValueError(f"Unknown model: {args.model}")


if __name__ == "__main__":
    main()
