"""Package-level training utilities for `cgg` models.

Run with: `python -m cgg.train --model cgvae ...`

This is a near-duplicate of the top-level `train.py` but adapted to live inside
the `cgg` package (uses relative imports and doesn't modify sys.path).
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Iterable, List

import torch
from torch.utils.data import DataLoader, Dataset

from . import data as cgg_data
from .models import (
    ConstrainedGraphVAE,
    HierarchicalAutoregressiveModel,
    DiscreteDiffusionModel,
)
from .models.types import GraphBatch


class GraphFunctionDataset(Dataset):
    """Dataset of NetworkX graphs created from top-level function ASTs."""

    def __init__(self, paths: Iterable[pathlib.Path], max_nodes: int = 128):
        self.funcs = []
        for p in paths:
            try:
                self.funcs.extend(cgg_data.load_file_function_defs(p))
            except Exception:
                # skip files that can't be parsed
                continue

        # Convert to networkx graphs
        self.graphs = [cgg_data.functiondef_to_nx(f) for f in self.funcs]
        self.max_nodes = max_nodes
        # Build vocabulary from functions
        self.vocab = cgg_data.build_vocab_from_functions(self.funcs)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        return self.graphs[idx]


def collate_graphs(batch: List, vocab: dict, max_nodes: int | None = None) -> GraphBatch:
    return cgg_data.graphs_to_batch(batch, vocab=vocab, max_nodes=max_nodes)


def train_cgvae(dataloader: DataLoader, device: torch.device, args):
    vocab_size = len(dataloader.dataset.vocab)
    model = ConstrainedGraphVAE(input_dim=vocab_size, hidden_dim=128, latent_dim=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        for i, graphs in enumerate(dataloader):
            # DataLoader may now yield a GraphBatch directly (preferred), but
            # keep compatibility if it yields a raw list of graphs.
            if isinstance(graphs, GraphBatch):
                batch = graphs
            else:
                batch = collate_graphs(graphs, dataloader.dataset.vocab, max_nodes=args.max_nodes)

            batch = batch.to(device)
            node_logits, edge_logits, mean, logvar = model(batch)

            # Immediately check forward outputs for non-finite values (robust diagnostics)
            def tensor_stats(t, name: str):
                try:
                    td = t.detach().cpu()
                    n_nan = int(torch.isnan(td).sum().item())
                    n_inf = int((~torch.isfinite(td)).sum().item())
                    finite_mask = torch.isfinite(td)
                    finite_vals = td[finite_mask]
                    if finite_vals.numel() > 0:
                        mean = float(finite_vals.mean().item())
                        mx = float(finite_vals.max().item())
                        mn = float(finite_vals.min().item())
                    else:
                        mean = float('nan')
                        mx = float('nan')
                        mn = float('nan')
                    print(f"{name}: shape={tuple(td.shape)} n_nan={n_nan} n_inf={n_inf} mean={mean:.6f} max={mx:.6f} min={mn:.6f}")
                    # print first 8 values (replace non-finite with 0 for inspect)
                    flat = td.flatten()
                    if flat.numel() > 0:
                        sample = flat.clone()
                        sample[~torch.isfinite(sample)] = 0.0
                        sample = sample[:8].numpy()
                        print(f"{name} sample:", sample)
                except Exception as exc:
                    print(f"{name}: <could not inspect: {exc}>")

            tensor_stats(node_logits, "node_logits")
            tensor_stats(edge_logits, "edge_logits")
            tensor_stats(mean, "mean")
            tensor_stats(logvar, "logvar")

            loss_dict = model.loss(batch, node_logits, edge_logits, mean, logvar)
            loss = loss_dict["loss"]

            # Detect NaN or Inf and dump diagnostics
            if not torch.isfinite(loss):
                print("Detected non-finite loss during training. Dumping diagnostics:")
                try:
                    print("loss:", loss)
                    print("batch.mask sum:", batch.mask.sum().item())
                    print("node_features: mean/min/max", batch.node_features.mean().item(), batch.node_features.min().item(), batch.node_features.max().item())
                    print("adjacency: mean/min/max", batch.adjacency.mean().item(), batch.adjacency.min().item(), batch.adjacency.max().item())
                except Exception:
                    pass

                def stats(tensor, name: str):
                    try:
                        print(f"{name}: mean={torch.nanmean(tensor).item():.6f} min={torch.nanmin(tensor).item():.6f} max={torch.nanmax(tensor).item():.6f} n_nan={int(torch.isnan(tensor).sum().item())} n_inf={int((~torch.isfinite(tensor)).sum().item())}")
                    except Exception:
                        print(f"{name}: <unavailable>")

                stats(node_logits, "node_logits")
                stats(edge_logits, "edge_logits")
                stats(mean, "mean")
                stats(logvar, "logvar")

                # check model parameters for non-finite values
                for n, p in model.named_parameters():
                    if not torch.isfinite(p).all():
                        print(f"parameter {n} contains non-finite values: n_nan={int(torch.isnan(p).sum().item())} n_inf={int((~torch.isfinite(p)).sum().item())}")

                # Suggest common fixes
                print("Suggestions: reduce learning rate, inspect input data, or normalize vocab/tokens. Aborting.")
                raise RuntimeError("Non-finite loss encountered; see diagnostics above.")

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
            # Debug: inspect incoming 'graphs' batch
            try:
                print(f"train_diffusion: received graphs batch of length={len(graphs)}; first type={type(graphs[0])}")
            except Exception:
                print(f"train_diffusion: received graphs batch of type {type(graphs)}")
            if isinstance(graphs, GraphBatch):
                batch = graphs
            else:
                batch = collate_graphs(graphs, dataloader.dataset.vocab, max_nodes=args.max_nodes)
            print(f"train_diffusion: produced GraphBatch.node_features.shape={tuple(batch.node_features.shape)}")
            batch = batch.to(device)
            # target tokens (from one-hot)
            target_tokens = batch.node_features.argmax(dim=-1)
            # derive pseudo segments by hashing token id
            target_segments = (target_tokens % model.num_segments).long()

            seg_logits, node_logits = model(batch, teacher_tokens=target_tokens)
            loss_dict = model.loss(seg_logits, node_logits, target_segments, target_tokens, batch.mask)
            opt.zero_grad()
            loss_dict["loss"].backward()
            opt.step()
        print(f"[hier] epoch {epoch} loss={loss_dict['loss'].item():.4f}")
    torch.save(model.state_dict(), args.out)


def train_diffusion(dataloader: DataLoader, device: torch.device, args):
    vocab_size = len(dataloader.dataset.vocab)
    model = DiscreteDiffusionModel(input_dim=vocab_size, num_node_tokens=vocab_size, num_steps=100, hidden_dim=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for graphs in dataloader:
            try:
                print(f"train_diffusion: received graphs batch length={len(graphs)}; first_type={type(graphs[0])}")
            except Exception:
                print(f"train_diffusion: received graphs batch of type {type(graphs)}")
            # Accept either a pre-collated GraphBatch or a list of raw graphs
            if isinstance(graphs, GraphBatch):
                batch = graphs
            else:
                batch = collate_graphs(graphs, dataloader.dataset.vocab, max_nodes=args.max_nodes)
            print(f"train_diffusion: produced GraphBatch.node_features.shape={tuple(batch.node_features.shape)} dtype={batch.node_features.dtype}")
            batch = batch.to(device)
            bsize = batch.node_features.shape[0]
            t = torch.randint(0, model.num_steps, (bsize,), device=device)
            loss_dict = model.p_losses(batch, t)
            opt.zero_grad()
            loss_dict["loss"].backward()
            opt.step()
        print(f"[diff] epoch {epoch} loss={loss_dict['loss'].item():.4f}")
    torch.save(model.state_dict(), args.out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cgvae", "hier", "diff"], required=True)
    p.add_argument("--data-path", default="data")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-nodes", type=int, default=128)
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

    dataset = GraphFunctionDataset(paths, max_nodes=args.max_nodes)
    if len(dataset) == 0:
        raise SystemExit("No functions found in dataset path; please populate `data/` or provide --data-path")

    # Provide a deterministic collate function that converts a list of graphs
    # into a single `GraphBatch`. This avoids accidental nested batches and
    # ensures each yielded element is a `GraphBatch`.
    def _collate_to_graphbatch(batch_list):
        return collate_graphs(batch_list, dataset.vocab, max_nodes=args.max_nodes)

    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=_collate_to_graphbatch)

    if args.model == "cgvae":
        train_cgvae(dl, device, args)
    elif args.model == "hier":
        train_hier(dl, device, args)
    else:
        train_diffusion(dl, device, args)


if __name__ == "__main__":
    main()
