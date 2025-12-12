"""Decode a single sample from a ConstrainedGraphVAE embedding and save a visualization.

Usage: python -m cgg.decode_sample --ckpt PATH --sample 0 --out out.png
If no checkpoint is provided, an untrained model will be used.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import networkx as nx
import torch

from .train import GraphFunctionDataset
from . import data as cgg_data
from .models.constrained_gvae import ConstrainedGraphVAE
from .models.discrete_diffusion import DiscreteDiffusionModel
from .models.hierarchical_autoregressive import HierarchicalAutoregressiveModel
from .models.types import GraphBatch
from .graph_utils import single_logits_to_nx
import torch.nn.functional as F


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", help="model checkpoint path", default=None)
    p.add_argument("--model", choices=["vae", "hier", "diff"], default="diff", help="which model to decode with")
    p.add_argument("--denoise", action="store_true", help="run iterative denoising sampling (diffusion only)")
    p.add_argument("--denoise-steps", type=int, default=50, help="number of denoising steps to run when --denoise is set")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--out", default="decoded.png")
    p.add_argument("--max-nodes", type=int, default=64)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    paths = cgg_data.load_algorithm_file_paths()
    if len(paths) == 0:
        paths = list(pathlib.Path("data").glob("**/*.py"))

    dataset = GraphFunctionDataset(paths, max_nodes=args.max_nodes)
    if len(dataset) == 0:
        raise SystemExit("no graphs found in data/")

    sample_idx = args.sample % len(dataset)
    graph = dataset.graphs[sample_idx]

    # create a GraphBatch for a single graph
    batch = cgg_data.graphs_to_batch([graph], vocab=dataset.vocab, max_nodes=args.max_nodes)
    batch = batch.to(device)

    node_logits = None
    edge_logits = None

    if args.model == "vae":
        vae = ConstrainedGraphVAE(input_dim=len(dataset.vocab), hidden_dim=128, latent_dim=64).to(device)
        if args.ckpt:
            sd = torch.load(args.ckpt, map_location=device)
            vae.load_state_dict(sd)

        vae.eval()
        with torch.no_grad():
            mean, logvar = vae.encode(batch)
            node_logits, edge_logits = vae.decode(mean, batch)

    elif args.model == "hier":
        hier = HierarchicalAutoregressiveModel(
            input_dim=len(dataset.vocab), hidden_dim=128, num_node_tokens=len(dataset.vocab)
        ).to(device)
        if args.ckpt:
            sd = torch.load(args.ckpt, map_location=device)
            hier.load_state_dict(sd)

        hier.eval()
        with torch.no_grad():
            segment_logits, node_logits = hier(batch)
            # hierarchical model does not predict edges; reuse input adjacency
            edge_logits = batch.adjacency

    elif args.model == "diff":
        # If a checkpoint is provided, inspect it to determine the original num_steps
        sd = None
        if args.ckpt:
            sd = torch.load(args.ckpt, map_location=device)

        # try to infer num_steps from saved 'betas' buffer in the checkpoint
        num_steps = None
        if sd is not None:
            for k, v in sd.items():
                if k.endswith("betas") or k.endswith(".betas"):
                    try:
                        num_steps = int(v.shape[0])
                        break
                    except Exception:
                        continue

        if num_steps is None:
            num_steps = 1000

        diff = DiscreteDiffusionModel(
            input_dim=len(dataset.vocab), num_node_tokens=len(dataset.vocab), num_steps=num_steps, hidden_dim=128
        ).to(device)
        if sd is not None:
            diff.load_state_dict(sd)

        diff.eval()
        B = batch.node_features.shape[0]

        # If requested, run an approximate iterative denoising (fast schedule)
        if args.denoise:
            num_steps = getattr(diff, "num_steps", 1000)
            # create timesteps spaced to run roughly denoise_steps iterations
            step = max(1, num_steps // max(1, args.denoise_steps))
            timesteps = list(range(num_steps - 1, -1, -step))

            # initialize with pure noise tokens
            x = torch.randint(0, diff.num_node_tokens, (B, batch.node_features.shape[1]), device=device)

            for t_val in timesteps:
                # build a batch with current tokens encoded as one-hot
                one_hot = F.one_hot(x, num_classes=diff.num_node_tokens).to(dtype=batch.node_features.dtype)
                cur_batch = GraphBatch(node_features=one_hot, adjacency=batch.adjacency, mask=batch.mask)
                with torch.no_grad():
                    out = diff(cur_batch, torch.full((B,), t_val, dtype=torch.long, device=device))
                logits = out["node_logits"] if isinstance(out, dict) else out[0]
                # greedy update
                x = logits.argmax(dim=-1)

            # final outputs from last iteration
            node_logits = logits
            edge_logits = out.get("edge_logits") if isinstance(out, dict) else None
        else:
            # single-step prediction at t=0 (model predicts clean tokens from near-clean input)
            t0 = torch.zeros((batch.node_features.shape[0],), dtype=torch.long, device=device)
            with torch.no_grad():
                out = diff(batch, t0)
            node_logits = out["node_logits"]
            edge_logits = out["edge_logits"]

    # Convert to NetworkX and draw
    # include all predicted edges (don't threshold at 0.5)
    g_pred = single_logits_to_nx(
        node_logits[0].cpu(), edge_logits[0].cpu(), vocab=dataset.vocab, mask=batch.mask[0].cpu(), edge_thresh=0.0
    )

    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(g_pred)
    labels = {n: data.get("token", data.get("label")) for n, data in g_pred.nodes(data=True)}
    nx.draw(g_pred, pos=pos, labels=labels, with_labels=True, node_size=200)
    plt.title(f"Decoded graph from sample {sample_idx}")
    plt.savefig(args.out)
    print(f"Saved decoded graph to {args.out}")


if __name__ == "__main__":
    main()
