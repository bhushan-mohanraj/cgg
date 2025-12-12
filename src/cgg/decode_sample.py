"""Decode a single sample and save visualizations of both predicted and original graphs.

Usage: python -m cgg.decode_sample --ckpt PATH --sample 0 --out out.png
If no checkpoint is provided, an untrained model will be used.

What this script does:
1. Loads the dataset of code graphs (parsed from Python AST)
2. Picks the graph at index `--sample` (e.g., sample 5 = the 6th function in the dataset)
3. Converts that graph to a batched tensor representation (GraphBatch)
4. Runs the specified model to predict node labels and edges from the input
5. Saves TWO images:
   - The predicted graph (from model output) -> {out}
   - The original ground-truth graph -> {out_dir}/{sample}-original.png
"""

from __future__ import annotations

import argparse
import pathlib

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


def draw_graph(g: nx.Graph, title: str, out_path: str):
    """Draw a NetworkX graph and save to file."""
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(g, seed=42)
    labels = {
        n: data.get("token") or data.get("label", str(n))
        for n, data in g.nodes(data=True)
    }
    nx.draw(g, pos=pos, labels=labels, with_labels=True, node_size=300, font_size=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", help="model checkpoint path", default=None)
    p.add_argument(
        "--model",
        choices=["vae", "hier", "diff"],
        default="diff",
        help="which model to decode with",
    )
    p.add_argument(
        "--denoise",
        action="store_true",
        help="run iterative denoising sampling (diffusion only)",
    )
    p.add_argument(
        "--denoise-steps",
        type=int,
        default=50,
        help="number of denoising steps to run when --denoise is set",
    )
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
    batch = cgg_data.graphs_to_batch(
        [graph], vocab=dataset.vocab, max_nodes=args.max_nodes
    )
    batch = batch.to(device)

    node_logits = None
    edge_logits = None

    if args.model == "vae":
        vae = ConstrainedGraphVAE(
            input_dim=len(dataset.vocab), hidden_dim=128, latent_dim=64
        ).to(device)
        if args.ckpt:
            sd = torch.load(args.ckpt, map_location=device)
            vae.load_state_dict(sd)

        vae.eval()
        with torch.no_grad():
            mean, logvar = vae.encode(batch)
            node_logits, edge_logits = vae.decode(mean, batch)

    elif args.model == "hier":
        hier = HierarchicalAutoregressiveModel(
            input_dim=len(dataset.vocab),
            hidden_dim=128,
            num_node_tokens=len(dataset.vocab),
        ).to(device)
        if args.ckpt:
            sd = torch.load(args.ckpt, map_location=device)
            hier.load_state_dict(sd)

        hier.eval()
        with torch.no_grad():
            segment_logits, node_logits, edge_logits = hier(batch)

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
            input_dim=len(dataset.vocab),
            num_node_tokens=len(dataset.vocab),
            num_steps=num_steps,
            hidden_dim=128,
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
            x = torch.randint(
                0,
                diff.num_node_tokens,
                (B, batch.node_features.shape[1]),
                device=device,
            )

            for t_val in timesteps:
                # build a batch with current tokens encoded as one-hot
                one_hot = F.one_hot(x, num_classes=diff.num_node_tokens).to(
                    dtype=batch.node_features.dtype
                )
                cur_batch = GraphBatch(
                    node_features=one_hot, adjacency=batch.adjacency, mask=batch.mask
                )
                with torch.no_grad():
                    out = diff(
                        cur_batch,
                        torch.full((B,), t_val, dtype=torch.long, device=device),
                    )
                logits = out["node_logits"] if isinstance(out, dict) else out[0]
                # greedy update
                x = logits.argmax(dim=-1)

            # final outputs from last iteration
            node_logits = logits
            edge_logits = out.get("edge_logits") if isinstance(out, dict) else None
        else:
            # single-step prediction at t=0 (model predicts clean tokens from near-clean input)
            t0 = torch.zeros(
                (batch.node_features.shape[0],), dtype=torch.long, device=device
            )
            with torch.no_grad():
                out = diff(batch, t0)
            node_logits = out["node_logits"]
            edge_logits = out["edge_logits"]

    # Convert to NetworkX and draw
    # Diagnostic prints: inspect edge logits / probabilities and ground-truth adjacency
    if edge_logits is None:
        print("Warning: model did not return edge_logits; no edges will be visualized.")
        # create a placeholder of very negative logits
        edge_logits = torch.full(
            (
                batch.node_features.shape[0],
                batch.node_features.shape[1],
                batch.node_features.shape[1],
            ),
            float("-inf"),
            device=node_logits.device,
        )

    # compute simple stats for the first sample
    el = edge_logits[0].cpu()
    probs = torch.sigmoid(el)
    mask0 = batch.mask[0].cpu().bool()
    valid_idxs = mask0.nonzero(as_tuple=False).squeeze(-1)
    if valid_idxs.numel() == 0:
        print("No valid nodes in mask for sample; nothing to show.")
    else:
        # restrict to valid node pairs for statistics
        sub_probs = probs[valid_idxs][:, valid_idxs]
        sub_logits = el[valid_idxs][:, valid_idxs]
        flat_logits = sub_logits.flatten()
        flat_probs = sub_probs.flatten()
        print(
            f"Edge logits: min={float(flat_logits.min()):.4f}, max={float(flat_logits.max()):.4f}, mean={float(flat_logits.mean()):.4f}"
        )
        print(
            f"Edge probs:  min={float(flat_probs.min()):.4f}, max={float(flat_probs.max()):.4f}, mean={float(flat_probs.mean()):.4f}"
        )
        n_pos = (flat_probs > 0.5).sum().item()
        tot = flat_probs.numel()
        print(
            f"Edges with prob>0.5: {n_pos}/{tot} ({100.0 * n_pos / max(1, tot):.2f}%)"
        )

        # show a few highest-probability edges
        topk = min(10, tot)
        vals, idxs = torch.topk(flat_probs, topk)
        edges_list = []
        N = sub_probs.shape[0]
        for v, idx in zip(vals.tolist(), idxs.tolist()):
            i = idx // N
            j = idx % N
            edges_list.append(
                (int(valid_idxs[i].item()), int(valid_idxs[j].item()), float(v))
            )
        print("Top edges (i, j, prob):", edges_list)

        # compare to ground-truth adjacency
        gt = batch.adjacency[0].cpu()
        gt_sub = gt[valid_idxs][:, valid_idxs]
        gt_count = int(gt_sub.sum().item())
        print(f"Ground-truth edges (masked): {gt_count}/{tot}")

    # threshold predicted edges at 0.5 (default behavior)
    g_pred = single_logits_to_nx(
        node_logits[0].cpu(),
        edge_logits[0].cpu(),
        vocab=dataset.vocab,
        mask=batch.mask[0].cpu(),
        edge_thresh=0.5,
    )

    # Ensure output directory exists
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Draw and save predicted graph
    draw_graph(
        g_pred,
        f"Predicted graph (sample {sample_idx}, model={args.model})",
        str(out_path),
    )

    # Also save the original ground-truth graph
    # The original graph is stored in dataset.graphs[sample_idx]
    original_graph = dataset.graphs[sample_idx]
    original_out = out_path.parent / f"{sample_idx}-original.png"
    draw_graph(
        original_graph, f"Original graph (sample {sample_idx})", str(original_out)
    )


if __name__ == "__main__":
    main()
