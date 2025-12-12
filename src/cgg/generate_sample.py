"""
Generate graphs from trained generators and compare with originals.

This script:
- Loads a trained encoder + generator checkpoint
- Uses test set samples (saved in checkpoint) for evaluation
- Encodes a reference graph and generates from its embedding
- Saves both original and generated graphs as separate images

Output format: {sample_number}-{model_type_or_original}.png

Usage:
    python -m cgg.generate_sample --checkpoint results/checkpoints/autoreg.pt --sample 0 --out-dir results/outputs
    python -m cgg.generate_sample --checkpoint results/checkpoints/vae_gen.pt --sample 5 --out-dir results/outputs
"""

import argparse
import pathlib
import os

import matplotlib.pyplot as plt
import networkx as nx
import torch

from . import data as cgg_data
from .models import (
    SharedGraphEncoder,
    AutoregressiveGraphGenerator,
    VAEGenerator,
    DiffusionGenerator,
)


def load_checkpoint(path: str, device: torch.device):
    """Load an encoder + generator checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Determine generator type
    generator_type = ckpt.get("generator_type", "AutoregressiveGraphGenerator")

    # Create generator based on type
    if generator_type == "VAEGenerator":
        generator = VAEGenerator(
            latent_dim=ckpt["latent_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_node_tokens=ckpt["vocab_size"],
            max_nodes=ckpt["max_nodes"],
        )
        model_name = "vae_gen"
    elif generator_type == "DiffusionGenerator":
        generator = DiffusionGenerator(
            latent_dim=ckpt["latent_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_node_tokens=ckpt["vocab_size"],
            max_nodes=ckpt["max_nodes"],
        )
        model_name = "diff_gen"
    else:  # AutoregressiveGraphGenerator or default
        generator = AutoregressiveGraphGenerator(
            latent_dim=ckpt["latent_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_node_tokens=ckpt["vocab_size"],
            max_nodes=ckpt["max_nodes"],
        )
        model_name = "autoreg"

    generator.load_state_dict(ckpt["generator"])
    generator.to(device)
    generator.eval()

    # Load encoder
    encoder = SharedGraphEncoder(
        input_dim=ckpt["vocab_size"],
        hidden_dim=ckpt["hidden_dim"],
        latent_dim=ckpt["latent_dim"],
        variational=(generator_type == "VAEGenerator"),
    )
    encoder.load_state_dict(ckpt["encoder"])
    encoder.to(device)
    encoder.eval()

    return generator, encoder, ckpt, model_name


def load_all_functions():
    """Load all functions from data directory."""
    paths = cgg_data.load_algorithm_file_paths()
    if not paths:
        paths = list(pathlib.Path("data").glob("**/*.py"))

    funcs = []
    for p in paths:
        try:
            funcs.extend(cgg_data.load_file_function_defs(p))
        except Exception:
            continue
    return funcs


def draw_graph(G: nx.DiGraph, title: str, out_path: str):
    """Draw a NetworkX graph and save to file."""
    fig, ax = plt.subplots(figsize=(10, 8))

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Empty Graph", ha="center", va="center", fontsize=16)
        ax.set_title(title)
    else:
        pos = nx.spring_layout(G, seed=42)
        labels = {n: str(G.nodes[n].get("label", n))[:10] for n in G.nodes()}
        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=True,
            labels=labels,
            node_size=300,
            node_color="lightblue",
            font_size=6,
            arrows=True,
            arrowsize=10,
        )
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def generate_from_sample(
    generator,
    encoder,
    ckpt,
    funcs,
    vocab,
    sample_idx: int,
    out_dir: str,
    model_name: str,
    max_nodes: int,
    temperature: float,
    device,
):
    """Generate a graph from a test sample and save both original and generated."""

    reverse_vocab = {v: k for k, v in vocab.items()}

    # Get the function at this index
    ref_func = funcs[sample_idx]
    ref_graph = cgg_data.functiondef_to_nx(ref_func)

    # Save original graph
    original_path = os.path.join(out_dir, f"{sample_idx}-original.png")
    draw_graph(
        ref_graph,
        f"Original: {ref_func.name} ({ref_graph.number_of_nodes()} nodes)",
        original_path,
    )

    # Encode reference graph
    batch = cgg_data.graphs_to_batch(
        [ref_graph], vocab=vocab, max_nodes=ckpt["max_nodes"]
    )
    batch = batch.to(device)

    with torch.no_grad():
        result = encoder(batch)
        if isinstance(result, tuple):
            mean, logvar = result
            if logvar is not None:
                z = encoder.reparameterize(mean, logvar)
            else:
                z = mean
        else:
            z = result

    # Generate graph
    with torch.no_grad():
        result = generator.generate(z, max_nodes=max_nodes, temperature=temperature)
        if isinstance(result, tuple) and len(result) == 2:
            node_tokens_batch, edge_list_batch = result
            node_tokens = (
                node_tokens_batch[0]
                if isinstance(node_tokens_batch[0], list)
                else node_tokens_batch
            )
            edge_list = (
                edge_list_batch[0]
                if isinstance(edge_list_batch[0], list)
                else edge_list_batch
            )
        else:
            node_tokens, edge_list = result

    # Convert to NetworkX
    G = nx.DiGraph()
    for i, token in enumerate(node_tokens):
        label = reverse_vocab.get(token, f"UNK_{token}")
        G.add_node(i, label=label, token=token)
    for src, dst in edge_list:
        if src < len(node_tokens) and dst < len(node_tokens):
            G.add_edge(src, dst)

    # Save generated graph
    generated_path = os.path.join(out_dir, f"{sample_idx}-{model_name}.png")
    draw_graph(
        G,
        f"Generated by {model_name}: ({len(node_tokens)} nodes, {len(edge_list)} edges)",
        generated_path,
    )

    return ref_graph, G


def main():
    parser = argparse.ArgumentParser(description="Generate graph from embedding")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--out-dir", default="results/outputs", help="Output directory")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Sample index to use (from test set if available)",
    )
    parser.add_argument(
        "--use-test-set",
        action="store_true",
        help="Use test set indices from checkpoint (default: use raw index)",
    )
    parser.add_argument(
        "--max-nodes", type=int, default=32, help="Max nodes to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    generator, encoder, ckpt, model_name = load_checkpoint(args.checkpoint, device)

    # Load vocab from checkpoint
    if "vocab" in ckpt:
        vocab = ckpt["vocab"]
        print(f"Loaded vocabulary from checkpoint ({len(vocab)} tokens)")
    else:
        raise ValueError("Checkpoint does not contain vocabulary. Please retrain.")

    # Load all functions
    funcs = load_all_functions()
    print(f"Loaded {len(funcs)} functions from data")

    # Determine which sample to use
    if args.use_test_set and "test_indices" in ckpt:
        test_indices = ckpt["test_indices"]
        if args.sample >= len(test_indices):
            raise ValueError(
                f"Sample {args.sample} not in test set (only {len(test_indices)} test samples)"
            )
        sample_idx = test_indices[args.sample]
        print(f"Using test sample {args.sample} (raw index {sample_idx})")
    else:
        sample_idx = args.sample
        print(f"Using raw sample index {sample_idx}")

    if sample_idx >= len(funcs):
        raise ValueError(
            f"Sample index {sample_idx} out of range (only {len(funcs)} functions)"
        )

    print(f"Generator: {model_name}, latent_dim={ckpt['latent_dim']}")
    print(f"Output directory: {args.out_dir}")
    print()

    # Generate
    ref_graph, gen_graph = generate_from_sample(
        generator,
        encoder,
        ckpt,
        funcs,
        vocab,
        sample_idx,
        args.out_dir,
        model_name,
        args.max_nodes,
        args.temperature,
        device,
    )

    print()
    print(
        f"Original: {ref_graph.number_of_nodes()} nodes, {ref_graph.number_of_edges()} edges"
    )
    print(
        f"Generated: {gen_graph.number_of_nodes()} nodes, {gen_graph.number_of_edges()} edges"
    )


if __name__ == "__main__":
    main()
