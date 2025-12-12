"""
Process the data for model training.
Load function definitions from `TheAlgorithms`
and convert those functions to code graphs analagous to ASTs.
"""

import ast
import pathlib
from typing import Iterable, List, Dict, Optional, Tuple

import networkx as nx
import torch

from .models.types import GraphBatch

DATA_PATH = pathlib.Path("data")
COLLECTIONS = [
    "boolean_algebra",
    "divide_and_conquer",
    "sorts",
    "physics",
]


def load_algorithm_file_paths() -> list[pathlib.Path]:
    paths = []

    for collection in COLLECTIONS:
        collection_path = DATA_PATH / collection
        paths.extend(
            path for path in collection_path.glob("*.py") if path.stem != "__init__"
        )

    return paths


def load_file_function_defs(path: pathlib.Path) -> list[ast.FunctionDef]:
    """
    Load all function definitions,
    as syntax trees produced by the Python standard library,
    from the Python file at `path`.
    """
    # Try to read the file as UTF-8; if decoding fails skip the file.
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, Exception):
        # Skip files that are not valid UTF-8 or cannot be read
        return []

    try:
        module: ast.Module = ast.parse(text)
    except SyntaxError:
        # Skip files that fail to parse
        return []

    return [
        node
        for node in module.body
        # All top-level functions
        if isinstance(node, ast.FunctionDef)
        # Ignore the main function defined in some modules
        and node.name != "main"
    ]


def functiondef_to_nx(func: ast.FunctionDef) -> nx.DiGraph:
    """Convert an `ast.FunctionDef` into a NetworkX directed graph.

    Nodes have attributes:
      - `label`: the AST node type name (e.g. "If", "Call", "Name")
      - `token`: optional short token (for Name/Constant/arg/attr)

    Edges are parent -> child (directed), and we also add a reverse edge
    to make adjacency effectively undirected for the models.
    """
    g = nx.DiGraph()

    node_counter = 0

    def add_node(n) -> int:
        nonlocal node_counter
        nid = node_counter
        node_counter += 1
        label = type(n).__name__
        token = None
        # Extract small tokens for certain node types
        if isinstance(n, ast.Name):
            token = n.id
        elif isinstance(n, ast.arg):
            token = n.arg
        elif isinstance(n, ast.Attribute):
            token = n.attr
        elif isinstance(n, ast.Constant):
            token = repr(n.value)
        elif isinstance(n, ast.Call):
            # try to get function name if simple
            if isinstance(n.func, ast.Name):
                token = n.func.id
        g.add_node(nid, label=label, token=token)
        return nid

    # Walk the AST and add nodes in a parent-first order
    parents: List[Tuple[int, ast.AST]] = []

    def visit(node, parent_id: Optional[int] = None):
        my_id = add_node(node)
        if parent_id is not None:
            g.add_edge(parent_id, my_id)
            g.add_edge(my_id, parent_id)
        for child in ast.iter_child_nodes(node):
            visit(child, my_id)

    visit(func)
    return g


def build_vocab_from_functions(funcs: Iterable[ast.FunctionDef], min_count: int = 1) -> Dict[str, int]:
    """Build a simple token vocabulary from a list of FunctionDef ASTs.

    Tokens are the `token` attribute attached to AST nodes (names, constants, attrs),
    falling back to the AST node label if token is None.
    """
    from collections import Counter

    ctr: Counter = Counter()
    for f in funcs:
        g = functiondef_to_nx(f)
        for _, data in g.nodes(data=True):
            tok = data.get("token") or data.get("label")
            if tok is None:
                continue
            ctr[str(tok)] += 1

    # Reserve 0 for PAD and 1 for UNK
    vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for tok, count in ctr.items():
        if count >= min_count:
            vocab[tok] = len(vocab)
    return vocab


def graphs_to_batch(graphs: List[nx.DiGraph], vocab: Dict[str, int], max_nodes: Optional[int] = None) -> GraphBatch:
    """Convert a list of NetworkX graphs into a batched `GraphBatch`.

    - node_features: one-hot token vectors (batch, nodes, vocab_size)
    - adjacency: binary adjacency (batch, nodes, nodes)
    - mask: boolean mask (batch, nodes)

    Graphs are truncated or padded to `max_nodes` (if provided) or to the largest
    graph in the list.
    """
    # Some DataLoader/collector setups may produce a nested list (batch of batches).
    # If so, flatten one level.
    if len(graphs) > 0 and isinstance(graphs[0], (list, tuple)):
        graphs = [g for sub in graphs for g in sub]

    # If the DataLoader was given a list of already-batched GraphBatch objects,
    # concatenate them along the batch dimension. This can happen if a custom
    # collate function or a previous step produced `GraphBatch` per-worker.
    if len(graphs) > 0 and isinstance(graphs[0], GraphBatch):
        try:
            node_features = torch.cat([g.node_features for g in graphs], dim=0)
            adjacency = torch.cat([g.adjacency for g in graphs], dim=0)
            mask = torch.cat([g.mask for g in graphs], dim=0)
            return GraphBatch(node_features=node_features, adjacency=adjacency, mask=mask)
        except Exception:
            # Fall back to normal path if concatenation fails for unexpected shapes
            pass

    batch_size = len(graphs)
    vocab_size = len(vocab)
    sizes = [g.number_of_nodes() for g in graphs]
    target_nodes = max(sizes) if max_nodes is None else min(max(sizes), max_nodes)

    node_features = torch.zeros((batch_size, target_nodes, vocab_size), dtype=torch.float32)
    adjacency = torch.zeros((batch_size, target_nodes, target_nodes), dtype=torch.float32)
    mask = torch.zeros((batch_size, target_nodes), dtype=torch.bool)

    for i, g in enumerate(graphs):
        if not isinstance(g, nx.Graph) and not isinstance(g, nx.DiGraph):
            # skip unexpected entries
            continue
        # ensure node ordering deterministic: nodes were created incrementally
        nodes = list(g.nodes())[:target_nodes]
        for j, nid in enumerate(nodes):
            data = g.nodes[nid]
            tok = data.get("token") or data.get("label")
            idx = vocab.get(str(tok), vocab.get("<UNK>"))
            node_features[i, j, idx] = 1.0
            mask[i, j] = True

        # adjacency
        for u, v in g.edges():
            # skip edges that go to truncated nodes
            try:
                uj = nodes.index(u)
                vj = nodes.index(v)
            except ValueError:
                continue
            adjacency[i, uj, vj] = 1.0

    return GraphBatch(node_features=node_features, adjacency=adjacency, mask=mask)
