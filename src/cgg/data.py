"""
Process the data for model training.
Load function definitions from `TheAlgorithms`
and convert those functions to code graphs analagous to ASTs.
"""

import ast
import pathlib

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

    with path.open("r") as f:
        module: ast.Module = ast.parse(f.read())

    return [
        node
        for node in module.body
        # All top-level functions
        if isinstance(node, ast.FunctionDef)
        # Ignore the main function defined in some modules
        and node.name != "main"
    ]
