import ast
import random

from . import data


def test_load_function_defs():
    path = random.choice(data.load_algorithm_file_paths())

    print(path)
    for function_def in data.load_file_function_defs(path):
        print(function_def.name)
        print(ast.dump(function_def, indent=4))
