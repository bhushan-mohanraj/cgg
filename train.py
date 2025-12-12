"""Shim to run the package-level trainer.

This file is a lightweight shim so users who run the repository-level
`python train.py` still execute `cgg.train` after the module was moved
into the package.
"""

if __name__ == "__main__":
    # Run the package module to preserve CLI behavior
    import runpy

    runpy.run_module("cgg.train", run_name="__main__")
