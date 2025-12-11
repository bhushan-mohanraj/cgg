# Code Graph Generation

Bhushan, Eric, Chirag

# Set Up

- `uv venv`: create a virtual environment
  - Activate the environment as instructed
- `uv sync`: install and update dependencies
- `uv add DEPENDENCY`: add a dependency

- `uv run python -m cgg`: run the current test or script
  - Update `__main__.py`
  to use either a test from `cgg.tests`
  or a script from `cgg.scripts`.
