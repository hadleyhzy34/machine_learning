# Deep Learning Basics (dl-basics)

This repository contains various machine learning and deep learning implementations (GANs, SVM, GRU, ViT, SimCLR, etc.).

## Setup

This project uses [uv](https://github.com/astral-sh/uv) to manage its environment and dependencies.

To set up the environment and install all dependencies:
```bash
uv sync
```

This will automatically create a virtual environment in `.venv` and install the configured dependencies, including the PyTorch CPU nightly build.

## Running Scripts

To run any script (e.g. `main.py`) inside the environment, prefix the command with `uv run`:
```bash
uv run python main.py
```

To run interactive Python or IPython:
```bash
uv run ipython
```

## Adding Dependencies

To add a new dependency:
```bash
uv add <package-name>
```
