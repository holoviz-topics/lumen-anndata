# ✨ lumen-anndata

[![CI](https://img.shields.io/github/actions/workflow/status/holoviz-topics/lumen-anndata/ci.yml?style=flat-square&branch=main)](https://github.com/holoviz-topics/lumen-anndata/actions/workflows/ci.yml)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/lumen-anndata?logoColor=white&logo=conda-forge&style=flat-square)](https://prefix.dev/channels/conda-forge/packages/lumen-anndata)
[![pypi-version](https://img.shields.io/pypi/v/lumen-anndata.svg?logo=pypi&logoColor=white&style=flat-square)](https://pypi.org/project/lumen-anndata)
[![python-version](https://img.shields.io/pypi/pyversions/lumen-anndata?logoColor=white&logo=python&style=flat-square)](https://pypi.org/project/lumen-anndata)


## ⚠️ Early Development Warning

This project is in pre-alpha development stage and NOT READY for use.
We are actively developing core functionality and the API may change significantly with each update.

## Overview

lumen-anndata is an extension to enable [Lumen](https://lumen.holoviz.org/) to interact with [anndata](https://anndata.readthedocs.io/) objects, a common format for single-cell genomics data. It aims to allow researchers to use natural language to explore, analyze, and visualize complex single-cell datasets through a chat interface.

## Features (Planned)
lumen-anndata aims to support the following capabilities:

- Natural language querying of anndata objects
- Integration with select scanpy functions for running analysis and static plotting
- Interactive visualization of single-cell data (UMAP, DotMap, HeatMap, Dendrogram, etc.) with HoloViz

## Installation

Install it via `pip`:

```bash
pip install lumen-anndata
```

## Usage

To launch the Lumen app, run:

```bash
lumen-anndata
```

## Development

```bash
git clone https://github.com/holoviz-topics/lumen-anndata
cd lumen-anndata
```

For a simple setup use [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv
source .venv/bin/activate # on linux. Similar commands for windows and osx
uv pip install -e .[dev]
uv pip install "git+https://github.com/holoviz/lumen@main#egg=lumen[ai-llama]"
pre-commit run
pytest tests
```

For the full Github Actions setup use [pixi](https://pixi.sh):

```bash
pixi run pre-commit-install
pixi run postinstall
pixi run test
```

This repository is adapted from [copier-template-panel-extension](https://github.com/panel-extensions/copier-template-panel-extension)

To update to the latest template version run:

```bash
pixi exec --spec copier --spec ruamel.yaml -- copier update --defaults --trust
```

Note: `copier` will show `Conflict` for files with manual changes during an update. This is normal. As long as there are no merge conflict markers, all patches applied cleanly.

To update the embeddings to use a new version of `scanpy`, increment the version in `scripts/embed_docs.py` and run:

```bash
python scripts/embed_docs.py
```

Ensure you have an OpenAI API key set in your environment variables. You can set it in your terminal session with:

```bash
export OPENAI_API_KEY="sk-..."
```

## ❤️ Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/YourFeature`.
3. Make your changes and commit them: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature/YourFeature`.
5. Open a pull request.

Please ensure your code adheres to the project's coding standards and passes all tests.
