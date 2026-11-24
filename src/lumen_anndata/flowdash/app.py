"""A Lumen FlowDash app over AnnData, with lumen-anndata's nodes and agent tools."""
from __future__ import annotations

import os

from collections.abc import Sequence
from functools import partial
from unittest import mock
from pathlib import Path
from typing import Any

import pandas as pd

from lumen.flowdash.contracts import (
    SOURCE_ID, SQL_ID, TABLE_ID, VIEW_ID,
)
from panel_flowdash import DashboardModel, DashboardStore, resolve_identity

from ..source import AnnDataSource
from .nodes import MANIFOLD_MAP_ID, SELECTION_TABLE

COMPONENT_LIBRARY = "lumen_anndata.flowdash"

# scanpy's small example datasets, by the name the command line accepts.
EXAMPLES = ("pbmc68k_reduced", "pbmc3k_processed")

STARTER_ID = "single-cell-overview"


def load_adata(data: str | os.PathLike | Any) -> Any:
    """An AnnData object for an .h5ad path, the name of a scanpy example dataset, or an AnnData object."""
    import anndata as ad

    if isinstance(data, ad.AnnData):
        return data
    if str(data) in EXAMPLES:
        import scanpy as sc

        return getattr(sc.datasets, str(data))()
    path = Path(data)
    if path.suffix != ".h5ad":
        raise ValueError(f"Expected an .h5ad file or one of {', '.join(EXAMPLES)}, got {str(data)!r}.")
    return ad.read_h5ad(path)


def anndata_sources(data: Sequence[str | os.PathLike | Any]) -> list[AnnDataSource]:
    """A source per dataset; one query never narrows another's, as views share them."""
    return [AnnDataSource(adata=load_adata(item), sticky_selection=False) for item in data]


# Columns a first map is coloured by, best first: annotated cell types, then clusters.
_GROUP_HINTS = ("cell_type", "celltype", "lineage", "label", "leiden", "louvain", "cluster")


def _group_column(adata: Any) -> str | None:
    """The obs column a first map is coloured and counted by: a categorical one with a few groups."""
    candidates = [
        column for column in adata.obs.columns
        if isinstance(adata.obs[column].dtype, pd.CategoricalDtype)
        and 2 <= adata.obs[column].nunique() <= 40
        and not column.endswith("_ontology_term_id")
    ]
    for hint in _GROUP_HINTS:
        for column in candidates:
            if hint in column.lower():
                return column
    return candidates[0] if candidates else None


def starter_dashboard(source: AnnDataSource, ref: str = "data") -> dict[str, Any]:
    """
    A dashboard over ``source``: its cells on the default embedding, and the
    cells per group of those selected on it (all of them until some are).
    """
    adata = source.get("obs", return_type="anndata")
    group = _group_column(adata)
    default = adata.uns.get("default_embedding")
    reduction = default if default in adata.obsm else ("X_umap" if "X_umap" in adata.obsm else "")
    items: list[dict[str, Any]] = [
        {"instance_id": "dataset", "component_id": SOURCE_ID, "x": 0, "y": 160, "config": {"ref": ref}},
        {"instance_id": "cells", "component_id": TABLE_ID, "x": 280, "y": 160, "config": {"table": "obs"}},
        {
            "instance_id": "cell_map", "component_id": MANIFOLD_MAP_ID, "x": 560, "y": 60,
            "config": {k: v for k, v in (("reduction", reduction), ("color_by", group)) if v},
        },
    ]
    edges = [
        {"source": "dataset", "source_port": "source", "target": "cells", "target_port": "source"},
        {"source": "cells", "source_port": "pipeline", "target": "cell_map", "target_port": "pipeline"},
    ]
    tiles = [{"instance_id": "cell_map", "index": 0, "width": 100 if group is None else 62, "height": 560}]
    if group is not None:
        items += [
            # Counts the cells selected on the map, every cell until something is.
            {
                "instance_id": "cells_per_group", "component_id": SQL_ID, "x": 860, "y": 60,
                "config": {
                    "sql": (
                        f'SELECT "{group}", COUNT(*) AS cells FROM {SELECTION_TABLE} '
                        "GROUP BY 1 ORDER BY 2 DESC"
                    ),
                    "table": "cells_per_group",
                },
            },
            {
                "instance_id": "cells_per_group_chart", "component_id": VIEW_ID, "x": 1140, "y": 60,
                "config": {"spec": {
                    "mark": "bar",
                    "encoding": {
                        "y": {"field": group, "type": "nominal", "sort": "-x", "title": None},
                        "x": {"field": "cells", "type": "quantitative"},
                    },
                }},
            },
        ]
        edges += [
            {
                "source": "cell_map", "source_port": "selected_source",
                "target": "cells_per_group", "target_port": "source",
            },
            {
                "source": "cells_per_group", "source_port": "pipeline",
                "target": "cells_per_group_chart", "target_port": "pipeline",
            },
        ]
        tiles.append({"instance_id": "cells_per_group_chart", "index": 1, "width": 38, "height": 560})
    title = adata.uns.get("title")
    return {
        "version": 4,
        "title": "Single-cell overview" if not title else f"Overview: {title[:60]}",
        "items": items,
        "edges": edges,
        # Source, table and query nodes stay on the canvas, off the dashboard.
        "tile_layout": [{**tile, "visible": True} for tile in tiles] + [
            {"instance_id": instance, "index": len(tiles) + i, "width": 50, "height": 300, "visible": False}
            for i, instance in enumerate(
                item["instance_id"] for item in items
                if item["instance_id"] not in {tile["instance_id"] for tile in tiles}
            )
        ],
    }


def seed_starter(store: DashboardStore, source: AnnDataSource, owner: str, ref: str = "data") -> bool:
    """
    Add the starter dashboard over ``source``, served as ``ref``. One the store
    has is kept, since it may have been edited, unless its source node names a
    ref the app no longer serves, which leaves it unable to show anything.
    """
    existing = store.find_by_id_or_title(STARTER_ID)
    if existing is not None:
        refs = [
            (getattr(item, "config", None) or {}).get("ref")
            for item in existing.items if getattr(item, "component_id", None) == SOURCE_ID
        ]
        if ref in refs:
            return False
    data = starter_dashboard(source, ref)
    store.save_dashboard(DashboardModel.from_dict({**data, "dashboard_id": STARTER_ID, "user_id": owner}))
    return True


def build_routes(
    data: Sequence[str | os.PathLike | Any] = ("pbmc68k_reduced",),
    *,
    db_path: str | os.PathLike | None = None,
    agent_model: str | Sequence[str] | None = None,
    title: str = "Lumen AnnData",
    seed: bool = True,
    **params: Any,
) -> dict[str, Any]:
    """
    The routes of Lumen's FlowDash app over AnnData datasets, with
    lumen-anndata's nodes next to Lumen's and, given ``agent_model``, the
    AnnData tools on Lumen's assistant.

    Parameters
    ----------
    data
        .h5ad paths, names of scanpy example datasets or AnnData objects.
        Source nodes name them by the refs Lumen gives them.
    db_path
        The SQLite database dashboards are saved in; ``dashboards.db`` in
        the working directory by default.
    agent_model
        A pydantic-ai model, e.g. ``"openai:gpt-6-luna"``, or several the
        person using the agent may pick from (``id`` or ``id=label``).
    seed
        Whether to add a starter dashboard over the first dataset.
    **params
        Further parameters of Lumen's ``build_routes``.
    """
    import lumen.agent

    from lumen.flowdash import app as lumen_app

    from .agent import create_agent

    sources = anndata_sources(data)
    if seed and sources:
        ref, first = next(iter(lumen_app.source_refs(sources).items()))
        store = DashboardStore(str(Path(db_path) if db_path else Path.cwd() / "dashboards.db"))
        seed_starter(store, first, resolve_identity().user, ref)
    # Lumen builds the app, so it serves what lumen serve --flowdash does; only
    # its assistant is made with the AnnData tools and instructions added.
    lumen_agent = lumen.agent.create_agent
    with mock.patch.object(lumen.agent, "create_agent", partial(create_agent, base=lumen_agent)):
        return lumen_app.build_routes(
            sources, db_path=db_path, agent_model=agent_model, title=title,
            component_libraries=[COMPONENT_LIBRARY, *params.pop("component_libraries", [])],
            **params,
        )


def serve(
    data: Sequence[str | os.PathLike | Any] = ("pbmc68k_reduced",),
    *,
    port: int = 5006,
    show: bool = False,
    **kwargs: Any,
) -> None:
    """Serve :func:`build_routes` until interrupted."""
    import panel as pn

    routes = build_routes(data, **kwargs)
    pn.serve(routes, port=port, show=show, title=kwargs.get("title", "Lumen AnnData"), location=True)
