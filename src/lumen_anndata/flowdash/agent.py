"""Tools that let Lumen's FlowDash agent place lumen-anndata's nodes."""
from __future__ import annotations

from typing import Any

import pandas as pd

from lumen.agent import _graph
from lumen.agent.authoring import _place, _test_run
from lumen.agent.data import lint
from lumen.ai.agents.sql import format_exploration_result
from lumen.flowdash import SQL_ID
from lumen.flowdash.nodes import effective_limit, primed_result
from lumen.pipeline import Pipeline
from panel_flowdash.agent import FlowDashDeps, OperationError
from pydantic_ai import FunctionToolset, ModelRetry, RunContext

from .nodes import (
    CLUSTERMAP_ID, CLUSTERMAP_MAX_CELLS, LEIDEN_ID, MANIFOLD_MAP_ID,
    PIPELINE_PORTS, TRACKSPLOT_ID, pipeline_adata,
)

# Lumen's own tools (add_chart, the node listing) find pipelines through these ports.
_graph.PIPELINE_PORTS.update(PIPELINE_PORTS)

INSTRUCTIONS = f"""
## Single-cell data (AnnData)
The source is a single-cell dataset. Its obs table has one row per cell (obs_id, the cell's
metadata such as clusters, samples and QC metrics), var one row per gene (var_id) and the
obsm_* tables the embeddings. Answer questions about cell counts and metadata with SQL on
obs as usual.
For single-cell analyses, use the AnnData tools on a node that outputs the cells: a table
node of obs, a filter node, or a SQL node whose query keeps obs_id.
- describe_cells lists the embeddings, the categorical obs columns and what is precomputed.
- add_manifold_map shows a UMAP, t-SNE or PCA embedding coloured by an obs column or a gene.
  Its selected output carries the cells the user lassoes on it (every cell until they do),
  so place nodes about "the selected cells" on that output.
- add_selection_sql places a SQL node summarizing a map's selected cells, which it reads as
  the table selected_cells; chart it with add_chart. Use it for counts or statistics that
  should follow the lasso, e.g. cells per type in the selection.
- add_leiden clusters cells; build on its clustered output, whose new column holds the
  clusters.
- add_tracksplot shows the top marker genes of each group of an obs column.
- add_clustermap shows mean expression by group, for at most {CLUSTERMAP_MAX_CELLS:,} cells: give it a
  map's selection or a filtered node.
Ground answers about scanpy in what the tools return, and say so when they cannot tell.
"""


def _cells(ctx: RunContext[FlowDashDeps], data: str) -> tuple[str, Any]:
    """The port of node ``data`` that carries cells, and their AnnData object."""
    ops = ctx.deps.ops
    if data not in _graph.nodes(ops):
        raise ModelRetry(f"No node {data!r}.")
    port = _graph.pipeline_port(ops, data)
    if port is None:
        raise ModelRetry(
            f"{data!r} outputs no cells; pass a table, filter or SQL node of obs, a manifold "
            "map (its selection) or a Leiden node."
        )
    try:
        pipeline = _graph.output(ops, data, port)
    except OperationError as exc:
        raise ModelRetry(str(exc)) from exc
    if not isinstance(pipeline, Pipeline):
        raise ModelRetry(f"{data!r} has no result yet.")
    try:
        return port, pipeline_adata(pipeline)
    except ValueError as exc:
        raise ModelRetry(str(exc)) from exc


def _check_column(adata: Any, column: str, genes: bool = False) -> None:
    if column in adata.obs.columns or (genes and column in adata.var_names):
        return
    kind = "obs column or gene" if genes else "obs column"
    raise ModelRetry(f"No {kind} {column!r}. The obs columns are: {', '.join(adata.obs.columns)}.")


def _categorical(adata: Any) -> list[str]:
    """The obs columns whose values form groups, with their number of groups."""
    lines = []
    for column in adata.obs.columns:
        values = adata.obs[column]
        groups = values.nunique()
        if isinstance(values.dtype, pd.CategoricalDtype) or (values.dtype == object and groups <= 50):
            lines.append(f"{column} ({groups} groups)")
    return lines


def describe_cells(ctx: RunContext[FlowDashDeps], data: str) -> str:
    """Describe the cells a node outputs: their number, embeddings, grouping columns and precomputed analyses.

    Args:
        data: A table, filter or SQL node of obs, a manifold map or a Leiden node.
    """
    _, adata = _cells(ctx, data)
    lines = [
        f"{adata.n_obs:,} cells and {adata.n_vars:,} genes.",
        f"Embeddings (reduction): {', '.join(adata.obsm) or 'none'}.",
        f"Grouping obs columns: {', '.join(_categorical(adata)) or 'none'}.",
    ]
    numeric = [c for c in adata.obs.columns if pd.api.types.is_numeric_dtype(adata.obs[c])]
    if numeric:
        lines.append(f"Numeric obs columns: {', '.join(numeric)}.")
    precomputed = [key for key in ("neighbors", "rank_genes_groups", "pca") if key in adata.uns]
    if "rank_genes_groups" in adata.uns:
        groupby = adata.uns["rank_genes_groups"].get("params", {}).get("groupby")
        precomputed[precomputed.index("rank_genes_groups")] = f"rank_genes_groups by {groupby}"
    lines.append(f"Precomputed: {', '.join(precomputed) or 'nothing'}.")
    if title := adata.uns.get("title"):
        lines.append(f"Dataset: {title}.")
    return "\n".join(lines)


def add_manifold_map(
    ctx: RunContext[FlowDashDeps], data: str, name: str, color_by: str = "", reduction: str = ""
) -> str:
    """Place a map of a cell embedding (UMAP, t-SNE or PCA) whose lasso selects cells.

    Args:
        data: Node whose cells to show.
        name: Descriptive snake_case node name.
        color_by: Obs column or gene to colour by; the map's default if empty.
        reduction: Embedding in adata.obsm, e.g. X_umap; the dataset's default if empty.
    """
    port, adata = _cells(ctx, data)
    if color_by:
        _check_column(adata, color_by, genes=True)
    if reduction and reduction not in adata.obsm:
        raise ModelRetry(f"No embedding {reduction!r}. The embeddings are: {', '.join(adata.obsm)}.")
    config = {key: value for key, value in (("color_by", color_by), ("reduction", reduction)) if value}
    _place(ctx.deps.ops, MANIFOLD_MAP_ID, name, config, f"{data}.{port}", f"{name}.pipeline")
    return (
        f"Placed manifold map {name!r} of {adata.n_obs:,} cells. Its 'selected' output carries "
        "the cells lassoed on it, every cell until then."
    )


async def add_selection_sql(ctx: RunContext[FlowDashDeps], map: str, name: str, sql: str) -> str:
    """Place a SQL node over the cells selected on a manifold map, following the selection.

    Args:
        map: Manifold map node whose selection to query.
        name: Descriptive snake_case node name.
        sql: One SELECT or WITH query reading the selected cells as the table selected_cells,
            whose columns are those of obs.
    """
    ops = ctx.deps.ops
    if map not in _graph.nodes(ops, MANIFOLD_MAP_ID):
        listed = ", ".join(_graph.nodes(ops, MANIFOLD_MAP_ID)) or "none"
        raise ModelRetry(f"No manifold map {map!r}. The manifold maps are: {listed}.")
    try:
        src = _graph.output(ops, map, "selected_source")
    except OperationError as exc:
        raise ModelRetry(str(exc)) from exc
    limit = effective_limit(ops.editor, None)
    sql, statement, frame = await _test_run(src, sql, limit)
    with primed_result(src, statement, frame):
        _place(ops, SQL_ID, name, {"sql": sql, "table": name}, f"{map}.selected_source", f"{name}.source")
    try:
        frame = _graph.output(ops, name, "data")
    except OperationError as exc:
        ops.remove_node(name)
        raise ModelRetry(str(exc)) from exc
    return (
        f"Placed SQL node {name!r} over the cells selected on {map!r}, which follows the "
        f"selection: {format_exploration_result(frame)}\n\nData checks:\n{lint(frame)}"
    )


def add_leiden(ctx: RunContext[FlowDashDeps], data: str, name: str, resolution: float = 1.0) -> str:
    """Place a Leiden clustering of a node's cells; its clustered output adds the cluster column.

    Args:
        data: Node whose cells to cluster.
        name: Descriptive snake_case node name.
        resolution: Higher values give more, smaller clusters.
    """
    port, _ = _cells(ctx, data)
    _place(ctx.deps.ops, LEIDEN_ID, name, {"resolution": resolution}, f"{data}.{port}", f"{name}.pipeline")
    try:
        clustered = _graph.output(ctx.deps.ops, name, "clustered")
    except OperationError as exc:
        raise ModelRetry(str(exc)) from exc
    column = f"leiden_{resolution:.1f}"
    groups = clustered.source.get("obs", return_type="anndata").obs[column].nunique()
    return f"Placed {name!r}: {groups} clusters in the obs column {column!r} of its clustered output."


def add_tracksplot(
    ctx: RunContext[FlowDashDeps], data: str, name: str, groupby: str, n_genes: int = 3
) -> str:
    """Place a tracksplot of the top marker genes of each group of an obs column.

    Args:
        data: Node whose cells to compare.
        name: Descriptive snake_case node name.
        groupby: Categorical obs column whose groups to compare, e.g. a Leiden column.
        n_genes: Top genes shown per group.
    """
    port, adata = _cells(ctx, data)
    _check_column(adata, groupby)
    config = {"groupby": groupby, "n_genes": n_genes}
    _place(ctx.deps.ops, TRACKSPLOT_ID, name, config, f"{data}.{port}", f"{name}.pipeline")
    return f"Placed tracksplot {name!r} of the top {n_genes} genes of each group of {groupby!r}."


def add_clustermap(ctx: RunContext[FlowDashDeps], data: str, name: str) -> str:
    """Place a clustered heatmap of mean expression by group, for a node with few enough cells.

    Args:
        data: Node whose cells to show, e.g. a manifold map's selection.
        name: Descriptive snake_case node name.
    """
    port, adata = _cells(ctx, data)
    if adata.n_obs > CLUSTERMAP_MAX_CELLS:
        raise ModelRetry(
            f"{data!r} has {adata.n_obs:,} cells; a clustermap takes at most "
            f"{CLUSTERMAP_MAX_CELLS:,}. Use a manifold map's selection or a filtered node."
        )
    _place(ctx.deps.ops, CLUSTERMAP_ID, name, {}, f"{data}.{port}", f"{name}.pipeline")
    return f"Placed clustermap {name!r} of {adata.n_obs:,} cells."


def anndata_toolset() -> FunctionToolset[FlowDashDeps]:
    """describe_cells and the tools placing lumen-anndata's nodes."""
    return FunctionToolset[FlowDashDeps](
        [describe_cells, add_manifold_map, add_selection_sql, add_leiden, add_tracksplot, add_clustermap],
        max_retries=3,
    )


def create_agent(model: Any, **kwargs: Any):
    """Lumen's FlowDash agent with the AnnData tools and instructions."""
    from lumen.agent import create_agent as create_lumen_agent

    instructions = INSTRUCTIONS + (f"\n{kwargs.pop('instructions')}" if kwargs.get("instructions") else "")
    toolsets = [anndata_toolset(), *kwargs.pop("toolsets", [])]
    return create_lumen_agent(model, instructions=instructions, toolsets=toolsets, **kwargs)
