"""lumen-anndata's FlowDash nodes, agent tools and starter dashboard."""

from types import SimpleNamespace

import pytest

pytest.importorskip("lumen.flowdash")
pytest.importorskip("panel_flowdash")

import scanpy as sc

from lumen.agent import _graph
from lumen.flowdash import COMPONENTS as LUMEN_COMPONENTS, SOURCE_ID, TABLE_ID
from lumen.flowdash.nodes import bind_source, filtered_pipeline, table_pipeline
from lumen.sources.duckdb import DuckDBSource
from panel_flowdash import FlowDash
from panel_flowdash.agent import DashboardOps, FlowDashDeps
from pydantic_ai import ModelRetry

from lumen_anndata.flowdash import (
    CLUSTERMAP_ID, COMPONENTS, LEIDEN_ID, MANIFOLD_MAP_ID, TRACKSPLOT_ID,
    ClustermapNode, LeidenNode, ManifoldMapNode, TracksplotNode,
    nodes as nodes_module, pipeline_adata,
)
from lumen_anndata.flowdash.agent import (
    add_clustermap, add_leiden, add_manifold_map, add_selection_sql,
    add_tracksplot, describe_cells,
)
from lumen_anndata.flowdash.app import starter_dashboard
from lumen_anndata.source import AnnDataSource


@pytest.fixture(scope="module")
def adata():
    return sc.datasets.pbmc68k_reduced()


@pytest.fixture
def source(adata):
    return AnnDataSource(adata=adata.copy(), sticky_selection=False)


@pytest.fixture
def cells(source):
    return table_pipeline(source, "obs", 100_000)


@pytest.fixture
def dendritic(cells):
    return filtered_pipeline(cells, "bulk_labels", ["Dendritic"])


def test_pipeline_adata_keeps_the_cells_a_filter_keeps(cells, dendritic):
    assert pipeline_adata(cells).n_obs == 700
    subset = pipeline_adata(dendritic)
    assert subset.n_obs == 240 and set(subset.obs["bulk_labels"]) == {"Dendritic"}


def test_pipeline_adata_ignores_the_row_limit(source):
    assert pipeline_adata(table_pipeline(source, "obs", 10)).n_obs == 700


def test_pipeline_adata_needs_an_anndata_source():
    source = DuckDBSource(uri=":memory:", tables={"t": "SELECT 1 AS a"})
    with pytest.raises(ValueError, match="over an AnnData source"):
        pipeline_adata(table_pipeline(source, "t"))


@pytest.mark.parametrize("sticky", [True, False])
def test_a_filtered_query_narrows_later_ones_only_when_sticky(adata, sticky):
    source = AnnDataSource(adata=adata.copy(), sticky_selection=sticky)
    filtered_pipeline(table_pipeline(source, "obs"), "bulk_labels", ["Dendritic"]).data
    assert len(source.get("obs")) == (240 if sticky else 700)


def test_manifold_map_passes_every_cell_until_a_selection(cells):
    import holoviews as hv

    node = ManifoldMapNode(pipeline=cells, color_by="bulk_labels")
    assert node._error is None and node.selected() is cells
    view = node._object
    node._select(view, hv.dim(str(view.current_kdims()[0])) > 0)
    selected = node.selected()
    expected = int((pipeline_adata(cells).obsm["X_umap"][:, 0] > 0).sum())
    assert pipeline_adata(selected).n_obs == len(selected.data) == expected
    node._select(view, None)
    assert node.selected() is cells


def _counted(source):
    return int(source.execute("SELECT COUNT(*) AS n FROM selected_cells")["n"].iloc[0])


def test_manifold_map_selected_source_follows_the_selection(cells, dendritic):
    import holoviews as hv

    node = ManifoldMapNode(pipeline=cells)
    assert _counted(node.selected_source()) == 700
    view = node._object
    node._select(view, hv.dim(str(view.current_kdims()[0])) > 0)
    assert _counted(node.selected_source()) == len(node.selected().data) < 700
    node._select(view, None)
    assert _counted(node.selected_source()) == 700
    assert _counted(ManifoldMapNode(pipeline=dendritic).selected_source()) == 240


def test_manifold_map_reports_an_unknown_colour(cells):
    node = ManifoldMapNode(pipeline=cells, color_by="nope")
    assert node._error.startswith("No color by 'nope'")


def test_leiden_clusters_the_filtered_cells(dendritic):
    node = LeidenNode(pipeline=dendritic, resolution=0.5)
    assert node._error is None
    clustered = node.clustered()
    assert "leiden_0.5" in clustered.data.columns and len(clustered.data) == 240
    assert node.__flowdash_preview__().endswith("of 240 cells in column leiden_0.5")


def test_tracksplot_ranks_the_groups_it_is_given(dendritic):
    clustered = LeidenNode(pipeline=dendritic, resolution=0.5).clustered()
    node = TracksplotNode(pipeline=clustered, groupby="leiden_0.5", n_genes=2)
    assert node._error is None
    assert node.__flowdash_preview__().startswith("Top 2 genes of each of")
    missing = TracksplotNode(pipeline=clustered, groupby="nope")
    assert missing._error.startswith("No obs column 'nope'")


def test_clustermap_refuses_too_many_cells(cells, dendritic, monkeypatch):
    monkeypatch.setattr(nodes_module, "CLUSTERMAP_MAX_CELLS", 300)
    assert "too many to cluster" in ClustermapNode(pipeline=cells)._error
    assert ClustermapNode(pipeline=dendritic)._error is None


def test_expression_matrices_are_told_apart(adata):
    assert nodes_module._matrix_kind(adata.X) == "scaled"
    assert nodes_module._matrix_kind(adata.raw.X) == "log"
    assert nodes_module._matrix_kind((abs(adata.X) * 100).round()) == "counts"


def test_starter_dashboard_colours_by_annotations_and_hides_the_plumbing(source):
    dashboard = starter_dashboard(source)
    items = {item["instance_id"]: item for item in dashboard["items"]}
    assert items["cell_map"]["config"] == {"reduction": "X_umap", "color_by": "bulk_labels"}
    # The counts follow what is selected on the map.
    assert "FROM selected_cells" in items["cells_per_group"]["config"]["sql"]
    assert {
        "source": "cell_map", "source_port": "selected_source",
        "target": "cells_per_group", "target_port": "source",
    } in dashboard["edges"]
    shown = {t["instance_id"] for t in dashboard["tile_layout"] if t["visible"]}
    assert shown == {"cell_map", "cells_per_group_chart"}


@pytest.fixture
def ctx(source):
    editor = FlowDash(components={**LUMEN_COMPONENTS, **COMPONENTS})
    ops = DashboardOps(editor)
    bind_source(editor, source, "data")
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    ops.add_node(TABLE_ID, {"table": "obs"}, name="cells")
    ops.connect("data.source", "cells.source")
    return SimpleNamespace(deps=FlowDashDeps(ops))


def _components(ctx):
    return {node["name"]: node["component"] for node in ctx.deps.ops.nodes()}


def test_describe_cells(ctx):
    text = describe_cells(ctx, "cells")
    assert "700 cells and 765 genes." in text
    assert "X_umap" in text and "bulk_labels (10 groups)" in text


def test_the_tools_place_and_wire_the_nodes(ctx):
    assert "selected" in add_manifold_map(ctx, "cells", "cell_map", color_by="bulk_labels")
    assert "clusters in the obs column 'leiden_1.0'" in add_leiden(ctx, "cell_map", "clusters")
    assert "tracksplot 'markers'" in add_tracksplot(ctx, "clusters", "markers", groupby="leiden_1.0")
    assert _components(ctx)["cell_map"] == MANIFOLD_MAP_ID
    assert _components(ctx)["clusters"] == LEIDEN_ID
    assert _components(ctx)["markers"] == TRACKSPLOT_ID


def test_the_tools_check_what_they_are_given(ctx, monkeypatch):
    with pytest.raises(ModelRetry, match="No node 'nothing'"):
        add_manifold_map(ctx, "nothing", "map")
    with pytest.raises(ModelRetry, match="outputs no cells"):
        add_leiden(ctx, "data", "clusters")
    with pytest.raises(ModelRetry, match="No obs column or gene 'nope'"):
        add_manifold_map(ctx, "cells", "map", color_by="nope")
    with pytest.raises(ModelRetry, match="No embedding 'X_tsne'"):
        add_manifold_map(ctx, "cells", "map", reduction="X_tsne")
    monkeypatch.setattr("lumen_anndata.flowdash.agent.CLUSTERMAP_MAX_CELLS", 300)
    with pytest.raises(ModelRetry, match="700 cells"):
        add_clustermap(ctx, "cells", "heatmap")
    assert "heatmap" not in _components(ctx) and CLUSTERMAP_ID not in _components(ctx).values()


async def test_a_sql_node_on_a_map_follows_its_selection(ctx):
    import holoviews as hv

    add_manifold_map(ctx, "cells", "cell_map")
    sql = "SELECT bulk_labels, COUNT(*) AS cells FROM selected_cells GROUP BY 1"
    assert "follows the selection" in await add_selection_sql(ctx, "cell_map", "per_label", sql)
    ops = ctx.deps.ops
    total = lambda: int(_graph.output(ops, "per_label", "data")["cells"].sum())  # noqa: E731
    assert total() == 700
    node = ops.editor._component_instances["cell_map"]
    node._select(node._object, hv.dim(str(node._object.current_kdims()[0])) > 0)
    assert total() == len(node.selected().data) < 700
    with pytest.raises(ModelRetry, match="No manifold map 'cells'"):
        await add_selection_sql(ctx, "cells", "nope", sql)
