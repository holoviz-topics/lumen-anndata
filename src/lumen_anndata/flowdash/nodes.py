"""
AnnData analyses as FlowDash nodes, next to Lumen's own.

Each node reads a Lumen pipeline over an :class:`~lumen_anndata.source.AnnDataSource`,
such as a Lumen table node on ``obs``, and works on the cells that pipeline
keeps: filter nodes and global filters upstream narrow the AnnData object the
node computes on, while the row limit of a table node does not.
"""
from __future__ import annotations

from typing import Any

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param

from hv_anndata import register as register_anndata_interface
from lumen.flowdash.nodes import (
    PIPELINE_TYPE, LumenNode, effective_limit, table_pipeline,
)
from lumen.pipeline import Pipeline
from lumen.transforms.base import Iloc
from lumen.transforms.sql import SQLFilter, SQLLimit
from lumen.util import get_dataframe_schema
from panel_flowdash import register
from panel_flowdash.dataflow_engine import failure_text
from panel_flowdash.util import is_failure

from ..operations import LeidenOperation
from ..source import AnnDataSource

register_anndata_interface()

MANIFOLD_MAP_ID = "AnnData/manifold_map"
LEIDEN_ID = "AnnData/leiden"
TRACKSPLOT_ID = "AnnData/tracksplot"
CLUSTERMAP_ID = "AnnData/clustermap"

# The table of a manifold map's selection in the source it derives.
SELECTION_TABLE = "obs_linked_selection"

# A clustermap clusters every cell it is given; past this many it is too slow to be useful.
CLUSTERMAP_MAX_CELLS = 20_000

# The output port of each node that carries a pipeline, for Lumen's agent tools.
PIPELINE_PORTS = {MANIFOLD_MAP_ID: "selected", LEIDEN_ID: "clustered"}


def pipeline_adata(pipeline: Any) -> Any:
    """
    The AnnData object of the cells ``pipeline`` keeps. Filters narrow it
    to the matching cells; a row limit does not, since it only caps how
    many rows a table shows.
    """
    if is_failure(pipeline):
        raise ValueError(f"The input failed: {failure_text(pipeline)}")
    if not isinstance(pipeline, Pipeline):
        raise ValueError(f"Expected a Lumen pipeline, got {type(pipeline).__name__}.")
    source = pipeline.source
    if not isinstance(source, AnnDataSource):
        raise ValueError(
            f"Expected a pipeline over an AnnData source; {pipeline.table!r} comes from a "
            f"{type(source).__name__}."
        )
    adata = source.get(pipeline.table, return_type="anndata")
    sql = [t for t in pipeline.sql_transforms if not isinstance(t, SQLLimit)]
    transforms = [t for t in pipeline.transforms if not isinstance(t, Iloc)]
    if not (sql or transforms or pipeline.filters or pipeline.pipeline is not None):
        return adata
    rows = pipeline.clone(sql_transforms=sql, transforms=transforms).data
    if "obs_id" not in rows.columns:
        raise ValueError(
            f"{pipeline.table!r} has no obs_id column, so its rows cannot be matched to cells; "
            "read the obs table, or a query that keeps obs_id."
        )
    return adata[adata.obs_names.isin(rows["obs_id"].astype(str))]


def obs_pipeline(source: AnnDataSource, table: str, frame: pd.DataFrame | None, limit: int) -> Pipeline:
    """A pipeline over ``table`` of ``source`` with the schema of ``frame``, when given."""
    if frame is None:
        return table_pipeline(source, table, limit)
    # Describing the frame at hand saves the per-column queries get_schema would
    # run, each re-running a selection's long IN list.
    schema = get_dataframe_schema(frame)["items"]["properties"]
    schema["__len__"] = len(frame)
    return Pipeline(
        source=source, table=table, schema=schema, name=table,
        sql_transforms=[SQLLimit(limit=limit, write=source.dialect)],
    )


def _matrix_kind(matrix: Any) -> str:
    """Whether an expression matrix holds raw "counts", "log" expression or "scaled" values."""
    from scipy.sparse import issparse

    sample = matrix[: min(matrix.shape[0], 500)]
    values = sample.data if issparse(sample) else np.asarray(sample).ravel()
    if values.size == 0:
        return "log"
    if values.min() < 0:
        return "scaled"
    if values.max() > 20 and np.allclose(values, np.round(values)):
        return "counts"
    return "log"


# var columns holding gene symbols, as CELLxGENE (feature_name) and others name them.
_SYMBOL_COLUMNS = ("feature_name", "gene_symbols", "gene_symbol", "gene_name", "symbol")


def _symbol_column(adata: Any) -> str | None:
    """The var column naming genes by symbol, when var_names are ids such as Ensembl's."""
    return next((c for c in _SYMBOL_COLUMNS if c in adata.var.columns), None)


def _option(objects: Any, value: str) -> Any:
    """The option of a selector whose name or value is ``value``, or None."""
    if not value:
        return None
    items = objects.items() if isinstance(objects, dict) else ((o, o) for o in objects or [])
    for name, option in items:
        if value in (name, option):
            return option
    return None


class AnnDataNode(LumenNode):
    """A node computing on the cells of a pipeline over an AnnData source."""

    __abstract = True

    pipeline = param.Parameter(doc="The upstream pipeline, over an AnnData source.")

    _waiting = "Waiting for a pipeline over an AnnData source"

    def _adata(self):
        if self.pipeline is None:
            return None
        return pipeline_adata(self.pipeline)

    def _body(self):
        if self._object is None:
            return pn.pane.Markdown(self._waiting)
        return self._object

    def __flowdash_preview__(self, output: str | None = None) -> Any:
        if self._error is not None or self._object is None:
            return None
        return self._summary()

    def _summary(self) -> str:
        return ""


@register(
    component=True, page=False, title="AnnData manifold map", icon="scatter_plot", tags=["view"],
    description=(
        "A UMAP, t-SNE or PCA embedding of the cells of a pipeline over an AnnData source "
        "(e.g. a Lumen table node on obs), coloured by an obs column or a gene. Lasso or box "
        "selecting cells sets the selected output, a pipeline of just those cells (every cell "
        "until something is selected), which feeds other views, Leiden clustering, a "
        "tracksplot or a clustermap."
    ),
    config=["reduction", "color_by", "show_labels"],
    default_size={"width": 760, "height": 540},
    requires=[{"key": "pipeline", "type": PIPELINE_TYPE, "required": True}],
    provides=[{"key": "selected", "type": PIPELINE_TYPE}],
)
class ManifoldMapNode(AnnDataNode):

    reduction = param.String(default="", doc="""
        The embedding to show, a key of adata.obsm such as X_umap; the
        dataset's default if empty.""")

    color_by = param.String(default="", doc="""
        The obs column or gene to colour the cells by; the map's default if
        empty.""")

    show_labels = param.Boolean(default=False, doc="""
        Whether to label the groups of a categorical colour column.""")

    _selection = param.Parameter(default=None, precedence=-1)

    def __init__(self, **params):
        self._adata_shown = None
        super().__init__(**params)

    @param.depends("pipeline", "reduction", "color_by", "show_labels", watch=True, on_init=True)
    def _refresh(self, *events):
        self._selection = None
        super()._refresh()

    def _resolve(self):
        from hv_anndata.plotting import ManifoldMap

        adata = self._adata()
        self._adata_shown = adata
        if adata is None:
            return None
        if not len(adata.obsm):
            raise ValueError("The cells have no embeddings in adata.obsm to show.")
        links = hv.link_selections.instance()
        view = ManifoldMap(adata=adata, ls=links, show_labels=self.show_labels, responsive=True)
        for name in ("reduction", "color_by"):
            value = getattr(self, name)
            option = _option(view.param[name].objects, value)
            if value and option is None:
                raise ValueError(
                    f"No {name.replace('_', ' ')} {value!r}. Pick one of: "
                    f"{', '.join(map(str, view.param[name].objects))}."
                )
            if option is not None:
                setattr(view, name, option)
        links.param.watch(lambda event: self._select(view, event.new), "selection_expr")
        return view

    def _select(self, view, expr):
        adata = self._adata_shown
        if expr is None or adata is None:
            self._selection = None
            return
        mask = expr.apply(hv.Dataset(adata, view.current_kdims()))
        ids = [str(obs_id) for obs_id in adata.obs_names[mask]]
        source = self.pipeline.source
        base = 'SELECT * FROM "obs"'
        tables = dict(source.tables)
        tables[SELECTION_TABLE] = (
            SQLFilter(conditions=[("obs_id", ids)]).apply(base) if ids
            else f"SELECT * FROM ({base}) WHERE FALSE"
        )
        derived = source.create_sql_expr_source(tables)
        frame = derived.get(SELECTION_TABLE)
        self._selection = obs_pipeline(
            derived, SELECTION_TABLE, frame, effective_limit(self.editor, None)
        )

    @param.output(param.Parameter)
    @param.depends("_object", "_error", "_selection")
    def selected(self):
        if self._error is not None:
            return self._value()
        return self._selection if self._selection is not None else self.pipeline

    def _summary(self) -> str:
        view, adata = self._object, self._adata_shown
        return f"{adata.n_obs:,} cells on {view.reduction}, coloured by {view.color_by}"


@register(
    component=True, page=False, title="AnnData Leiden clustering", icon="hub", tags=["transform"],
    description=(
        "Leiden clustering of the cells of a pipeline over an AnnData source, computing "
        "neighbours first if the data has none. The clustered output is a pipeline over the cells (obs) with "
        "a new categorical column, leiden_<resolution> by default, for manifold maps, "
        "tracksplots, charts and further nodes. Clustering a filtered or selected subset "
        "clusters only those cells."
    ),
    config=["resolution", "n_iterations", "random_state", "key_added"],
    default_size={"width": 360, "height": 320},
    requires=[{"key": "pipeline", "type": PIPELINE_TYPE, "required": True}],
    provides=[{"key": "clustered", "type": PIPELINE_TYPE}],
)
class LeidenNode(AnnDataNode):

    resolution = param.Number(default=1.0, bounds=(0, None), doc="""
        Higher values give more, smaller clusters.""")

    n_iterations = param.Integer(default=2, doc="""
        Iterations of the Leiden algorithm; -1 iterates until it converges.""")

    random_state = param.Integer(default=0, allow_None=True, doc="""
        Seed, so the same input gives the same clusters.""")

    key_added = param.String(default="leiden_{resolution:.1f}", doc="""
        The obs column the clusters are written to; {resolution} is
        replaced by the resolution.""")

    _reads_source = False

    @param.depends(
        "pipeline", "resolution", "n_iterations", "random_state", "key_added",
        watch=True, on_init=True,
    )
    def _refresh(self, *events):
        super()._refresh()

    @property
    def column(self) -> str:
        return self.key_added.format(resolution=self.resolution)

    def _resolve(self):
        adata = self._adata()
        if adata is None:
            return None
        operation = LeidenOperation.instance(
            resolution=self.resolution, n_iterations=self.n_iterations,
            random_state=self.random_state, key_added=self.key_added,
        )
        adata = operation(adata.copy())
        source = self.pipeline.source
        # The clustered object goes into the new source as it is; listing the
        # operation as well would cluster again on every read. Tables derived
        # from the old object, such as a query node's result, do not carry over.
        clustered = source.create_sql_expr_source(tables={}, adata=adata)
        return obs_pipeline(clustered, "obs", None, effective_limit(self.editor, None))

    @param.output(param.Parameter)
    @param.depends("_object", "_error")
    def clustered(self):
        return self._value()

    def _body(self):
        if self._object is None:
            return pn.pane.Markdown(self._waiting)
        sizes = self._sizes()
        table = sizes.rename_axis(self.column).reset_index(name="cells")
        return pn.Column(
            pn.pane.Markdown(self._summary()),
            pn.widgets.Tabulator(table, disabled=True, show_index=False, sizing_mode="stretch_both"),
            sizing_mode="stretch_both",
        )

    def _sizes(self) -> pd.Series:
        adata = self._object.source.get("obs", return_type="anndata")
        return adata.obs[self.column].value_counts().sort_index()

    def _summary(self) -> str:
        sizes = self._sizes()
        return f"{len(sizes)} clusters of {int(sizes.sum()):,} cells in column {self.column}"


@register(
    component=True, page=False, title="AnnData tracksplot", icon="stacked_line_chart", tags=["view"],
    description=(
        "A scanpy tracksplot of the top marker genes of each group of an obs column, e.g. "
        "clusters, for the cells of a pipeline over an AnnData source. Ranks the genes with "
        "rank_genes_groups first unless the data already ranks them by that column."
    ),
    config=["groupby", "n_genes"],
    default_size={"width": 960, "height": 440},
    requires=[{"key": "pipeline", "type": PIPELINE_TYPE, "required": True}],
)
class TracksplotNode(AnnDataNode):

    groupby = param.String(default="", doc="""
        The categorical obs column whose groups to compare, e.g. a Leiden
        cluster column.""")

    n_genes = param.Integer(default=3, bounds=(1, None), doc="""
        Top genes shown per group.""")

    @param.depends("pipeline", "groupby", "n_genes", watch=True, on_init=True)
    def _refresh(self, *events):
        super()._refresh()

    def _resolve(self):
        import matplotlib

        matplotlib.use("Agg")
        import scanpy as sc

        adata = self._adata()
        if adata is None or not self.groupby:
            return None
        if self.groupby not in adata.obs.columns:
            raise ValueError(
                f"No obs column {self.groupby!r}. Its columns are: {', '.join(adata.obs.columns)}."
            )
        adata = adata.copy()
        if not isinstance(adata.obs[self.groupby].dtype, pd.CategoricalDtype):
            adata.obs[self.groupby] = adata.obs[self.groupby].astype("category")
        # Genes are ranked and drawn on log expression: in X, else in raw (when
        # X is scaled), else X normalized from counts. Counts would rank genes
        # by library size and scaled values break the fold changes.
        kind = _matrix_kind(adata.X)
        use_raw = kind != "log" and adata.raw is not None and _matrix_kind(adata.raw.X) == "log"
        ranked = adata.uns.get("rank_genes_groups", {}).get("params", {}).get("groupby")
        if ranked != self.groupby:
            if kind == "counts" and not use_raw:
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
            adata.uns.setdefault("log1p", {"base": None})
            sc.tl.rank_genes_groups(adata, groupby=self.groupby, use_raw=use_raw)
        symbols = _symbol_column(adata)
        if symbols is not None:
            # CELLxGENE's feature_name repeats the Ensembl id after most symbols.
            adata.var["_symbol"] = adata.var[symbols].astype(str).str.replace(r"_ENSG\d+$", "", regex=True)
            symbols = "_symbol"
        axes = sc.pl.rank_genes_groups_tracksplot(
            adata, n_genes=self.n_genes, use_raw=use_raw, gene_symbols=symbols, show=False
        )
        figure = axes["track_axes"][0].figure
        self._groups = adata.obs[self.groupby].nunique()
        # PNG, as an SVG would hold a shape per cell and track.
        return pn.pane.Matplotlib(figure, dpi=144, tight=True, sizing_mode="stretch_both")

    def _summary(self) -> str:
        return f"Top {self.n_genes} genes of each of {self._groups} groups of {self.groupby}"


@register(
    component=True, page=False, title="AnnData clustermap", icon="grid_on", tags=["view"],
    description=(
        f"A clustered heatmap of mean expression by group for the cells of a pipeline over an "
        f"AnnData source, with its own controls for the grouping and genes. Clusters every cell "
        f"it is given, so it takes at most {CLUSTERMAP_MAX_CELLS:,}: feed it a manifold map's "
        f"selection or a filtered table."
    ),
    config=["cmap"],
    default_size={"width": 760, "height": 540},
    requires=[{"key": "pipeline", "type": PIPELINE_TYPE, "required": True}],
)
class ClustermapNode(AnnDataNode):

    cmap = param.String(default="viridis", doc="The colormap of the heatmap.")

    @param.depends("pipeline", "cmap", watch=True, on_init=True)
    def _refresh(self, *events):
        super()._refresh()

    def _resolve(self):
        from hv_anndata.plotting import ClusterMap

        adata = self._adata()
        if adata is None:
            return None
        if adata.n_obs > CLUSTERMAP_MAX_CELLS:
            raise ValueError(
                f"{adata.n_obs:,} cells are too many to cluster; select or filter at most "
                f"{CLUSTERMAP_MAX_CELLS:,} first, e.g. with a manifold map's selection."
            )
        view = ClusterMap(adata=adata)
        option = _option(view.param.cmap.objects, self.cmap)
        if option is None:
            raise ValueError(f"No colormap {self.cmap!r}.")
        view.cmap = option
        self._cells = adata.n_obs
        return view

    def _summary(self) -> str:
        return f"Clustermap of {self._cells:,} cells"


COMPONENTS = {
    MANIFOLD_MAP_ID: ManifoldMapNode,
    LEIDEN_ID: LeidenNode,
    TRACKSPLOT_ID: TracksplotNode,
    CLUSTERMAP_ID: ClustermapNode,
}
