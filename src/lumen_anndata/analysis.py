from functools import partial

import param

from holoviews import Dataset, renderer
from hv_anndata import register
from lumen.ai.analysis import Analysis
from lumen.ai.utils import describe_data
from panel.layout import Column
from panel.pane.markup import Markdown
from panel_material_ui import Button
from param.parameterized import bothmethod

from lumen_anndata.operations import LeidenOperation

from .source import AnnDataSource
from .views import (
    ClustermapPanel, ManifoldMapPanel, RankGenesGroupsTracksplotPanel,
)

renderer("bokeh").webgl = False
register()

#: Name of the table materialized from a manifold-map linked selection.
SELECTION_TABLE = "obs_linked_selection"


class AnnDataAnalysis(Analysis):
    """
    Base class for analyses that operate on AnnData objects.
    This class is used to ensure that the analysis can be applied
    to an AnnDataSource.
    """

    compute_required = param.Boolean(doc="""
        If True, the analysis will run required computations before rendering.""", precedence=-1)

    @classmethod
    async def applies(cls, pipeline) -> bool:
        source = pipeline.source
        if not isinstance(source, AnnDataSource):
            return False
        adata = source.get(pipeline.table, return_type="anndata")
        return adata is not None and len(adata.obsm) > 0


class ManifoldMapVisualization(AnnDataAnalysis):
    """
    Use this to visualize any requests for UMAP, PCA, tSNE results,
    unless explicitly otherwise specified by the user.
    """

    @bothmethod
    def instance(self_or_cls, **params):
        instance = super().instance(**params)
        instance._initialized_selection = False
        instance._chat_message = None
        instance._reset_col = None
        instance._selection_markdown = None
        instance._context = None
        instance._pipeline = None
        return instance

    def __call__(self, pipeline, context):
        # Hold references to the lumen context and the original (unfiltered)
        # pipeline so the selection callbacks (which fire after __call__
        # returns) can read/write shared state and restore on reset.
        self._context = context
        self._pipeline = pipeline
        self._mm = ManifoldMapPanel(pipeline=pipeline)
        self._mm.param.watch(partial(self._sync_selection, pipeline), 'selection_expr')
        return self._mm

    def _reset_selection(self, event):
        # Restore the original, unfiltered pipeline/source and clear the plot.
        context = self._context
        context['source'] = self._pipeline.source
        context['pipeline'] = self._pipeline
        context['table'] = self._pipeline.table
        self._mm.selection_expr = None
        self._mm._ls.selection_expr = None
        self._initialized_selection = False
        if self._selection_markdown is not None:
            self._selection_markdown.object = "Selection cleared."

    async def _sync_selection(self, pipeline, event):
        if event.new is None:
            return

        context = self._context

        # Map the selection expression to the selected observation IDs, using
        # the dimensions the plot is currently displaying so the expression
        # resolves against the Dataset.
        adata = pipeline.source.get('obs', return_type='anndata')
        ds = Dataset(adata, self._mm._manifold_map.current_kdims())
        mask = event.new.apply(ds)
        selected = pipeline.data[mask]
        obs_ids = list(selected.obs_id)

        # Materialize the selection as its own queryable table so downstream
        # actors (SQLAgent, VegaLite, table views) can target it directly
        # instead of relying on filter state that raw SQL bypasses.
        if obs_ids:
            id_list = ", ".join(
                "'" + str(obs_id).replace("'", "''") + "'" for obs_id in obs_ids
            )
            where = f' WHERE "obs_id" IN ({id_list})'
        else:
            where = " WHERE FALSE"
        tables = dict(pipeline.source.tables)
        tables[SELECTION_TABLE] = f'SELECT * FROM "obs"{where}'
        source = pipeline.source.create_sql_expr_source(tables)

        context['source'] = source
        context['pipeline'] = pipeline.clone(
            source=source, table=SELECTION_TABLE, schema=None
        )
        context['table'] = SELECTION_TABLE
        context['data'] = await describe_data(selected)

        if not self._initialized_selection:
            button = Button(
                label="Reset Selection",
                on_click=self._reset_selection
            )
            self._selection_markdown = Markdown()
            self._reset_col = Column(self._selection_markdown, button)
            self._initialized_selection = True
        self._selection_markdown.object = (
            f"Selected {len(obs_ids)} points into table `{SELECTION_TABLE}`, "
            "which will be used for subsequent calls."
        )
        self._chat_message = self.interface.stream(self._reset_col, user="Assistant", message=self._chat_message)


class LeidenComputation(AnnDataAnalysis):
    """Perform Leiden clustering."""

    category = param.Selector(default=None, objects=[], doc="Category for the analysis.")

    random_state = param.Integer(
        default=0,
        allow_None=True,
        doc="""
        Random state for reproducibility.""",
    )

    resolution = param.Number(
        default=1.0,
        bounds=(0, None),
        doc="""
        Resolution parameter for clustering. Higher values lead to more clusters.""",
    )

    n_iterations = param.Integer(
        default=2,
        doc="""
        Number of iterations for the Leiden algorithm. -1 means iterate until convergence.""",
    )

    key_added = param.String(
        default="leiden_{resolution:.1f}",
        doc="""
        Key under which to store the clustering in adata.obs.""",
    )

    def __call__(self, pipeline, context):
        source = pipeline.source
        adata = source.get(pipeline.table, return_type="anndata")
        available_cols = list(adata.obs.columns)
        self.param.category.objects = available_cols
        self.category = available_cols[0]

        # Perform Leiden clustering
        leiden_operation = LeidenOperation.instance(
            random_state=self.random_state,
            resolution=self.resolution,
            n_iterations=self.n_iterations,
            key_added=self.key_added.format(resolution=self.resolution),
        )
        adata = leiden_operation(adata)

        # Create new source with updated adata
        pipeline.source = source.create_sql_expr_source(
            tables=source.tables,
            adata=adata,
            operations=source.operations + [leiden_operation],
        )
        self.message = (
            f"Leiden clustering completed with resolution {self.resolution} and stored in `adata.obs['{self.key_added.format(resolution=self.resolution)}']`."
        )
        return pipeline


class RankGenesGroupsTracksplot(AnnDataAnalysis):
    """Create a tracksplot visualization of top differentially expressed genes from rank_genes_groups analysis."""

    compute_required = param.Boolean(doc="""
        Whether to compute rank_genes_groups on the adata before rendering.""")

    groupby = param.Selector(default=None, objects=[], doc="Groupby category for the analysis.")

    n_genes = param.Integer(
        default=3,
        bounds=(1, None),
        doc="""
        Number of top genes to display in the tracksplot.""",
    )

    def __call__(self, pipeline, context):
        if not self.param.groupby.objects:
            source = pipeline.source
            adata = source.get(pipeline.table, return_type="anndata")
            available_cols = list(adata.obs.columns)
            self.param.groupby.objects = available_cols
        if not self.groupby:
            self.groupby = available_cols[0]
        return RankGenesGroupsTracksplotPanel(
            pipeline=pipeline,
            groupby=self.groupby,
            n_genes=self.n_genes,
            compute_required=self.compute_required
        )


class ClustermapVisualization(AnnDataAnalysis):
    """Create a clustered heatmap showing mean expression by groups, following scanpy paradigm."""

    def __call__(self, pipeline, context):
        # Simple validation that we have the required data
        source = pipeline.source
        adata = source.get(pipeline.table, return_type="anndata")

        # Check we have observation and variable data
        if len(adata.obs.columns) == 0:
            self.message = "No observation metadata available for grouping."
            return pipeline

        if len(adata.var.index) == 0:
            self.message = "No genes available in the dataset."
            return pipeline

        self.message = (
            f"ClustermapVisualization view ready with {len(adata.obs.columns)} grouping options "
            f"and {len(adata.var.index)} genes available."
        )

        return ClustermapPanel(pipeline=pipeline)
