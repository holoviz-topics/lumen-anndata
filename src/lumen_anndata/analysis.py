from functools import partial

import param

from holoviews import Dataset, renderer
from hv_anndata import register
from lumen.ai.analysis import Analysis
from lumen.ai.utils import describe_data
from lumen.transforms import SQLFilter
from lumen.util import get_dataframe_schema
from panel.layout import Column
from panel.pane.markup import Markdown
from panel_material_ui import Button
from param.parameterized import bothmethod

from lumen_anndata.operations import LeidenOperation

from .source import AnnDataSource
from .views import (
    ClustermapPanel, DotMapPanel, ManifoldMapPanel,
    RankGenesGroupsTracksplotPanel,
)

renderer("bokeh").webgl = False
register()

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

    icon = "scatter_plot"

    @bothmethod
    def instance(self_or_cls, **params):
        instance = super().instance(**params)
        instance._initialized_selection = False
        instance._chat_message = None
        instance._reset_col = None
        instance._selection_markdown = None
        instance._pipeline = None
        return instance

    def __call__(self, pipeline, context):
        self._pipeline = pipeline
        self._mm = ManifoldMapPanel(pipeline=pipeline)
        self._mm.param.watch(partial(self._sync_selection, pipeline), 'selection_expr')
        return self._mm

    def _reset_selection(self, event):
        # Republish the original pipeline so downstream reverts to it.
        self._dynamic_provides = {
            'source': self._pipeline.source,
            'pipeline': self._pipeline,
            'table': self._pipeline.table,
        }
        self._mm.selection_expr = None
        self._mm._ls.selection_expr = None
        self._initialized_selection = False
        if self._selection_markdown is not None:
            self._selection_markdown.object = "Selection cleared."

    async def _sync_selection(self, pipeline, event):
        if event.new is None:
            return

        # Apply the lasso to the data the map is currently showing (its own
        # adata), so selecting on an already-filtered map narrows that subset
        # instead of re-selecting against the full obs table. obs_id is the obs
        # index promoted to a column, so obs_names gives the selected ids.
        adata = self._mm.adata
        ds = Dataset(adata, self._mm._manifold_map.current_kdims())
        mask = event.new.apply(ds)
        obs_ids = [str(obs_id) for obs_id in adata.obs_names[mask]]

        # Materialize the selection keyed by obs_id off base obs; obs_id is a
        # global identity, so filtering base obs by the ids selected on the
        # current view yields the narrowed subset regardless of current table.
        # SQLFilter handles quoting/dialect (as AnnDataSource does for obs_id);
        # an empty list is a no-op there, so guard it with an empty result.
        base_sql = 'SELECT * FROM "obs"'
        tables = dict(pipeline.source.tables)
        if obs_ids:
            tables[SELECTION_TABLE] = SQLFilter(
                conditions=[("obs_id", obs_ids)]
            ).apply(base_sql)
        else:
            tables[SELECTION_TABLE] = f'SELECT * FROM ({base_sql}) WHERE FALSE'
        source = pipeline.source.create_sql_expr_source(tables)

        # Fetch the selection once and derive both the schema and the summary
        # from it. Leaving schema=None makes Pipeline.__init__ call
        # source.get_schema(SELECTION_TABLE), which issues a DISTINCT per
        # column plus a count and a min/max; the selection's stored SQL
        # expression carries one literal per selected obs_id, so each of those
        # ~41 queries re-executes it (measured: 33s for 37k rows). Describing
        # the frame we already hold gives the same schema -- with the subset's
        # real ranges and enums, not the parent's -- for the cost of one query.
        selection_df = source.get(SELECTION_TABLE)
        schema = get_dataframe_schema(selection_df)["items"]["properties"]
        # get_dataframe_schema omits the row count that BaseSQLSource.get_schema
        # adds, and consumers read it off the schema.
        schema["__len__"] = len(selection_df)

        # Publish through the analysis out-context (Analysis._dynamic_provides);
        # the input context is a snapshot, so mutating it wouldn't propagate.
        # `source` is included so it registers for downstream discovery.
        self._dynamic_provides = {
            'source': source,
            'pipeline': pipeline.clone(
                source=source, table=SELECTION_TABLE, schema=schema
            ),
            'table': SELECTION_TABLE,
            'data': await describe_data(selection_df),
        }

        if not self._initialized_selection:
            button = Button(
                label="Reset Selection",
                on_click=self._reset_selection,
                margin=(0, 0, 15, 15)
            )
            self._selection_markdown = Markdown()
            self._reset_col = Column(self._selection_markdown, button)
            self._initialized_selection = True
        self._selection_markdown.object = (
            f"Selected {len(obs_ids)} points into table `{SELECTION_TABLE}`, "
            "which will be used for subsequent calls."
        )
        self._chat_message = self.interface.stream(self._reset_col, user="Assistant", message=self._chat_message)


class DotMapVisualization(AnnDataAnalysis):

    compute_required = param.Boolean(doc="""
        Whether to compute pca, neighbors, and umap on the adata before rendering.""")

    groupby = param.Selector(default=None, objects=[], doc="""
        Groupby variable for the dot map.""")

    marker_genes = param.Dict(default={}, doc="""
        Marker genes for the dot map.""")

    populate_marker_genes = param.Boolean(default=False, doc="""
        Whether to populate marker genes arbitrarily with samples.""")

    def __call__(self, pipeline):
        adata = pipeline.source.get(pipeline.table, return_type="anndata")
        self._adata = adata

        # Populate groupby options
        groupby_values = list(adata.obs.columns)
        self.param["groupby"].objects = groupby_values
        self.groupby = groupby_values[0]

        self._dot_map = DotMapPanel(
            pipeline=pipeline,
            groupby=self.groupby,
            marker_genes=self.marker_genes,
            compute_required=self.compute_required
        )
        return self._dot_map

    @param.depends("populate_marker_genes", watch=True, on_init=True)
    def _populate_marker_genes(self):
        if not self.populate_marker_genes:
            return
        all_genes = self._adata.var_names.tolist()[:30]
        self.marker_genes = {
            f"Sample {i}": all_genes[i:i+10]
            for i in range(0, len(all_genes), 10)
        }

class LeidenComputation(AnnDataAnalysis):
    """Perform Leiden clustering."""

    icon = "hub"

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

    icon = "stacked_line_chart"

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

    icon = "grid_on"

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
