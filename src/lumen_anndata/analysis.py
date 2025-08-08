from functools import partial

import param

from holoviews import Dataset
from hv_anndata.interface import ACCESSOR as A
from lumen.ai.analysis import Analysis
from lumen.filters import ConstantFilter
from lumen_anndata.operations import LeidenOperation

from .source import AnnDataSource
from .views import ManifoldMapPanel


class AnnDataAnalysis(Analysis):
    """
    Base class for analyses that operate on AnnData objects.
    This class is used to ensure that the analysis can be applied
    to an AnnDataSource.
    """

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

    selection = param.Parameter()

    def __call__(self, pipeline):
        source = pipeline.source.create_sql_expr_source(pipeline.source.tables)
        if 'sources' not in self._memory:
            self._memory['sources'] = []
        self._memory['sources'] += [source]
        self._memory['source'] = [source]
        self._memory['pipeline'] = selected = pipeline.clone(source=source)
        filt = ConstantFilter(field='obs_id')
        selected.add_filter(filt)
        mm = ManifoldMapPanel(pipeline=pipeline)
        mm.param.watch(partial(self._sync_selection, pipeline, filt), 'selection_expr')
        return mm

    def _sync_selection(self, pipeline, obs_filter, event):
        adata = pipeline.source.get('obs', return_type='anndata')
        dr_options = list(adata.obsm.keys())
        var = dr_options[0]
        ds = Dataset(adata, [A.obsm[var][:, 0], A.obsm[var][:, 1]])
        mask = event.new.apply(ds)
        obs_filter.value = list(pipeline.data[mask].obs_id)


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

    def __call__(self, pipeline):
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
            f"Leiden clustering completed with resolution {self.resolution} "
            f"and stored in `adata.obs['{self.key_added.format(resolution=self.resolution)}']`."
        )
        return pipeline
