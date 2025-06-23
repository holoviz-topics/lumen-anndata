import param
import scanpy as sc

from lumen.ai.analysis import Analysis

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

    def __call__(self, pipeline):
        return ManifoldMapPanel(pipeline=pipeline)


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

        if "neighbors" not in adata.uns:
            sc.pp.neighbors(adata, random_state=self.random_state, copy=False)

        sc.tl.leiden(
            adata,
            resolution=self.resolution,
            n_iterations=self.n_iterations,
            random_state=self.random_state,
            key_added=self.key_added.format(resolution=self.resolution),
            copy=False,
            flavor="igraph",
        )

        # Create new source with updated adata
        pipeline.source = source.create_sql_expr_source(
            tables=source.tables,
            adata=adata,
        )
        return pipeline
