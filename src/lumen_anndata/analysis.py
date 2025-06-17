import param

from lumen.ai.analysis import Analysis

from .source import AnnDataSource
from .views import ManifoldMapPanel, UMAPPanel


class ManifoldMapAnalysis(Analysis):
    """
    Use this to visualize any requests for UMAP, PCA, tSNE results,
    unless explicitly otherwise specified by the user.
    """

    def __call__(self, pipeline):
        return ManifoldMapPanel(pipeline=pipeline)

    @classmethod
    async def applies(cls, pipeline) -> bool:
        source = pipeline.source
        if not isinstance(source, AnnDataSource):
            return False
        adata = source.get(pipeline.table, return_type="anndata")
        return adata is not None and len(adata.obsm) > 0


class ComputeEmbeddingAnalysis(Analysis):
    category = param.Selector(default=None, doc="Category to color points by in UMAP embedding.")

    def __call__(self, pipeline):
        source = pipeline.source
        adata = source.get(pipeline.table, return_type="anndata")
        available_cols = list(adata.obs.columns)
        self.param.category.objects = available_cols
        self.category = available_cols[0]
        return UMAPPanel(pipeline=pipeline, category=self.category, operation="ComputeEmbedding")

    @classmethod
    async def applies(cls, pipeline) -> bool:
        source = pipeline.source
        if not isinstance(source, AnnDataSource) or source._obs_ids_selected is None:
            return False
        adata = source.get(pipeline.table, return_type="anndata")
        return adata is not None and len(adata.obsm) > 0
