import colorcet as cc
import datashader as ds
import holoviews as hv
import panel as pn
import param

from holoviews.operation.datashader import datashade, spread
from hv_anndata import ManifoldMap
from lumen.views import View

from .operations import ComputeEmbedding, Leiden, labeller


class ManifoldMapPanel(View):

    view_type = "manifold_map"

    def get_panel(self):

        hv.extension("bokeh")

        return ManifoldMap(adata=self.pipeline.get(self.pipeline.table, return_type="anndata"))


class UMAPPanel(View):

    category = param.Selector(
        default=None,
        objects=[
            "leiden",
            "cell_type",
            "tissue",
            "development_stage",
            "observation_joinid",
        ],
        doc="Category to color points by in UMAP embedding.",
    )

    operation = param.Selector(
        default=None,
        objects=[ComputeEmbedding, Leiden],
        doc="Operation to apply for UMAP embedding.",
    )

    operation_kwargs = param.Dict(
        default={},
        doc="Keyword arguments to pass to the selected operation.",
    )

    view_type = "umap"

    def get_panel(self):

        hv.extension("bokeh")

        adata = self.pipeline.source.get(self.pipeline.table, return_type="anndata")
        if self.operation:
            adata = self.operation(**self.operation_kwargs).apply(adata)
        data = tuple(adata.obsm["X_umap"].T)
        vdims = []
        agg = "count"
        color_key = "glasbey"
        if self.category:
            data += (adata.obs[self.category].values,)
            vdims = [self.category]
            agg = ds.count_cat(self.category)
            color_key = cc.glasbey_dark[: len(adata.obs[self.category].unique())]
        points = hv.Points(data, vdims=vdims)
        shaded = spread(datashade(points, aggregator=agg, color_key=color_key), px=4).opts(
            responsive=True, height=600, xaxis=None, yaxis=None, show_legend=True, show_grid=True
        )
        if self.category:
            return pn.panel(shaded * labeller(points).opts(text_color="black"))
        return pn.panel(shaded)
