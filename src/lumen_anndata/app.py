import asyncio

from typing import Literal

import anndata as ad
import datashader as ds
import holoviews as hv
import lumen.ai as lmai
import panel as pn
import param
import pooch
import scanpy as sc

from holoviews.operation import Operation
from holoviews.operation.datashader import datashade, spread

from lumen_anndata.source import AnnDataSource


class labeller(Operation):

    column = param.String()

    max_labels = param.Integer(10)

    min_count = param.Integer(default=100)

    streams = param.List([hv.streams.RangeXY])

    x_range = param.Tuple(default=None, length=2, doc="""
       The x_range as a tuple of min and max x-value. Auto-ranges
       if set to None.""")

    y_range = param.Tuple(default=None, length=2, doc="""
       The x_range as a tuple of min and max x-value. Auto-ranges
       if set to None.""")

    def _process(self, el, key=None):
        if self.p.x_range and self.p.y_range:
            el = el[slice(*self.p.x_range), slice(*self.p.y_range)]
        df = el.dframe()
        xd, yd, cd = el.dimensions()[:3]
        col = self.p.column or cd.name
        result = df.groupby(col).agg(
            count=(col, 'size'),  # count of rows per group
            x=(xd.name, 'mean'),
            y=(yd.name, 'mean')
        ).query(f'count > {self.p.min_count}').sort_values('count', ascending=False).iloc[:self.p.max_labels].reset_index()
        return hv.Labels(result, ['x', 'y'], col)


def upload_h5ad(file, table) -> int:
    """
    Uploads an h5ad file and returns an AnnDataSource.
    """
    adata = ad.read_h5ad(file)
    try:
        src = AnnDataSource(adata=adata)
        lmai.memory['sources'] = lmai.memory["sources"] + [src]
        lmai.memory['source'] = src
        return 1
    except Exception:
        return 0

# def umap_plot(source, category):
#     """
#     Plots a UMAP plot of the current source data and optionally colors by category.
#     """
#     adata = source.get('obs', return_type="adata")
#     data = tuple(adata.obsm["X_umap"].T)
#     vdims = []
#     agg = 'count'
#     if category:
#         data = data+(adata.obs[category].values,)
#         vdims = [category]
#         agg = ds.count_cat(category)
#     points = hv.Points(data, vdims=vdims)
#     shaded = spread(datashade(points, aggregator=agg), px=4).opts(responsive=True, height=600, xaxis=None, yaxis=None, show_legend=True, show_grid=True)
#     if category:
#         return pn.panel(shaded * labeller(points).opts(text_color='black'))
#     return pn.panel(shaded)

async def load_data():
    await asyncio.sleep(0.1)
    with ui.interface.param.update(loading=True):
        anndata_file_path = pooch.retrieve(
            url="https://datasets.cellxgene.cziscience.com/ad4aac9c-28e6-4a1f-ab48-c4ae7154c0cb.h5ad",
            fname="ad4aac9c-28e6-4a1f-ab48-c4ae7154c0cb.h5ad",
            known_hash="00ee1a7d9dbb77dc5b8e27d868d3c371f1f53e6ef79a18e5f1fede166b31e2eb",
            path="data-download"
        )
        adata = pn.state.as_cached('anndata', ad.read_h5ad, filename=anndata_file_path)
        src = AnnDataSource(adata=adata)
        lmai.memory['sources'] = [src]
        lmai.memory['source'] = src

ui = lmai.ExplorerUI(
    title='AnnData Explorer',
    # tools=[lmai.tools.FunctionTool(umap_plot, requires=['source'])],
    table_upload_callbacks={
        ".h5ad": upload_h5ad,
    },
    log_level="DEBUG"
)
pn.state.onload(load_data)
ui.servable()
