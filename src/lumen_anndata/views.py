import holoviews as hv
import param

from hv_anndata import Dotmap, ManifoldMap
from lumen.views import View


class ManifoldMapPanel(View):

    selection_expr = param.Parameter(doc="""
        A selection expression capturing the current selection applied
        on the plot.""")

    selection_group = param.String(default='anndata', doc="""
        Declares a selection group the plot is part of.""")

    view_type = "manifold_map"

    def get_panel(self):
        hv.Store.set_current_backend("bokeh")
        if self._ls is None:
            self._init_link_selections()
        adata = self.pipeline.source.get(self.pipeline.table, return_type="anndata")
        return ManifoldMap(adata=adata, ls=self._ls)


class DotMapPanel(View):

    view_type = "dot_map"

    groupby = param.String(default="cell_type")

    marker_genes = param.Dict(default={})

    def get_panel(self):
        hv.Store.set_current_backend("bokeh")
        adata = self.pipeline.source.get(self.pipeline.table, return_type="anndata")
        return Dotmap(adata=adata, groupby=self.groupby, marker_genes=self.marker_genes)
