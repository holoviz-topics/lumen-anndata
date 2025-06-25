import holoviews as hv

from hv_anndata import ManifoldMap
from lumen.views import View


class ManifoldMapPanel(View):
    view_type = "manifold_map"

    def get_panel(self):
        hv.Store.set_current_backend("bokeh")
        return ManifoldMap(adata=self.pipeline.source.get(self.pipeline.table, return_type="anndata"))
