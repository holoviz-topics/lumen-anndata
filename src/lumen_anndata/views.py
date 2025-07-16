import holoviews as hv
import panel as pn
import panel_material_ui as pmui

from hv_anndata import Dotmap, ManifoldMap
from lumen.views import View

from .components import AutoCompleteMultiChoice


class ManifoldMapPanel(View):
    view_type = "manifold_map"

    def __init__(self, **params):
        super().__init__(**params)
        self.adata = self.pipeline.source.get(self.pipeline.table, return_type="anndata")

    def get_panel(self):
        hv.Store.set_current_backend("bokeh")

        mmap = ManifoldMap(adata=self.adata, height=475)
        return mmap

class DotMapPanel(View):
    view_type = "dot_map"

    def __init__(self, **params):
        super().__init__(**params)
        self.adata = self.pipeline.source.get(self.pipeline.table, return_type="anndata")

    def get_panel(self):
        hv.Store.set_current_backend("bokeh")

        cols = list(self.adata.obs.columns)
        ac_input = pmui.AutocompleteInput(
            options=cols, value=cols[0],
        )
        acmc_input = AutoCompleteMultiChoice(options=list(self.adata.var_names))
        dmap = pn.Row(
            pn.Column(
                ac_input,
                acmc_input,
            ),
            pn.bind(
                self._update_groupby,
                groupby=ac_input.param.value,
                marker_genes=acmc_input.param.value,
            ),
            sizing_mode="stretch_both",
        )
        return dmap

    def _update_groupby(self, groupby, marker_genes):
        try:
            return Dotmap(
                adata=self.adata,
                marker_genes=marker_genes,
                groupby=groupby,
            )
        except Exception as e:
            print(e)  # noqa: T201
            return pn.pane.Placeholder(sizing_mode="stretch_both")
