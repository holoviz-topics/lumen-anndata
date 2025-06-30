import holoviews as hv
import panel as pn

from hv_anndata import Dotmap, ManifoldMap
from lumen.views import View


class ManifoldMapPanel(View):
    view_type = "manifold_map"

    def __init__(self, **params):
        super().__init__(**params)
        self.adata = self.pipeline.source.get(self.pipeline.table, return_type="anndata")

    def get_panel(self):
        hv.Store.set_current_backend("bokeh")

        sel_marker_genes = {
            "CD14+ Mono": [],
            "CD16+ Mono": [],
            "ID2-hi myeloid prog": [],
            "cDC2": [],
            "Lymph prog": [],
            "B1 B": [],
            "Plasma cells": [],
            "CD4+ T activated": [],
            "pDC": [],
        }

        json_editor = pn.widgets.JSONEditor(
            value=sel_marker_genes, width=310, sizing_mode="stretch_height"
        )

        mmap = ManifoldMap(adata=self.adata, height=475)
        dmap = pn.Row(
            json_editor,
            pn.bind(
                self._update_groupby,
                groupby=mmap.param.color_by,
                marker_genes=json_editor.param.value,
            ),
            sizing_mode="stretch_both",
        )
        return pn.Column(
            mmap,
            dmap,
            sizing_mode="stretch_both",
        )

    def _update_groupby(self, groupby, marker_genes):
        try:
            return Dotmap(
                adata=self.adata,
                marker_genes=marker_genes,
                groupby=groupby,
                sizing_mode="stretch_both",
            )
        except Exception:
            return pn.pane.Placeholder(sizing_mode="stretch_both")
