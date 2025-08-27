import holoviews as hv
import matplotlib
import panel as pn
import param

matplotlib.use("Agg")
import scanpy as sc

from hv_anndata import ManifoldMap
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


class RankGenesGroupsTracksplotPanel(View):
    view_type = "rank_genes_groups_tracksplot"

    n_genes = param.Integer(
        default=3,
        bounds=(1, None),
        doc="Number of top genes to display in the tracksplot.",
    )

    key_added = param.String(
        default="clusters",
        doc="Key under which to store the clustering in adata.obs. Defaults to 'clusters'."
    )

    def __init__(self, **params):
        super().__init__(**params)
        self.adata = self.pipeline.source.get(self.pipeline.table, return_type="anndata")

    def get_panel(self):
        axes = sc.pl.rank_genes_groups_tracksplot(self.adata, n_genes=self.n_genes, show=False)["track_axes"]
        return pn.pane.Matplotlib(axes[0].figure, tight=True, sizing_mode="stretch_both")
