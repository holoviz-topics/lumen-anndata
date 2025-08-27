import holoviews as hv
import matplotlib
import panel as pn
import param

matplotlib.use("Agg")
import scanpy as sc

from hv_anndata import ManifoldMap
from lumen.views import View


class AnnDataPanel(View):

    solve_dependencies = param.Boolean(default=False)

    def __init__(self, **params):
        super().__init__(**params)
        source = self.pipeline.source
        table = self.pipeline.table
        self.adata = source.get(table, return_type="anndata")

    def _resolve_dependencies(self) -> bool:
        return True

    def _render_visualization(self):
        raise NotImplementedError("Subclasses must implement render_visualization method.")

    def get_panel(self):
        hv.Store.set_current_backend("bokeh")
        try:
            if self.solve_dependencies:
                self._resolve_dependencies()
            return self._render_visualization()
        except Exception as e:
            return pn.pane.Alert(f"""
                Encountered error: {e}, which might be a missing dependency.
                To try auto-resolving dependency issues, please check off
                'Solve dependencies' and click 'Run' again, or try a
                different dataset with all depdendencies satisfied.""",
                alert_type="warning"
            )


class ManifoldMapPanel(AnnDataPanel):

    selection_expr = param.Parameter(doc="""
        A selection expression capturing the current selection applied
        on the plot.""")

    selection_group = param.String(default='anndata', doc="""
        Declares a selection group the plot is part of.""")

    view_type = "manifold_map"

    def _render_visualization(self):
        if self._ls is None:
            self._init_link_selections()
        return ManifoldMap(adata=self.adata, ls=self._ls)


class RankGenesGroupsTracksplotPanel(AnnDataPanel):

    groupby = param.Selector(default=None, objects=[], doc="Groupby category for the analysis.")

    n_genes = param.Integer(
        default=3,
        bounds=(1, None),
        doc="Number of top genes to display in the tracksplot.",
    )

    view_type = "rank_genes_groups_tracksplot"

    def _resolve_dependencies(self) -> bool:
        sc.tl.rank_genes_groups(self.adata, groupby=self.groupby)

    def _render_visualization(self):
        axes = sc.pl.rank_genes_groups_tracksplot(self.adata, n_genes=self.n_genes, show=False)["track_axes"]
        return pn.pane.Matplotlib(axes[0].figure, tight=True, sizing_mode="stretch_both")
