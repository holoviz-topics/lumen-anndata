import holoviews as hv
import matplotlib
import panel as pn
import param

matplotlib.use("Agg")
import scanpy as sc

from hv_anndata.plotting import ClusterMap, Dotmap, ManifoldMap
from lumen.views import View


class AnnDataPanel(View):

    compute_required = param.Boolean(default=False)

    def __init__(self, **params):
        super().__init__(**params)
        source = self.pipeline.source
        table = self.pipeline.table
        self.adata = source.get(table, return_type="anndata")

    def _compute_required(self) -> bool:
        return True

    def _render_visualization(self):
        raise NotImplementedError("Subclasses must implement render_visualization method.")

    def get_panel(self):
        hv.Store.set_current_backend("bokeh")
        try:
            if self.compute_required:
                self._compute_required()
            return self._render_visualization()
        except Exception as e:
            return pn.pane.Alert(f"""
                Encountered error: {e}, which might require additional computation.
                To try auto-running required computations, please check off
                'Compute required' and click 'Run' again, or try a
                different dataset with all computations completed.""",
                alert_type="warning"
            )


class ManifoldMapPanel(AnnDataPanel):

    selection_expr = param.Parameter(doc="""
        A selection expression capturing the current selection applied
        on the plot.""")

    selection_group = param.String(default='anndata', doc="""
        Declares a selection group the plot is part of.""")

    view_type = "manifold_map"

    def __init__(self, **params):
        super().__init__(**params)
        self._manifold_map = None  # set on render; read by selection callbacks

    def _render_visualization(self):
        if self._ls is None:
            self._init_link_selections()
        self._manifold_map = ManifoldMap(adata=self.adata, ls=self._ls)
        return self._manifold_map


class RankGenesGroupsTracksplotPanel(AnnDataPanel):

    groupby = param.Selector(default=None, objects=[], doc="Groupby category for the analysis.")

    n_genes = param.Integer(
        default=3,
        bounds=(1, None),
        doc="Number of top genes to display in the tracksplot.",
    )

    view_type = "rank_genes_groups_tracksplot"

    def _compute_required(self) -> bool:
        sc.tl.rank_genes_groups(self.adata, groupby=self.groupby)

    def _render_visualization(self):
        axes = sc.pl.rank_genes_groups_tracksplot(self.adata, n_genes=self.n_genes, show=False)["track_axes"]
        return pn.pane.Matplotlib(axes[0].figure, tight=True, sizing_mode="stretch_both")


class ClustermapPanel(View):
    view_type = "clustermap"

    def get_panel(self):
        hv.Store.set_current_backend("bokeh")
        return ClusterMap(adata=self.pipeline.source.get(self.pipeline.table, return_type="anndata"))


class DotmapPanel(AnnDataPanel):

    groupby = param.String(default=None, allow_None=True, doc="""
        Categorical obs column to group cells by; Dotmap falls back to the
        first categorical column when unset.""")

    marker_genes = param.ClassSelector(class_=(dict, list), default={}, doc="""
        Marker genes to plot, either a flat list or a mapping of group name to
        gene list. Also editable in the plot's gene selector.""")

    view_type = "dotmap"

    def _render_visualization(self):
        dotmap = Dotmap(adata=self.adata, marker_genes=self.marker_genes)
        # Dotmap._on_input_data fires on init and resets groupby to the first
        # categorical obs column, so anything passed to the constructor is
        # discarded; assign afterwards to make it stick.
        if self.groupby:
            dotmap.groupby = self.groupby
        return dotmap
