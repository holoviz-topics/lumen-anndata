import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn

from holoviews.operation import dendrogram
from hv_anndata import ManifoldMap
from lumen.views import View
from panel_material_ui import MultiChoice


class ManifoldMapPanel(View):
    view_type = "manifold_map"

    def get_panel(self):
        hv.Store.set_current_backend("bokeh")
        return ManifoldMap(adata=self.pipeline.source.get(self.pipeline.table, return_type="anndata"))


class ClustermapPanel(View):
    view_type = "clustermap"

    def __init__(self, **params):
        super().__init__(**params)
        self.adata = self.pipeline.source.get(self.pipeline.table, return_type="anndata")

    def get_panel(self):
        hv.Store.set_current_backend("bokeh")
        hv.extension("bokeh")

        # Get available options
        gene_names = list(self.adata.var.index)
        obs_names = list(self.adata.obs.index)

        # Create widgets for selection
        genes_input = MultiChoice(
            name="Genes",
            options=gene_names,
            value=gene_names[:50] if len(gene_names) >= 50 else gene_names,  # Default to first 50 genes
            height=200,
        )

        max_cells_input = pn.widgets.IntSlider(name="Max cells", start=10, end=min(1000, len(obs_names)), value=min(100, len(obs_names)), step=10)

        # Additional options
        use_raw_input = pn.widgets.Checkbox(name="Use raw data", value=False)

        cluster_genes_input = pn.widgets.Checkbox(name="Cluster genes", value=True)

        cluster_cells_input = pn.widgets.Checkbox(name="Cluster cells", value=True)

        # Create layout with widgets and bound plot
        controls = pn.Column(genes_input, max_cells_input, use_raw_input, cluster_genes_input, cluster_cells_input, width=300)

        plot_pane = pn.bind(
            self._update_clustermap,
            genes=genes_input.param.value,
            max_cells=max_cells_input.param.value,
            use_raw=use_raw_input.param.value,
            cluster_genes=cluster_genes_input.param.value,
            cluster_cells=cluster_cells_input.param.value,
        )

        return pn.Row(
            controls,
            plot_pane,
            sizing_mode="stretch_both",
        )

    def _update_clustermap(self, genes, max_cells, use_raw, cluster_genes, cluster_cells):
        if not genes:
            return pn.pane.Placeholder("Please select at least one gene.", sizing_mode="stretch_both")

        # Filter to valid genes
        valid_genes = [gene for gene in genes if gene in self.adata.var.index]

        if not valid_genes:
            return pn.pane.Placeholder("No valid genes selected.", sizing_mode="stretch_both")

        # Subsample cells if needed
        if max_cells < len(self.adata.obs):
            # Random sample of cells
            np.random.seed(42)  # For reproducibility
            cell_indices = np.random.choice(len(self.adata.obs), max_cells, replace=False)
            selected_cells = self.adata.obs.index[cell_indices]
        else:
            selected_cells = self.adata.obs.index

        # Get expression data
        adata_subset = self.adata[selected_cells, valid_genes]

        if use_raw and adata_subset.raw is not None:
            expr_data = adata_subset.raw.X
        else:
            expr_data = adata_subset.X

        # Create DataFrame with expression data
        expr_df = pd.DataFrame(expr_data, index=selected_cells, columns=valid_genes)

        # Prepare data for HoloViews HeatMap (convert to long format)
        heatmap_df = expr_df.stack().reset_index()
        heatmap_df.columns = ["cell", "gene", "expression"]

        heatmap = hv.HeatMap(heatmap_df, ["gene", "cell"], "expression").opts(
            colorbar=True,
            title="Hierarchical Clustering Heatmap",
            xrotation=90,
            fontsize={"labels": 8, "title": 12},
            # responsive=True,  # the dendrograms are not responsive, so we set fixed size
            width=500, height=500,
            tools=["hover"],
        )

        adjoint_dims = []
        if cluster_genes:
            adjoint_dims.append("gene")
        if cluster_cells:
            adjoint_dims.append("cell")

        if adjoint_dims:
            dendrogram_plot = dendrogram(
                heatmap,
                adjoint_dims=adjoint_dims,
                main_dim="expression",
            )
            return dendrogram_plot
        return heatmap
