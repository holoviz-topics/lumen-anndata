import anndata as ad
import holoviews as hv
import lumen.ai as lmai
import param

from holoviews.operation import Operation

from lumen_anndata.source import AnnDataSource


class labeller(Operation):
    column = param.String()

    max_labels = param.Integer(10)

    min_count = param.Integer(default=100)

    streams = param.List([hv.streams.RangeXY])

    x_range = param.Tuple(
        default=None,
        length=2,
        doc="""
       The x_range as a tuple of min and max x-value. Auto-ranges
       if set to None.""",
    )

    y_range = param.Tuple(
        default=None,
        length=2,
        doc="""
       The x_range as a tuple of min and max x-value. Auto-ranges
       if set to None.""",
    )

    def _process(self, el, key=None):
        if self.p.x_range and self.p.y_range:
            el = el[slice(*self.p.x_range), slice(*self.p.y_range)]
        df = el.dframe()
        xd, yd, cd = el.dimensions()[:3]
        col = self.p.column or cd.name
        result = (
            df.groupby(col)
            .agg(
                count=(col, "size"),  # count of rows per group
                x=(xd.name, "mean"),
                y=(yd.name, "mean"),
            )
            .query(f"count > {self.p.min_count}")
            .sort_values("count", ascending=False)
            .iloc[: self.p.max_labels]
            .reset_index()
        )
        return hv.Labels(result, ["x", "y"], col)


def upload_h5ad(file, table) -> int:
    """
    Uploads an h5ad file and returns an AnnDataSource.
    """
    adata = ad.read_h5ad(file)
    try:
        src = AnnDataSource(adata=adata)
        lmai.memory["sources"] = lmai.memory["sources"] + [src]
        lmai.memory["source"] = src
        return 1
    except Exception:
        return 0


ui = lmai.ExplorerUI(
    title="AnnData Explorer",
    table_upload_callbacks={
        ".h5ad": upload_h5ad,
    },
    log_level="DEBUG",
)
ui.servable()
