
import holoviews as hv
import lumen.ai as lmai

from lumen_anndata.analysis import ManifoldMapAnalysis
from lumen_anndata.utils import upload_h5ad

hv.extension("bokeh")


ui = lmai.ExplorerUI(
    title="AnnData Explorer",
    table_upload_callbacks={
        ".h5ad": upload_h5ad,
    },
    analyses=[ManifoldMapAnalysis],
    log_level="DEBUG",
)
ui.servable()
