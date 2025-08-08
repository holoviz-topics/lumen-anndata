import holoviews as hv
import param

from hv_anndata import ManifoldMap
from lumen.ai.memory import _Memory, memory
from lumen.views import View


class ManifoldMapPanel(View):

    memory = param.ClassSelector(class_=_Memory, default=None, doc="""
        Local memory which will be used to provide the agent context.
        If None the global memory will be used.""")

    view_type = "manifold_map"

    def get_panel(self):
        hv.Store.set_current_backend("bokeh")
        bounds_xy = hv.streams.BoundsXY()
        bounds_xy.param.watch(self._stream_to_memory, "bounds")
        self._mmap = ManifoldMap(adata=self.pipeline.source.get(self.pipeline.table, return_type="anndata"), streams=[bounds_xy])
        return self._mmap

    @property
    def _memory(self):
        return memory if self.memory is None else self.memory

    def _stream_to_memory(self, event):
        x0, y0, x1, y1 = event.new
        source = self.pipeline.source
        reduction = self._mmap.reduction
        obsm_pca = source.get(f"obsm_{reduction}")
        # The ManifoldMap index is 1-based, so we need to subtract 1 to get the correct index
        x_axis_num = int(''.join(c for c in str(self._mmap.x_axis) if c.isdigit())) - 1
        y_axis_num = int(''.join(c for c in str(self._mmap.y_axis) if c.isdigit())) - 1
        obs_ids = obsm_pca.loc[
            (obsm_pca[f"{reduction}_{x_axis_num}"] > x0) & (obsm_pca[f"{reduction}_{x_axis_num}"] < x1) &
            (obsm_pca[f"{reduction}_{y_axis_num}"] > y0) & (obsm_pca[f"{reduction}_{y_axis_num}"] < y1),
            "obs_id"
        ]
        new_source = source.create_sql_expr_source(source.tables)
        new_source._obs_ids_selected = obs_ids
        self._memory["sources"] = self._memory.get("sources", []) + [new_source]
        self._memory["source"] = new_source
        self._memory.trigger("sources")

    def to_spec(self, context: dict | None = None) -> dict:
        spec = super().to_spec()
        spec.pop("memory", None)
        return spec
