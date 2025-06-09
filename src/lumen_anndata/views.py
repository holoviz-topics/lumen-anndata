from hv_anndata import ManifoldMap
from lumen.views import View


class ManifoldMapPanel(View):
    def to_spec(self, context=None):
        return super().to_spec(context=context)

    @classmethod
    def from_spec(cls, spec):
        return super().from_spec(spec)

    def get_panel(self):
        import holoviews as hv
        hv.extension("bokeh")

        return ManifoldMap(adata=self.pipeline.source._adata_store)
