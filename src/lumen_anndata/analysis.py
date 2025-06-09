from lumen.ai.analysis import Analysis

from .views import ManifoldMapPanel


class ManifoldMapAnalysis(Analysis):
    def __call__(self, pipeline):
        return ManifoldMapPanel(pipeline=pipeline)
