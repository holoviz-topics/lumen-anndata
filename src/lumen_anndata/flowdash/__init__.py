"""
lumen-anndata's analyses as FlowDash nodes, for Lumen's FlowDash app.

The package is a FlowDash component library next to ``lumen.flowdash``:
``lumen-anndata --flowdash data.h5ad`` serves both, with an agent that can
place these nodes. Experimental, like Lumen's FlowDash integration it builds on.
"""
from .nodes import (
    CLUSTERMAP_ID, COMPONENTS, LEIDEN_ID, MANIFOLD_MAP_ID, PIPELINE_PORTS,
    SELECTION_TABLE, TRACKSPLOT_ID, ClustermapNode, LeidenNode,
    ManifoldMapNode, TracksplotNode, pipeline_adata,
)

__flowdash_components__ = COMPONENTS

__all__ = [
    "CLUSTERMAP_ID",
    "COMPONENTS",
    "LEIDEN_ID",
    "MANIFOLD_MAP_ID",
    "PIPELINE_PORTS",
    "SELECTION_TABLE",
    "TRACKSPLOT_ID",
    "ClustermapNode",
    "LeidenNode",
    "ManifoldMapNode",
    "TracksplotNode",
    "pipeline_adata",
]
