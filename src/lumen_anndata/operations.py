"""Operations for transforming AnnData objects in Lumen."""

from __future__ import annotations

import holoviews as hv
import param
import scanpy as sc

from holoviews.operation import Operation


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


class AnnDataOperation(param.ParameterizedFunction):
    """Base class for operations that can be applied to AnnData objects."""

    requires = param.List(default=[], doc="Tables required by this operation.")

    def __hash__(self):
        """Hash the operation based on its parameters."""
        return hash((self.__class__,) + tuple(sorted(self.param.values().items())))


class LeidenOperation(AnnDataOperation):
    """Operation that performs Leiden clustering on AnnData."""

    random_state = param.Integer(default=0, allow_None=True, doc="Random state for reproducibility.")

    resolution = param.Number(default=1.0, bounds=(0, None), doc="Resolution parameter for clustering. Higher values lead to more clusters.")

    n_iterations = param.Integer(default=2, doc="Number of iterations for the Leiden algorithm. -1 means iterate until convergence.")

    key_added = param.String(default="leiden_{resolution:.1f}", doc="Key under which to store the clustering in adata.obs.")

    def __call__(self, adata):
        """Apply Leiden clustering to the AnnData object."""
        if "neighbors" not in adata.uns:
            sc.pp.neighbors(adata, random_state=self.random_state, copy=False)

        sc.tl.leiden(
            adata,
            resolution=self.resolution,
            n_iterations=self.n_iterations,
            random_state=self.random_state,
            key_added=self.key_added.format(resolution=self.resolution),
            copy=False,
            flavor="igraph",
        )
        return adata
