"""Operations for transforming AnnData objects in Lumen."""

from __future__ import annotations

from typing import TYPE_CHECKING

import holoviews as hv
import param
import scanpy as sc

from holoviews.operation import Operation

if TYPE_CHECKING:
    from anndata import AnnData


class AnnDataOperation(param.Parameterized):
    """Base class for operations that can be applied to AnnData objects.

    All operations should:
    - Use param parameters for configuration
    - Implement the apply() method
    - Return a modified AnnData object (or modify in-place based on copy parameter)
    - Be serializable through param
    - Declare requires tables via _requires_tables class variable
    """

    requires = param.List(
        default=[],
        doc="""
        List of table names that this operation reads from.
        This is used to determine if the operation applies to a specific table.
        """,
    )

    def __init__(self, **params):
        super().__init__(**params)

    @classmethod
    def applies(cls, table: str) -> bool:
        """Check if this operation affects a specific table.

        Parameters
        ----------
        table : str
            Name of the table to check

        Returns
        -------
        bool
            True if this operation reads from or writes to the table
        """
        return table in cls.requires

    def apply(self, adata: AnnData) -> AnnData:
        """Apply the operation to an AnnData object.

        Parameters
        ----------
        adata : AnnData
            The AnnData object to transform

        Returns
        -------
        AnnData
            The transformed AnnData object
        """
        raise NotImplementedError("Subclasses must implement apply()")


class Leiden(AnnDataOperation):
    """Perform Leiden clustering."""

    requires = ["obs"]

    random_state = param.Integer(
        default=0,
        allow_None=True,
        doc="""
        Random state for reproducibility.""",
    )

    resolution = param.Number(
        default=1.0,
        bounds=(0, None),
        doc="""
        Resolution parameter for clustering. Higher values lead to more clusters.""",
    )

    n_iterations = param.Integer(
        default=-1,
        doc="""
        Number of iterations for the Leiden algorithm. -1 means iterate until convergence.""",
    )

    key_added = param.String(
        default="leiden",
        doc="""
        Key under which to store the clustering in adata.obs.""",
    )

    def apply(self, adata: AnnData) -> AnnData:
        """Perform Leiden clustering."""
        # Compute neighbors if not present
        if "neighbors" not in adata.uns:
            sc.pp.neighbors(adata, random_state=self.random_state, copy=False)

        sc.tl.leiden(
            adata, resolution=self.resolution, n_iterations=self.n_iterations, random_state=self.random_state, key_added=self.key_added, copy=False, flavor="igraph"
        )

        return adata


class ComputeEmbedding(AnnDataOperation):
    """Compute a standard embedding (PCA + neighbors + UMAP)."""

    n_pcs = param.Integer(
        default=50,
        bounds=(1, None),
        doc="""
        Number of principal components to use.""",
    )

    n_neighbors = param.Integer(
        default=15,
        bounds=(2, None),
        doc="""
        Number of neighbors for UMAP.""",
    )

    min_dist = param.Number(
        default=0.5,
        bounds=(0, None),
        doc="""
        Minimum distance for UMAP.""",
    )

    random_state = param.Integer(
        default=0,
        allow_None=True,
        doc="""
        Random state for reproducibility.""",
    )

    def apply(self, adata: AnnData) -> AnnData:
        """Compute standard embedding pipeline."""
        # Run PCA if not present
        if "X_pca" not in adata.obsm:
            sc.pp.pca(adata, n_comps=self.n_pcs, random_state=self.random_state)

        # Compute neighbors
        sc.pp.neighbors(adata, n_neighbors=self.n_neighbors, n_pcs=self.n_pcs, random_state=self.random_state)

        # Compute UMAP
        sc.tl.umap(adata, min_dist=self.min_dist, random_state=self.random_state)

        return adata


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
