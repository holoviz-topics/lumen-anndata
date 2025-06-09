from io import BytesIO

import anndata as ad

from lumen.ai.memory import memory

from lumen_anndata.source import AnnDataSource


def upload_h5ad(file: BytesIO, table: str) -> int:
    """
    Uploads an h5ad file and returns an AnnDataSource.
    """
    adata = ad.read_h5ad(file)
    try:
        src = AnnDataSource(adata=adata)
        memory["sources"] = memory["sources"] + [src]
        memory["source"] = src
        return 1
    except Exception:
        return 0
