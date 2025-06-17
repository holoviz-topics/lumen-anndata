from io import BytesIO

import anndata as ad

from lumen.ai.memory import memory

from lumen_anndata.source import AnnDataSource


def upload_h5ad(file: BytesIO, table: str, filename: str) -> int:
    """
    Uploads an h5ad file and returns an AnnDataSource.
    """
    adata = ad.read_h5ad(file)
    try:
        src = AnnDataSource(adata=adata, uploaded_filename=filename)
        memory["source"] = src
        memory["sources"] = memory["sources"] + [src]
        return 1
    except Exception as e:
        print(f"Error uploading h5ad file: {e}")  # noqa: T201
        return 0
