from io import BytesIO

import anndata as ad

from lumen_anndata.source import AnnDataSource


def upload_h5ad(context, file: BytesIO, table: str, filename: str) -> AnnDataSource | None:
    """
    Uploads an h5ad file and returns an AnnDataSource.
    """
    adata = ad.read_h5ad(file)
    try:
        src = AnnDataSource(adata=adata, uploaded_filename=filename)
        for table in src.get_tables():
            if table not in src.metadata:
                src.metadata[table] = {}
            if "filename" not in src.metadata[table]:
                src.metadata[table]["filename"] = filename
        return src
    except Exception as e:
        print(f"Error uploading h5ad file: {e}")  # noqa: T201
        return None
