"""
Lumen AI - Scanpy Explorer.

This is a simple web application that allows users to explore Scanpy datasets using Lumen AI.
"""

import logging

from pathlib import Path

import anndata as ad
import lumen.ai as lmai
import panel as pn
import pooch

from .source import AnnDataSource

pn.config.disconnect_notification = "Connection lost, try reloading the page!"
pn.config.ready_notification = "Application fully loaded."
pn.extension("filedropper")

INSTRUCTIONS = """
You are an expert scientist working in Python, with a specialty using Anndata and Scanpy.
Help the user with their questions, and if you don't know the answer, say so.
"""

db_uri = str(Path(__file__).parent / "embeddings" / "scanpy.db")
vector_store = lmai.vector_store.DuckDBVectorStore(
    uri=db_uri, embeddings=lmai.embeddings.OpenAIEmbeddings()
)
doc_lookup = lmai.tools.VectorLookupTool(vector_store=vector_store, n=3)


def upload_h5ad(file, table) -> int:
    """
    Uploads an h5ad file and returns an AnnDataSource.
    """
    adata = ad.read_h5ad(file)
    try:
        src = AnnDataSource(adata=adata)
        lmai.memory["sources"] = lmai.memory["sources"] + [src]
        lmai.memory["source"] = src
        return 1
    except Exception as e:
        print(f"Error uploading file: {e}")
        return 0


fname_brca = pooch.retrieve(
    url="https://storage.googleapis.com/tcga-anndata-public/test2025-04/brca_test.h5ad",
    known_hash="md5:0e17ecf3716174153bc31988ba6dd161",
)

brca_ad = ad.read_h5ad(fname_brca)
logging.debug(f"AnnData Loaded: {brca_ad}")

brca = AnnDataSource(brca_ad)
logging.debug(f"AnnDataSource: {brca}")

ui = lmai.ExplorerUI(
    data=brca,
    agents=[
        lmai.agents.ChatAgent(
            tools=[doc_lookup],
            template_overrides={"main": {"instructions": INSTRUCTIONS}},
        )
    ],
    llm=lmai.llm.OpenAI(),
    default_agents=[],
    log_level="debug",
)
ui.servable()
