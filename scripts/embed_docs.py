import asyncio

from pathlib import Path

from lumen.ai.embeddings import HuggingFaceEmbeddings
from lumen.ai.llm import OpenAI
from lumen.ai.vector_store import DuckDBVectorStore

VERSION = "1.11.1"
EMBEDDINGS_DIR = Path(__file__).parent.parent / "src" / "lumen_anndata" / "embeddings"


async def start():
    """Create a DuckDB vector store and add the scanpy embeddings to it."""
    db_path = str(EMBEDDINGS_DIR / "scanpy.db")
    vector_store = DuckDBVectorStore(uri=db_path, llm=OpenAI(), embeddings=HuggingFaceEmbeddings(), chunk_size=512)
    print(db_path)
    await vector_store.add_directory(
        f"scanpy_{VERSION}/html", pattern="*.html", exclude_patterns=["index.html", "api/*.html"],
        metadata={"version": VERSION}, situate=True
    )


if __name__ == "__main__":
    asyncio.run(start())
