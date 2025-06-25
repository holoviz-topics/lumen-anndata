import asyncio

from pathlib import Path

from lumen.ai.embeddings import OpenAIEmbeddings
from lumen.ai.llm import OpenAI
from lumen.ai.vector_store import DuckDBVectorStore

VERSION = "1.11.1"
EMBEDDINGS_DIR = Path(__file__).parent.parent / "src" / "lumen_anndata" / "embeddings"


async def start():
    """Create a DuckDB vector store and add the scanpy embeddings to it."""
    vector_store = DuckDBVectorStore(uri=str(EMBEDDINGS_DIR / "scanpy.db"), llm=OpenAI(), embeddings=OpenAIEmbeddings(), chunk_size=512)
    await vector_store.add_directory(f"scanpy_{VERSION}", pattern="*", metadata={"version": VERSION}, situate=True)


if __name__ == "__main__":
    asyncio.run(start())
