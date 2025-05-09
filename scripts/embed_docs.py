import asyncio
from lumen.ai.llm import OpenAI
from lumen.ai.embeddings import OpenAIEmbeddings
from lumen.ai.vector_store import DuckDBVectorStore


async def start():
    vector_store = DuckDBVectorStore(uri="scanpy.db", llm=OpenAI(), embeddings=OpenAIEmbeddings(), chunk_size=512)
    await vector_store.add_directory(
        "scanpy_1.11.1", pattern="*", metadata={"version": "1.11.1"}, situate=True
    )


if __name__ == "__main__":
    asyncio.run(start())
