import asyncio
from pathlib import Path

from tqdm.auto import tqdm
from lumen.ai.vector_store import DuckDBVectorStore
from lumen.ai.embeddings import OpenAIEmbeddings
from lumen.ai.llm import OpenAI


def filter_paths(
    base_dir: str,
    file_pattern: str = "*.md",
    exclude_patterns: list[str] | None = None,
) -> list[Path]:
    """
    Filter paths based on inclusion and exclusion patterns.

    Args:
        base_dir: Base directory to search
        file_pattern: Glob pattern for files to include
        exclude_patterns: List of patterns to exclude

    Returns:
        List of filtered paths
    """
    base_dir_path = Path(base_dir)

    # Default exclude patterns for documentation sites
    if exclude_patterns is None:
        exclude_patterns = [
            "search.md",
            "genindex.md",
            "py-modindex.md",
            "releases.md",
            "_static",
            "pyodide",
        ]

    # Find all matching files
    all_paths = list(base_dir_path.rglob(file_pattern))

    # Filter out excluded paths
    filtered_paths = []
    for path in all_paths:
        path_str = str(path)
        if not any(pattern in path_str for pattern in exclude_patterns):
            filtered_paths.append(path)

    print(f"Found {len(filtered_paths)} files after filtering (from {len(all_paths)} total)")
    return filtered_paths


async def index_documents(paths: list[Path], db_name: str, error_log: str = "error.log") -> None:
    """
    Index documents in the vector store.

    Args:
        paths: List of paths to index
        db_name: Name of the vector store database
        error_log: Path to error log file
    """
    vs = DuckDBVectorStore(
        embeddings=OpenAIEmbeddings(),
        uri=db_name,
        situate=True,
        llm=OpenAI(),
    )

    for path in tqdm(paths, desc="Indexing documents", unit="file"):
        try:
            await vs.add_file(path, upsert=True)
        except Exception as e:
            with open(error_log, "a") as f:
                f.write(f"Error processing {path}: {e}\n")
            print(f"Error processing {path}: {e}")


async def process_docs(
    output_dir: str,
    db_name: str,
    index_pattern: str = "*.md",
    exclude_patterns: list[str] = None,
) -> None:
    """
    Run the full conversion and indexing process.

    Args:
        output_dir: Output directory for converted markdown files
        db_name: Name of the vector database file (e.g. 'scanpy.db')
        index_pattern: File pattern to match for indexing
        exclude_patterns: Patterns to exclude from indexing
    """
    # Filter paths
    paths = filter_paths(
        base_dir=output_dir,
        file_pattern=index_pattern,
        exclude_patterns=exclude_patterns,
    )

    # Index documents
    await index_documents(paths=paths, db_name=db_name)


if __name__ == "__main__":
    asyncio.run(
        process_docs(
            output_dir="scanpy",
            db_name="scanpy.db",
            index_pattern="*.md",
            exclude_patterns=[
                "search.md",
                "genindex.md",
                "py-modindex.md",
                "releases.md",
                "_static",
                "pyodide",
            ],
        )
    )
