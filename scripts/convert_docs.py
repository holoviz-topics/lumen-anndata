import asyncio
import re
from pathlib import Path

from tqdm.auto import tqdm
from markitdown import MarkItDown
from docling.document_converter import DocumentConverter

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


async def convert_documents(
    source_dir: str,
    output_dir: str,
    sample_size: int = 3,
    file_pattern: str = "*.html",
) -> Path:
    """
    Convert documents from source directory to markdown files in output directory.

    Args:
        source_dir: Source directory containing documentation
        output_dir: Output directory for converted markdown files
        sample_size: Number of documents to sample for finding common lines
        file_pattern: Glob pattern to match source files
        skip_conversion: If True, skip conversion and assume output_dir already has markdown files

    Returns:
        Path to output directory with converted files
    """
    source_dir_path = Path(source_dir)
    output_dir_path = Path(output_dir)

    # Create output directory
    output_dir_path.mkdir(exist_ok=True, parents=True)

    if file_pattern.endswith(".html"):
        dc = DocumentConverter()
        mid = MarkItDown()

        # Find all matching files
        paths = list(source_dir_path.rglob(file_pattern))

        if not paths:
            raise FileNotFoundError(
                f"No files matching {file_pattern} found in {source_dir_path}"
            )

        print(f"Found {len(paths)} files to convert")

        # Find common lines to remove (headers, footers, etc.)
        sample_size = min(sample_size, len(paths))
        if sample_size > 1:
            print(f"Sampling {sample_size} documents to find common lines")
            texts = [
                dc.convert(str(p)).document.export_to_markdown()
                for p in paths[:sample_size]
            ]
            shared = set(texts[0].splitlines()).intersection(
                *(t.splitlines() for t in texts[1:])
            )

            # Add common known lines that might appear in documentation
            known_common_lines = [
                "Open this notebook in Jupyterlite | Download this notebook from GitHub (right-click to download)."
            ]
            for line in known_common_lines:
                shared.add(line)

            print(f"Found {len(shared)} common lines to remove")
        else:
            shared = set()

        # Convert each file
        shortest_line = ("", float("inf"))
        for path in tqdm(paths, desc="Converting documents", unit="file"):
            if "<dt>" in path.read_text():
                text = mid.convert(str(path)).text_content
                # find where the first # heading is
                first_heading = text.find("# ")
                footer = text.find("\n[previous")
                if first_heading != -1:
                    # remove everything before the first # heading
                    text = text[first_heading:footer]
            else:
                text = dc.convert(str(path)).document.export_to_markdown()

            # Remove common lines
            for line in shared:
                text = text.replace(line, "")

            # Collapse multiple newlines
            text = re.sub(r"\n{3,}", "\n\n", text)

            # Write to output file
            output_path = output_dir_path / path.relative_to(
                source_dir_path
            ).with_suffix(".md")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text, encoding="utf-8")

            num_lines = len(text.splitlines())
            if num_lines < shortest_line[1]:
                shortest_line = (output_path, num_lines)
            if num_lines < 10:
                print(f"Warning: {output_path} (from {path}) has only {num_lines} lines")

    print(f"Shortest line: {shortest_line[0]} ({shortest_line[1]} lines)")
    return output_dir_path


if __name__ == "__main__":
    asyncio.run(
        convert_documents(
            source_dir="scanpy.readthedocs.io/",
            output_dir="scanpy",
            sample_size=5,
            file_pattern="*.html",
        )
    )
