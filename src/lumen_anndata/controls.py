import asyncio

from io import BytesIO

import panel as pn
import param
import s3fs

from lumen.ai.controls import CatalogSourceControls, SourceResult
from panel.pane import Markdown

from .utils import upload_h5ad


class CellXGeneSourceControls(CatalogSourceControls):
    """Browser for CELLxGENE Census datasets."""

    census_version = param.String("2025-01-30")

    uri = param.String(default=None, doc="Base URI for CELLxGENE Census")

    soma_kwargs = param.Dict(default={}, doc="Additional parameters for soma connection")

    load_mode = "manual"  # Row clicks trigger loading, not a button

    label = '<span class="material-icons" style="vertical-align: middle;">biotech</span> CELLxGENE Census Datasets'

    display_columns = param.Dict(default={
        "collection_name": {"title": "Collection", "width": "40%"},
        "dataset_title":   {"title": "Dataset Title", "width": "35%"},
        "dataset_total_cell_count": {
            "title": "Cells",
            "width": "8%",
            "formatter": {"type": "money", "thousand": ",", "symbol": ""},
        },
        "dataset_id": {"title": "Dataset ID", "width": "10%"},
    })

    filter_columns = param.Dict(default={
        "collection_name":          {"type": "input",  "func": "like", "placeholder": "Enter name"},
        "dataset_title":            {"type": "input",  "func": "like", "placeholder": "Enter title"},
        "dataset_total_cell_count": {"type": "number", "func": ">=",   "placeholder": "Enter min cells"},
        "dataset_id":               {"type": "input",  "func": "like", "placeholder": "Enter ID"},
    })

    search_columns = param.List(default=[
        "collection_name",
        "dataset_title",
        "dataset_id",
    ])

    detail_columns = param.List(default=[
        "dataset_version_id",
        "collection_doi",
        "collection_doi_label",
        "dataset_h5ad_path",
        "dataset_total_cell_count",
    ])

    # ──────────────────────────────────────────────────────────────────────────
    # CatalogSourceControls contract
    # ──────────────────────────────────────────────────────────────────────────

    async def _load_catalog(self):
        """Fetch and return the CELLxGENE datasets catalog as a DataFrame."""
        return await asyncio.to_thread(
            self._fetch_datasets_catalog,
            self.census_version,
            self.uri,
            **self.soma_kwargs,
        )

    async def _fetch_entry(self, entry) -> SourceResult:
        """Download and process a single CELLxGENE dataset."""
        import cellxgene_census

        dataset_id = entry["dataset_id"]
        dataset_title = entry["dataset_title"]

        self.progress(f"Fetching S3 URL for {dataset_title}...")
        await asyncio.sleep(0.01)

        locator = cellxgene_census.get_source_h5ad_uri(
            dataset_id, census_version=self.census_version
        )

        self.progress(f"Downloading {dataset_title}...")
        file_buffer = await self._download_file(locator)

        self.progress(f"Processing {dataset_title}...")
        source = upload_h5ad(self.context, file_buffer, dataset_title, dataset_title)

        if source is None:
            return SourceResult.empty(f"Failed to process {dataset_title}")

        return SourceResult.from_source(
            source,
            message=f"Dataset '{dataset_title}' loaded successfully.",
        )

    def _get_row_content(self, row):
        """Generate expanded row content with technical details."""
        dataset_id = row["dataset_id"]
        full_info = self.catalog_df[self.catalog_df["dataset_id"] == dataset_id].iloc[0]

        lines = [
            "Technical Details",
            "",
            "Identifiers & Links",
            f"  Dataset ID: {dataset_id}",
            f"  Dataset Version ID: {full_info.get('dataset_version_id', 'N/A')}",
            f"  Collection DOI: {full_info.get('collection_doi', 'N/A')}",
            f"  Collection DOI Label: {full_info.get('collection_doi_label', 'N/A')}",
            "",
            "File Information",
            f"  H5AD Path: {full_info.get('dataset_h5ad_path', 'N/A')}",
            f"  Total Cells: {full_info.get('dataset_total_cell_count', 'N/A'):,}",
        ]
        return Markdown(
            "\n".join(lines),
            sizing_mode="stretch_width",
            styles={"color": "black"},
        )

    # ──────────────────────────────────────────────────────────────────────────
    # CELLxGENE-specific helpers
    # ──────────────────────────────────────────────────────────────────────────

    @pn.cache
    def _fetch_datasets_catalog(self, census_version: str, uri: str, **soma_kwargs):
        """Fetch and cache the datasets catalog (synchronous, run in thread)."""
        import cellxgene_census
        with cellxgene_census.open_soma(
            census_version=census_version, uri=uri, **soma_kwargs
        ) as census:
            return census["census_info"]["datasets"].read().concat().to_pandas()

    async def _download_file(self, locator, chunk_size=5_000_000, max_concurrency=8):
        """Download file from S3 with progress reporting."""
        fs = s3fs.S3FileSystem(
            config_kwargs={"user_agent": "lumen-anndata"},
            anon=True,
            asynchronous=True,
            cache_regions=True,
        )
        await fs.set_session()

        info = await fs._info(locator["uri"])
        total_size = int(info["size"])

        self.progress("Downloading...", current=0, total=total_size)
        downloaded = 0

        buf = BytesIO()
        buf.seek(total_size - 1)
        buf.write(b"\0")
        buf.seek(0)

        sem = asyncio.Semaphore(max_concurrency)

        async def fetch_range(start, end):
            nonlocal downloaded
            async with sem:
                data = await fs._cat_file(locator["uri"], start=start, end=end)
                buf.seek(start)
                buf.write(data)
                downloaded += len(data)
                self.progress(current=downloaded, total=total_size)
                await asyncio.sleep(0.01)

        ranges = [
            (start, min(start + chunk_size, total_size))
            for start in range(0, total_size, chunk_size)
        ]
        tasks = [asyncio.create_task(fetch_range(s, e)) for s, e in ranges]
        await asyncio.gather(*tasks)

        buf.seek(0)
        return buf
