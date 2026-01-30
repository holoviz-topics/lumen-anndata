import asyncio

from io import BytesIO

import cellxgene_census
import panel as pn
import param
import s3fs

from lumen.ai.controls import BaseSourceControls, SourceResult
from panel.pane import Markdown
from panel.widgets import Tabulator

from .utils import upload_h5ad


class CellXGeneSourceControls(BaseSourceControls):
    """Browser for CELLxGENE Census datasets with Tabulator interface."""

    census_version = param.String("2025-01-30")

    uri = param.String(default=None, doc="Base URI for CELLxGENE Census")

    soma_kwargs = param.Dict(default={}, doc="Additional parameters for soma connection")

    upload_handlers = param.Dict(default={
        ".h5ad": upload_h5ad,
    })

    load_mode = "manual"  # Row clicks trigger loading, not a button

    label = '<span class="material-icons" style="vertical-align: middle;">biotech</span> CELLxGENE Census Datasets'

    def __init__(self, **params):
        super().__init__(**params)
        self._layout.loading = True
        pn.state.onload(self._load_catalog)

    def _render_controls(self):
        """Build the Tabulator browser."""
        filters = {
            "collection_name": {"type": "input", "func": "like", "placeholder": "Enter name"},
            "dataset_title": {"type": "input", "func": "like", "placeholder": "Enter title"},
            "dataset_total_cell_count": {"type": "number", "func": ">=", "placeholder": "Enter min cells"},
            "dataset_id": {"type": "input", "func": "like", "placeholder": "Enter ID"},
        }
        self._tabulator = Tabulator(
            page_size=5,
            pagination="local",
            sizing_mode="stretch_width",
            show_index=False,
            header_filters=filters,
            on_click=self._on_row_click,
            row_content=self._get_row_content,
            titles={
                "collection_name": "Collection",
                "dataset_title": "Dataset Title",
                "dataset_total_cell_count": "Cells",
                "dataset_id": "Dataset ID",
            },
            buttons={"download": '<i class="fa fa-download"></i>'},
            widths={
                "download": "2%",
                "collection_name": "40%",
                "dataset_title": "35%",
                "dataset_total_cell_count": "8%",
                "dataset_id": "10%"
            },
            formatters={
                "dataset_total_cell_count": {"type": "money", "thousand": ",", "symbol": ""}
            },
            editors={
                "collection_name": None,
                "dataset_title": None,
                "dataset_total_cell_count": None,
                "dataset_id": None
            },
        )

        return [
            Markdown("*Click on download icons to ingest datasets.*", margin=(0, 10)),
            self._tabulator,
        ]

    @pn.cache
    def _fetch_datasets_catalog(self, census_version: str, uri: str, **soma_kwargs):
        """Fetch and cache the datasets catalog."""
        with cellxgene_census.open_soma(
            census_version=census_version, uri=uri, **soma_kwargs
        ) as census:
            return census["census_info"]["datasets"].read().concat().to_pandas()

    def _load_catalog(self):
        """Load the datasets catalog on page ready."""
        try:
            self.datasets_df = self._fetch_datasets_catalog(
                self.census_version, self.uri, **self.soma_kwargs
            )
        except Exception as e:
            pn.state.notifications.error(f"Failed to load datasets: {e}")
            self._show_message(f"Failed to load datasets: {e}", error=True)
            self._layout.loading = False
            return

        # Select display columns
        display_df = self.datasets_df[[
            "collection_name",
            "dataset_title",
            "dataset_total_cell_count",
            "dataset_id",
        ]]
        self._tabulator.value = display_df
        self._layout.loading = False

    def _get_row_content(self, row):
        """Generate expanded row content with technical details."""
        dataset_id = row["dataset_id"]
        full_info = self.datasets_df[self.datasets_df["dataset_id"] == dataset_id].iloc[0]

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
            styles={"color": "black"}
        )

    async def _on_row_click(self, event):
        """Handle row click - download the selected dataset."""
        await self._run_load(self._download_dataset(event.row))

    async def _download_dataset(self, row_idx) -> SourceResult:
        """Download and process a dataset from CELLxGENE."""
        dataset_id = self.datasets_df.loc[row_idx, "dataset_id"]
        dataset_title = self.datasets_df.loc[row_idx, "dataset_title"]

        # Get S3 URL
        self.progress(f"Fetching S3 URL for {dataset_title}...")
        locator = cellxgene_census.get_source_h5ad_uri(
            dataset_id, census_version=self.census_version
        )

        # Download file with progress
        self.progress(f"Downloading {dataset_title}...")
        file_buffer = await self._download_file(locator)

        # Process the h5ad file
        self.progress(f"Processing {dataset_title}...")
        source = upload_h5ad(self.context, file_buffer, dataset_title, dataset_title)

        if source is None:
            return SourceResult.empty(f"Failed to process {dataset_title}")

        return SourceResult.from_source(
            source,
            message=f"Dataset '{dataset_title}' loaded successfully."
        )

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

        # Setup progress tracking
        self.progress("Downloading...", current=0, total=total_size)
        downloaded = 0

        # Allocate buffer
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

        # Build range requests
        ranges = [
            (start, min(start + chunk_size, total_size))
            for start in range(0, total_size, chunk_size)
        ]

        # Execute parallel downloads
        tasks = [asyncio.create_task(fetch_range(s, e)) for s, e in ranges]
        await asyncio.gather(*tasks)

        buf.seek(0)
        return buf
