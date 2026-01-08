import asyncio

from io import BytesIO

import cellxgene_census
import panel as pn
import param
import s3fs

from lumen.ai.controls import BaseSourceControls, DownloadControls
from panel.layout import Row
from panel.pane import Markdown
from panel.widgets import Tabulator
from panel_material_ui import Column, Tabs

from .utils import upload_h5ad


class CellXGeneSourceControls(DownloadControls):
    """Simple tabulator browser for CELLxGENE Census datasets"""

    active = param.Integer(default=1, doc="Active tab index")

    census_version = param.String("2025-01-30")

    input_placeholder = param.String(
        default="Select datasets by clicking the download icon, or input custom URLs, delimited by new lines",
        doc="Placeholder text for input field",
    )

    uri = param.String(default=None, doc="Base URI for CELLxGENE Census")

    status = param.String(
        default="*Click on download icons to ingest datasets.*",
        doc="Message displayed in the UI",
    )

    soma_kwargs = param.Dict(default={}, doc="Additional parameters for soma connection")

    table_upload_callbacks = param.Dict(default={
        ".h5ad": upload_h5ad,
    })

    label = '<span class="material-icons" style="vertical-align: middle;">biotech</span> CELLxGENE Census Datasets'

    def __init__(self, **params):
        BaseSourceControls.__init__(self, **params)
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
            # Client-side filtering in headers
            header_filters=filters,
            on_click=self._ingest_h5ad,
            # Row content function for technical details
            row_content=self._get_row_content,
            # Column configuration
            titles={
                "collection_name": "Collection",
                "dataset_title": "Dataset Title",
                "dataset_total_cell_count": "Cells",
                "dataset_id": "Dataset ID",
            },
            buttons={"download": '<i class="fa fa-download"></i>'},
            # Column widths
            widths={"download": "2%", "collection_name": "40%", "dataset_title": "35%", "dataset_total_cell_count": "8%", "dataset_id": "10%"},
            # Formatters
            formatters={"dataset_total_cell_count": {"type": "money", "thousand": ",", "symbol": ""}},
            # Disable editing
            editors={"collection_name": None, "dataset_title": None, "dataset_total_cell_count": None, "dataset_id": None},
        )
        self._layout = Column(
            Markdown(
                object="*Click on download icons to ingest datasets.*",
                margin=(0, 10),
            ),
            self._tabulator,
            self._error_placeholder,
            self._message_placeholder,
            self._progress_bar,
            self._progress_description,
            loading=True
        )
        pn.state.onload(self._onload)

    @pn.cache
    def _load_datasets_catalog(self, census_version: str, uri: str, **soma_kwargs):
        with cellxgene_census.open_soma(census_version=census_version, uri=uri, **soma_kwargs) as census:
            return census["census_info"]["datasets"].read().concat().to_pandas()

    def _onload(self):
        try:
            self.datasets_df = self._load_datasets_catalog(self.census_version, self.uri, **self.soma_kwargs)
        except Exception as e:
            pn.state.notifications.error(f"Failed to load datasets: {e}")
            self.status = "Failed to load datasets. Please check your connection or parameters."
            return
        # Select only user-friendly columns for the main table
        display_df = self.datasets_df[
            [
                "collection_name",
                "dataset_title",
                "dataset_total_cell_count",
                "dataset_id",  # Keep this for row content lookup
            ]
        ]
        self._tabulator.value = display_df
        self._layout.loading = False

    def _get_row_content(self, row):
        """
        Get technical details for expanded row content

        Args:
            row (pd.Series): The row data from the main table

        Returns:
            pn.pane: Panel object with technical details
        """
        dataset_id = row["dataset_id"]

        # Get full dataset info from the cached dataframe
        full_info = self.datasets_df[self.datasets_df["dataset_id"] == dataset_id].iloc[0]

        # Build technical details HTML
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
        text_content = "\n".join(lines)
        return Markdown(text_content, sizing_mode="stretch_width", styles={"color": "black"})

    async def _download_file(self, locator, chunk_size=5_000_000, max_concurrency=8):
        fs = s3fs.S3FileSystem(
            config_kwargs={"user_agent": "lumen-anndata"},
            anon=True,
            asynchronous=True,
            cache_regions=True,
        )
        session = await fs.set_session()

        info = await fs._info(locator["uri"])
        total_size = int(info["size"])
        downloaded = 0
        self._setup_progress_bar(total_size)

        await asyncio.sleep(0.01)

        # Prepare output buffer
        buf = BytesIO()
        buf.seek(total_size - 1)
        buf.write(b"\0")   # allocate size
        buf.seek(0)

        sem = asyncio.Semaphore(max_concurrency)

        async def fetch_range(start, end):
            nonlocal downloaded
            # S3 range is [start, end), end exclusive in s3fs
            async with sem:
                data = await fs._cat_file(locator["uri"], start=start, end=end)
                # place in the correct spot
                buf.seek(start)
                buf.write(data)
                # update progress
                downloaded += len(data)
                progress = min((downloaded / total_size) * 100, 100)
                self._progress_bar.value = progress
                self._progress_description.object = f"{progress}%"
                await asyncio.sleep(0.01)

        ranges = []
        for start in range(0, total_size, chunk_size):
            end = min(start + chunk_size, total_size)
            ranges.append((start, end))

        try:
            tasks = [asyncio.create_task(fetch_range(s, e)) for s, e in ranges]
            await asyncio.gather(*tasks)
        finally:
            await session.close()
            self._progress_description.object = ""
            self._progress_bar.visible = False

        buf.seek(0)
        return buf

    @param.depends("add", watch=True)
    def _on_add(self):
        if not self._upload_cards:
            return
        self._process_files()
        self._count += 3
        self._clear_uploads()
        self.param.trigger('upload_successful')

    async def _ingest_h5ad(self, event):
        """
        Uploads an h5ad file and returns an AnnDataSource.
        """
        with self.param.update(disabled=True):
            dataset_id = self.datasets_df.loc[event.row, "dataset_id"]
            dataset_title = self.datasets_df.loc[event.row, "dataset_title"]
            self._setup_progress_bar(0)
            self._progress_description.object = f"Fetching S3 URL for {dataset_title}"
            await asyncio.sleep(0.05)  # yield the event loop to ensure UI updates
            locator = cellxgene_census.get_source_h5ad_uri(dataset_id, census_version=self.census_version)
            self._progress_description.object = f"Beginning file download for {dataset_title}"
            file_buffer = await self._download_file(locator)
            downloaded_files = {f"{dataset_title}.h5ad": file_buffer}
            self._generate_file_cards(downloaded_files)
            self.param.trigger("add")  # automatically trigger the add
            self.status = f"Dataset '{dataset_title}' has been added successfully."
