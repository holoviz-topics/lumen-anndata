import asyncio

from io import BytesIO

import cellxgene_census
import panel as pn
import param
import s3fs

from lumen.ai.controls import SourceControls


class CellXGeneSourceControls(SourceControls):
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

    def __init__(self, **params):
        super().__init__(**params)
        filters = {
            "collection_name": {"type": "input", "func": "like", "placeholder": "Enter name"},
            "dataset_title": {"type": "input", "func": "like", "placeholder": "Enter title"},
            "dataset_total_cell_count": {"type": "number", "func": ">=", "placeholder": "Enter min cells"},
            "dataset_id": {"type": "input", "func": "like", "placeholder": "Enter ID"},
        }
        self._tabulator = pn.widgets.Tabulator(
            page_size=5,
            pagination="local",
            sizing_mode="stretch_width",
            show_index=False,
            # Client-side filtering in headers
            header_filters=filters,
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
            loading=True,
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
        self._tabulator.on_click(self._ingest_h5ad)
        self._tabulator.param.update(
            value=display_df,
            loading=False,
        )
        # self._czi_controls.loading = False

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
        return pn.pane.Markdown(text_content, sizing_mode="stretch_width", styles={"color": "black"})

    async def _ingest_h5ad(self, event):
        """
        Uploads an h5ad file and returns an AnnDataSource.
        """
        with self._tabulator.param.update(loading=True), self.param.update(disabled=True):
            await asyncio.sleep(0.05)  # yield the event loop to ensure UI updates
            dataset_id = self.datasets_df.loc[event.row, "dataset_id"]
            dataset_title = self.datasets_df.loc[event.row, "dataset_title"]
            locator = cellxgene_census.get_source_h5ad_uri(dataset_id, census_version=self.census_version)
            # Initialize s3fs
            fs = s3fs.S3FileSystem(
                config_kwargs={"user_agent": "lumen-anndata"},
                anon=True,
                cache_regions=True,
            )
            buffer = BytesIO()
            with fs.open(locator["uri"], "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    buffer.write(chunk)
            buffer.seek(0)  # reset for reading
            self.downloaded_files = {f"{dataset_title}.h5ad": buffer}
            self.param.trigger("trigger_add")  # automatically trigger the add
            self.status = f"Dataset '{dataset_title}' has been added successfully."

    def __panel__(self):
        self._czi_controls = pn.Column(
            pn.pane.Markdown(
                object="*Click on download icons to ingest datasets.*",
                margin=0,
            ),
            self._tabulator,
            loading=True
        )
        original_controls = super().__panel__()
        return pn.Column(
            pn.Tabs(
                ("Upload Datasets", original_controls),
                ("CELLxGENE Census Datasets", self._czi_controls),
                dynamic=True,
            ),
        )
