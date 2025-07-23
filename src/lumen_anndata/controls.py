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

    soma_kwargs = param.Dict(default={}, doc="Additional parameters for soma connection")

    def __init__(self, **params):
        super().__init__(**params)

        self.datasets_df = self._load_datasets_catalog(self.census_version, self.uri, **self.soma_kwargs)

        # Select only user-friendly columns for the main table
        display_df = self.datasets_df[
            [
                "collection_name",
                "dataset_title",
                "dataset_total_cell_count",
                "dataset_id",  # Keep this for row content lookup
            ]
        ].copy()

        filters = {
            "collection_name": {"type": "input", "func": "like", "placeholder": "Enter name"},
            "dataset_title": {"type": "input", "func": "like", "placeholder": "Enter title"},
            "dataset_total_cell_count": {"type": "number", "func": ">=", "placeholder": "Enter min cells"},
            "dataset_id": {"type": "input", "func": "like", "placeholder": "Enter ID"},
        }

        # Create tabulator
        self._tabulator = pn.widgets.Tabulator(
            display_df,
            page_size=10,
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
        )
        self._tabulator.on_click(self._ingest_h5ad)

    @pn.cache
    def _load_datasets_catalog(self, census_version: str, uri: str, **soma_kwargs):
        with cellxgene_census.open_soma(census_version=census_version, uri=uri, **soma_kwargs) as census:
            return census["census_info"]["datasets"].read().concat().to_pandas()

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
        html_content = f"""
        <div>
            <h4>📋 Technical Details</h4>
            <div>
                <h5>🔗 Identifiers & Links</h5>
                <p><strong>Dataset ID:</strong> <code>{dataset_id}</code></p>
                <p><strong>Dataset Version ID:</strong> <code>{full_info.get("dataset_version_id", "N/A")}</code></p>
                <p><strong>Collection DOI:</strong>
                   <a href="https://doi.org/{full_info.get("collection_doi", "")}" target="_blank">
                   {full_info.get("collection_doi", "N/A")}
                   </a>
                </p>
                <p><strong>Collection DOI Label:</strong> {full_info.get("collection_doi_label", "N/A")}</p>
            </div>

            <div>
                <h5>📁 File Information</h5>
                <p><strong>H5AD Path:</strong> <code>{full_info.get("dataset_h5ad_path", "N/A")}</code></p>
                <p><strong>Total Cells:</strong> {full_info.get("dataset_total_cell_count", "N/A"):,}</p>
            </div>
        </div>
        """

        return pn.pane.HTML(html_content, sizing_mode="stretch_width")

    async def _ingest_h5ad(self, event):
        """
        Uploads an h5ad file and returns an AnnDataSource.
        """
        with self._tabulator.param.update(loading=True), self.param.update(disabled=True):
            await asyncio.sleep(0.05)  # yield the event loop to ensure UI updates
            dataset_id = self.datasets_df.loc[event.row, "dataset_id"]
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
            self.downloaded_files = {f"{dataset_id}.h5ad": buffer}

    def __panel__(self):
        original_controls = super().__panel__()
        return pn.Column(
            original_controls,
            self._tabulator,
        )
