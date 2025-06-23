"""
Tests for lumen_anndata.source.AnnDataSource
"""

import anndata as ad
import numpy as np
import pandas as pd
import param
import pytest
import scipy.sparse as sp

from lumen_anndata.operations import AnnDataOperation
from lumen_anndata.source import AnnDataSource


@pytest.fixture
def fixed_sample_anndata():
    X = np.array([[1, 0, 3], [0, 5, 0], [2, 0, 0], [0, 1, 1]], dtype=np.float32)
    obs_df = pd.DataFrame(
        {"cell_type": pd.Categorical(["B", "T", "B", "NK"]), "n_genes": [10, 20, 5, 15], "sample_name": ["1261A", "1262C", "1263B", "1264D"]},
        index=["cell_0", "cell_1", "cell_2", "cell_3"],
    )
    var_df = pd.DataFrame(
        {
            "gene_type": pd.Categorical(["coding", "noncoding", "coding"]),
            "highly_variable": [True, False, True],
        },
        index=["gene_A", "gene_B", "gene_C"],
    )
    adata = ad.AnnData(X, obs=obs_df, var=var_df)
    adata.layers["counts"] = X * 2
    adata.obsm["X_pca"] = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    adata.uns["info"] = {"version": "1.0"}
    return adata


@pytest.fixture
def sample_anndata():
    """Create a sample AnnData object with various components for testing."""
    # Create core data matrix (sparse)
    n_obs, n_vars = 100, 50
    data = np.random.poisson(1, size=(n_obs, n_vars)).astype(np.float32)
    X = sp.csr_matrix(data)

    # Create observation metadata
    obs_df = pd.DataFrame(
        {
            "cell_type": pd.Categorical(np.random.choice(["B", "T", "NK"], size=n_obs)),
            "n_genes_by_counts": np.random.randint(5000, 10000, size=n_obs),
            "sample_id": np.random.choice(["sample1", "sample2", "sample3"], size=n_obs),
        }
    )

    # Create variable metadata
    var_df = pd.DataFrame(
        {
            "gene_type": pd.Categorical(np.random.choice(["protein_coding", "lncRNA", "miRNA"], size=n_vars)),
            "highly_variable": np.random.choice([True, False], size=n_vars),
        }
    )

    # Create AnnData object
    adata = ad.AnnData(X, obs=obs_df, var=var_df)

    # Add index names
    adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]

    # Add layers
    adata.layers["normalized"] = np.log1p(data)
    adata.layers["binary"] = (data > 0).astype(np.float32)

    # Add multidimensional arrays
    adata.obsm["X_pca"] = np.random.normal(0, 1, size=(n_obs, 10))
    adata.obsm["X_umap"] = np.random.normal(0, 1, size=(n_obs, 2))
    adata.varm["PCs"] = np.random.normal(0, 1, size=(n_vars, 10))

    # Add pairwise matrices
    adata.obsp["distances"] = sp.csr_matrix(np.random.exponential(1, size=(n_obs, n_obs)))
    adata.varp["correlations"] = sp.csr_matrix(np.random.normal(0, 1, size=(n_vars, n_vars)))

    # Add unstructured data
    adata.uns["clustering_params"] = {"resolution": 0.8, "method": "leiden"}
    adata.uns["metadata"] = {"experiment_date": "2025-01-01", "operator": "Test User"}
    adata.uns["colors"] = ["red", "blue", "green"]

    return adata


def test_initialization(sample_anndata):
    """Test initialization of AnnDataSource with various parameters."""
    # Test initialization with AnnData object
    source = AnnDataSource(adata=sample_anndata)

    assert source._adata_store is not None
    assert source._component_registry, "Component registry should not be empty"
    assert "obs" in source._materialized_tables
    assert "var" in source._materialized_tables
    assert len(source._materialized_tables) == 2  # Initially only obs and var

    # Check that obs and var tables are correctly prepared
    obs_df_sql = source.execute("SELECT * FROM obs ORDER BY obs_id LIMIT 2")
    pd.testing.assert_series_equal(
        obs_df_sql["obs_id"].astype(str).reset_index(drop=True), pd.Series(sample_anndata.obs_names[:2].astype(str)).reset_index(drop=True), check_names=False
    )
    assert "cell_type" in obs_df_sql.columns
    assert obs_df_sql["cell_type"].tolist() == sample_anndata.obs["cell_type"].iloc[:2].tolist()

    var_df_sql = source.execute("SELECT * FROM var ORDER BY var_id LIMIT 2")
    pd.testing.assert_series_equal(
        var_df_sql["var_id"].astype(str).reset_index(drop=True), pd.Series(sample_anndata.var_names[:2].astype(str)).reset_index(drop=True), check_names=False
    )
    assert "gene_type" in var_df_sql.columns

    # Test initialization parameters
    source_with_params = AnnDataSource(adata=sample_anndata, filter_in_sql=False)
    assert source_with_params.filter_in_sql is False
    assert source_with_params._obs_ids_selected is None  # Check initial selection state
    assert source_with_params._var_ids_selected is None


def test_get_tables(sample_anndata):
    """Test the get_tables method."""
    source = AnnDataSource(adata=sample_anndata)

    expected_tables = {
        "obs",
        "var",
        "obsm_X_pca",
        "obsm_X_umap",
        "varm_PCs",
    }
    all_tables = source.get_tables()
    assert set(all_tables) == expected_tables

    materialized_tables = source.get_tables(materialized_only=True)
    assert set(materialized_tables) == {"obs", "var"}


def test_execute_basic_queries_fixed(fixed_sample_anndata):
    source = AnnDataSource(adata=fixed_sample_anndata)
    obs_result = source.execute("SELECT * FROM  obs WHERE cell_type = 'B' ORDER BY obs_id")
    expected_obs_b = fixed_sample_anndata.obs[fixed_sample_anndata.obs["cell_type"] == "B"].copy()
    expected_obs_b["obs_id"] = expected_obs_b.index.astype(str)
    pd.testing.assert_frame_equal(
        obs_result.reset_index(drop=True),
        expected_obs_b.reset_index(drop=True).sort_values("obs_id").reset_index(drop=True),
        check_dtype=False,
        check_categorical=False,
    )

    agg_result = source.execute("SELECT cell_type, SUM(n_genes) as total_genes FROM obs GROUP BY cell_type ORDER BY cell_type")
    expected_agg = fixed_sample_anndata.obs.groupby("cell_type", observed=False)["n_genes"].sum().reset_index(name="total_genes").sort_values("cell_type")
    pd.testing.assert_frame_equal(agg_result.reset_index(drop=True), expected_agg.reset_index(drop=True), check_dtype=False, check_categorical=False)


def test_get_fixed_dataframe(fixed_sample_anndata):
    source = AnnDataSource(adata=fixed_sample_anndata)

    # Get obs table with filter
    b_cells_df = source.get("obs", cell_type="B")
    expected_b_ids = fixed_sample_anndata.obs_names[fixed_sample_anndata.obs["cell_type"] == "B"].tolist()
    pd.testing.assert_series_equal(
        b_cells_df["obs_id"].sort_values().reset_index(drop=True), pd.Series(expected_b_ids).astype(str).sort_values().reset_index(drop=True), check_names=False
    )
    assert source._obs_ids_selected is not None
    assert set(source._obs_ids_selected) == set(expected_b_ids)

    # Test with no match filter
    no_match_df = source.get("obs", cell_type="NonExistent")
    assert no_match_df.empty
    assert source._obs_ids_selected is not None
    assert len(source._obs_ids_selected) == 0


def test_get_anndata(fixed_sample_anndata):
    source = AnnDataSource(adata=fixed_sample_anndata)

    # Filter obs to 'B' cells
    b_cell_ids = fixed_sample_anndata.obs_names[fixed_sample_anndata.obs["cell_type"] == "B"].tolist()
    source.get("obs", cell_type="B")

    # Get AnnData with obs filter, and var filter directly in query
    highly_var_gene_ids = fixed_sample_anndata.var_names[fixed_sample_anndata.var["highly_variable"]].tolist()
    filtered_adata = source.get("anndata", return_type="anndata", highly_variable=True)

    assert isinstance(filtered_adata, ad.AnnData)
    pd.testing.assert_index_equal(filtered_adata.obs_names.astype(str), pd.Index(b_cell_ids).astype(str))
    assert (filtered_adata.obs["cell_type"] == "B").all()
    pd.testing.assert_index_equal(filtered_adata.var_names.astype(str), pd.Index(highly_var_gene_ids).astype(str))
    assert filtered_adata.var["highly_variable"].all()

    if "cell_0" in filtered_adata.obs_names and "gene_A" in filtered_adata.var_names:
        assert filtered_adata["cell_0", "gene_A"].X.item() == pytest.approx(fixed_sample_anndata["cell_0", "gene_A"].X.item())

    no_obs_adata = source.get("anndata", return_type="anndata", cell_type="NonExistentType")
    assert no_obs_adata.n_obs == 0
    assert no_obs_adata.n_vars == fixed_sample_anndata.n_vars


def test_materialization_on_execute(sample_anndata):
    source = AnnDataSource(adata=sample_anndata)
    assert "obsm_X_pca" not in source._materialized_tables
    source.execute("SELECT * FROM obsm_X_pca LIMIT 1")
    assert "obsm_X_pca" in source._materialized_tables


def test_get_adata_slice_labels(fixed_sample_anndata):
    source = AnnDataSource(adata=fixed_sample_anndata)
    original_index = pd.Index(["a", "b", "c", "d"])
    assert source._get_adata_slice_labels(original_index, None) == slice(None)
    assert source._get_adata_slice_labels(original_index, ["b", "d", "e"]) == ["b", "d"]
    assert source._get_adata_slice_labels(original_index, pd.Series(["c", "a", "c"])) == ["a", "c"]
    assert source._get_adata_slice_labels(original_index, np.array([1, 2], dtype=object)) == []
    assert source._get_adata_slice_labels(pd.Index([10, 20, 30]), ["10", "50"]) == ["10"]


def test_empty_adata_components():
    """Test behavior with empty AnnData objects or components."""
    empty_adata = ad.AnnData(np.empty((0, 0)))
    source = AnnDataSource(adata=empty_adata)
    tables = source.get_tables()
    assert not tables


def test_execute_basic_queries(sample_anndata):
    """Test executing basic SQL queries."""
    source = AnnDataSource(adata=sample_anndata)

    # Test simple SELECT query on obs table
    obs_result = source.execute("SELECT * FROM obs LIMIT 10")
    assert len(obs_result) == 10
    assert "obs_id" in obs_result.columns
    assert "cell_type" in obs_result.columns

    # Test filtering with WHERE clause
    filtered_obs = source.execute("SELECT * FROM obs WHERE cell_type = 'B'")
    assert all(filtered_obs["cell_type"] == "B")

    # Test query with aggregation
    agg_result = source.execute("""
        SELECT cell_type, COUNT(*) as count, AVG(n_genes_by_counts) as avg_genes
        FROM obs
        GROUP BY cell_type
    """)
    assert len(agg_result) <= 3  # Should have at most 3 cell types
    assert "count" in agg_result.columns
    assert "avg_genes" in agg_result.columns


def test_execute_multidim_queries(sample_anndata):
    """Test executing queries on multidimensional arrays."""
    source = AnnDataSource(adata=sample_anndata)

    # Query obsm component
    pca_result = source.execute("SELECT * FROM obsm_X_pca LIMIT 5")
    assert "obs_id" in pca_result.columns
    assert "X_pca_0" in pca_result.columns

    # Query varm component
    varm_result = source.execute("SELECT * FROM varm_PCs LIMIT 5")
    assert "var_id" in varm_result.columns
    assert "PCs_0" in varm_result.columns


def test_execute_with_joins(sample_anndata):
    """Test executing queries with joins between tables."""
    source = AnnDataSource(adata=sample_anndata)

    # Join obs metadata with obsm_X_pca data
    join_result = source.execute("""
        SELECT o.cell_type, COUNT(*) as pca_count
        FROM obsm_X_pca x
        JOIN obs o ON x.obs_id = o.obs_id
        GROUP BY o.cell_type
    """)

    assert "cell_type" in join_result.columns
    assert "pca_count" in join_result.columns

    # Join var metadata with varm_PCs data
    gene_pca = source.execute("""
        SELECT v.gene_type, COUNT(*) as pc_count
        FROM varm_PCs p
        JOIN var v ON p.var_id = v.var_id
        GROUP BY v.gene_type
    """)

    assert "gene_type" in gene_pca.columns
    assert "pc_count" in gene_pca.columns
    assert len(gene_pca) <= 3  # Should have at most 3 gene types


def test_get_dataframe_random(sample_anndata):
    """Test the get method returning DataFrame with random data."""
    source = AnnDataSource(adata=sample_anndata)

    # Get obs table with filter
    b_cells = source.get("obs", cell_type="B")
    assert all(b_cells["cell_type"] == "B")

    # Get with multiple filters
    filtered = source.get("obs", cell_type="B", sample_id="sample1")
    assert all(filtered["cell_type"] == "B")
    assert all(filtered["sample_id"] == "sample1")

    # Get with list filter
    multi_sample = source.get("obs", sample_id=["sample1", "sample2"])
    assert all(multi_sample["sample_id"].isin(["sample1", "sample2"]))

    # Get obsm data with selection tracking from previous query
    # (This tests that the sample_id filtered obs_ids are used for obsm_X_pca)
    pca_data = source.get("obsm_X_pca")
    assert all(np.isin(pca_data["obs_id"], filtered["obs_id"]))

    # Verify internal state tracking
    assert source._obs_ids_selected is not None
    assert len(source._obs_ids_selected) == len(filtered)


def test_get_anndata_sample(sample_anndata):
    """Test the get method returning AnnData."""
    source = AnnDataSource(adata=sample_anndata)

    # First filter the obs table to establish a selection
    source.get("obs", cell_type="T")

    # Get filtered AnnData
    filtered_adata = source.get("anndata", return_type="anndata")

    # Check if filtering was applied correctly
    assert isinstance(filtered_adata, ad.AnnData)
    assert np.all(filtered_adata.obs["cell_type"] == "T")
    assert filtered_adata.n_obs < sample_anndata.n_obs
    assert filtered_adata.n_vars == sample_anndata.n_vars  # Vars not filtered

    # Test filtering both obs and vars in the same call
    # This passes the filter directly to the AnnData getter
    filtered_adata_2 = source.get("anndata", return_type="anndata", highly_variable=True)

    # Check if both filters were applied
    assert np.all(filtered_adata_2.obs["cell_type"] == "T")
    # All variables in the result should be highly variable
    assert filtered_adata_2.var["highly_variable"].all()
    assert filtered_adata_2.n_obs < sample_anndata.n_obs
    assert filtered_adata_2.n_vars < sample_anndata.n_vars


def test_chained_filtering(sample_anndata):
    """Test the effect of chained filtering operations."""
    source = AnnDataSource(adata=sample_anndata)

    # Select T cells
    t_cells = source.get("obs", cell_type="T")
    t_cell_count = len(t_cells)

    # Further filter to sample1
    t_cells_sample1 = source.get("obs", sample_id="sample1")
    assert len(t_cells_sample1) < t_cell_count  # Should be fewer rows

    # Should now have both filters applied
    pca_data = source.get("obsm_X_pca")

    # Verify through direct SQL to check
    verification = source.execute("""
        SELECT COUNT(*) as count FROM obsm_X_pca x
        JOIN obs o ON x.obs_id = o.obs_id
        WHERE o.cell_type = 'T' AND o.sample_id = 'sample1'
    """)

    assert len(pca_data) == verification["count"].iloc[0]


def test_get_with_sql_transforms(sample_anndata):
    """Test the get method with SQL transforms."""
    from lumen.transforms import SQLFilter

    source = AnnDataSource(adata=sample_anndata)

    # Define a SQL transform to add a WHERE clause
    sql_filter = SQLFilter(conditions=[("cell_type", "B")])

    # Get data with the transform
    filtered = source.get("obs", sql_transforms=[sql_filter])
    assert all(filtered["cell_type"] == "B")

    # Combine with direct filtering
    combined = source.get("obs", sample_id="sample1", sql_transforms=[sql_filter])
    assert all(combined["cell_type"] == "B")
    assert all(combined["sample_id"] == "sample1")


def test_create_sql_expr_source_with_modified_adata(fixed_sample_anndata):
    """Test creating a new source with modified AnnData preserves modifications."""
    source = AnnDataSource(adata=fixed_sample_anndata)

    new_source = source.create_sql_expr_source({"obs_b": "SELECT * FROM obs WHERE cell_type = 'B'"})

    # The old source should retain the original adata
    assert "obs_b" not in source.get_tables()

    # new source should have been subset
    obs_df = source.get("obs")
    new_obs_df = new_source.get("obs")
    assert not obs_df.equals(new_obs_df)

    # Create a new source with the modified adata
    # Modify the adata object by adding a new column
    modified_adata = fixed_sample_anndata.copy()
    modified_adata.obs["new_column"] = ["value1", "value2", "value3", "value4"]
    new_adata_source = new_source.create_sql_expr_source(new_source.tables, adata=modified_adata)

    # The new source should have the new column
    obs_df = new_adata_source.get("obs")
    assert "new_column" in obs_df.columns
    assert list(obs_df["new_column"]) == ["value1", "value3"]

    # The obs_b should still be available in the new source
    obs_b_df = new_adata_source.get("obs_b")
    assert not obs_b_df.empty


def test_create_sql_expr_source_updates_component_registry(fixed_sample_anndata):
    """Test that component registry is updated when creating source with modified adata."""
    source = AnnDataSource(adata=fixed_sample_anndata)

    # Get initial state
    initial_obs_columns = set(source._component_registry["obs"]["obj_ref"].columns)

    # Modify adata
    modified_adata = fixed_sample_anndata.copy()
    modified_adata.obs["test_col"] = "test_value"

    # Create new source
    new_source = source.create_sql_expr_source({}, adata=modified_adata)

    # Check that component registry was updated
    new_obs_columns = set(new_source._component_registry["obs"]["obj_ref"].columns)
    assert "test_col" in new_obs_columns
    assert new_obs_columns == initial_obs_columns | {"test_col"}

    # Verify the SQL table also has the new column
    schema = new_source.execute('PRAGMA table_info("obs")')
    column_names = schema["name"].tolist()
    assert "test_col" in column_names


def test_create_sql_expr_source(fixed_sample_anndata):
    """Test creating a new source with SQL expressions."""
    source = AnnDataSource(adata=fixed_sample_anndata)

    # Create a new source with a SQL expression for specific sample
    new_source = source.create_sql_expr_source({"new_table": "SELECT * FROM obs WHERE sample_name = '1262C'"})

    # Verify the new table exists in the new source
    assert "new_table" in new_source.get_tables()

    # Get data from the new table
    new_table_data = new_source.get("new_table")

    # Verify the contents match the expected filtered data
    assert len(new_table_data) == 1
    assert new_table_data.iloc[0]["sample_name"] == "1262C"
    assert new_table_data.iloc[0]["cell_type"] == "T"

    # Test with multiple tables
    multi_source = source.create_sql_expr_source(
        {"b_cells": "SELECT * FROM obs WHERE cell_type = 'B'", "count_by_type": "SELECT cell_type, COUNT(*) as count FROM obs GROUP BY cell_type"}
    )

    # Verify both tables exist
    assert "b_cells" in multi_source.get_tables()
    assert "count_by_type" in multi_source.get_tables()

    # Check contents of the tables
    b_cells = multi_source.get("b_cells")
    assert len(b_cells) == 2  # There are 2 B cells in the fixed sample
    assert all(b_cells["cell_type"] == "B")

    count_by_type = multi_source.get("count_by_type")
    assert len(count_by_type) == 3  # There are 3 cell types (B, T, NK)

    # Check that correct counts are present
    b_count = count_by_type[count_by_type["cell_type"] == "B"]["count"].iloc[0]
    assert b_count == 2

    t_count = count_by_type[count_by_type["cell_type"] == "T"]["count"].iloc[0]
    assert t_count == 1

    nk_count = count_by_type[count_by_type["cell_type"] == "NK"]["count"].iloc[0]
    assert nk_count == 1

    # Test that the tables are actually materialized
    all_tables = [item[0] for item in multi_source._connection.execute("SHOW TABLES").fetchall()]
    assert "b_cells" in all_tables
    assert "count_by_type" in all_tables


class TestOperation(AnnDataOperation):
    """A simple test operation that adds a new column."""

    some_value = param.Integer(default=42, doc="A test parameter for the operation.")

    def __call__(self, adata):
        adata.obs[f"test_{self.some_value}"] = ["test_value"] * len(adata.obs)
        return adata


def test_operation_persisted(fixed_sample_anndata):
    """Test that operations are persisted in the new source."""

    source = AnnDataSource(adata=fixed_sample_anndata, operations=[TestOperation.instance(), TestOperation.instance(some_value=100)])
    assert "test_42" in source.get("obs", return_type="anndata").obs.columns
    assert "test_100" in source.get("obs", return_type="anndata").obs.columns

    new_source = source.create_sql_expr_source({"test_table": "SELECT * FROM obs WHERE cell_type = 'B'"})
    assert "test_42" in new_source.get("obs", return_type="anndata").obs.columns
    assert "test_100" in new_source.get("obs", return_type="anndata").obs.columns

    assert "operations" in new_source.to_spec()
    assert AnnDataSource.from_spec(new_source.to_spec()).get("obs", return_type="anndata").obs["test_42"].tolist()[0] == "test_value"
