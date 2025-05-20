"""
Tests for lumen_anndata.source.AnnDataSource
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from lumen_anndata.source import AnnDataSource


@pytest.fixture
def fixed_sample_anndata():
    X = np.array([[1, 0, 3], [0, 5, 0], [2, 0, 0], [0, 1, 1]], dtype=np.float32)
    obs_df = pd.DataFrame(
        {
            "cell_type": pd.Categorical(["B", "T", "B", "NK"]),
            "n_genes": [10, 20, 5, 15],
        },
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
    assert source._adata_store is not sample_anndata  # Should be a copy
    assert source._component_registry, "Component registry should not be empty"
    assert "obs" in source._materialized_tables
    assert "var" in source._materialized_tables
    assert len(source._materialized_tables) == 2  # Initially only obs and var

    # Check that obs and var tables are correctly prepared
    obs_df_sql = source.execute("SELECT * FROM obs ORDER BY obs_id LIMIT 2")
    pd.testing.assert_series_equal(obs_df_sql["obs_id"].astype(str).reset_index(drop=True), pd.Series(sample_anndata.obs_names[:2].astype(str)).reset_index(drop=True), check_names=False)
    assert "cell_type" in obs_df_sql.columns
    assert obs_df_sql["cell_type"].tolist() == sample_anndata.obs["cell_type"].iloc[:2].tolist()

    var_df_sql = source.execute("SELECT * FROM var ORDER BY var_id LIMIT 2")
    pd.testing.assert_series_equal(var_df_sql["var_id"].astype(str).reset_index(drop=True), pd.Series(sample_anndata.var_names[:2].astype(str)).reset_index(drop=True), check_names=False)
    assert "gene_type" in var_df_sql.columns

    # Test initialization parameters
    source_with_params = AnnDataSource(adata=sample_anndata, dense_matrix_warning_threshold=500, filter_in_sql=False)
    assert source_with_params.dense_matrix_warning_threshold == 500
    assert source_with_params.filter_in_sql is False
    assert source_with_params._obs_ids_selected is None  # Check initial selection state
    assert source_with_params._var_ids_selected is None


def test_get_tables(sample_anndata):
    """Test the get_tables method."""
    source = AnnDataSource(adata=sample_anndata)

    expected_tables = {
        "obs",
        "var",
        "X",
        "layer_normalized",
        "layer_binary",
        "obsm_X_pca",
        "obsm_X_umap",
        "varm_PCs",
        "obsp_distances",
        "varp_correlations",
        "uns_clustering_params",
        "uns_metadata",
        "uns_colors",
        "uns_keys",
    }
    all_tables = source.get_tables()
    assert set(all_tables) == expected_tables

    materialized_tables = source.get_tables(materialized_only=True)
    assert set(materialized_tables) == {"obs", "var"}

    # Materialize X and check again
    source.get("X", limit=1)  # Querying X should materialize it
    materialized_tables_after_get = source.get_tables(materialized_only=True)
    assert set(materialized_tables_after_get) == {"obs", "var", "X"}


def test_execute_basic_queries_fixed(fixed_sample_anndata):
    source = AnnDataSource(adata=fixed_sample_anndata)
    obs_result = source.execute("SELECT * FROM obs WHERE cell_type = 'B' ORDER BY obs_id")
    expected_obs_b = fixed_sample_anndata.obs[fixed_sample_anndata.obs["cell_type"] == "B"].copy()
    expected_obs_b["obs_id"] = expected_obs_b.index.astype(str)
    pd.testing.assert_frame_equal(
        obs_result.reset_index(drop=True), expected_obs_b.reset_index(drop=True).sort_values("obs_id").reset_index(drop=True), check_dtype=False, check_categorical=False
    )

    agg_result = source.execute("SELECT cell_type, SUM(n_genes) as total_genes FROM obs GROUP BY cell_type ORDER BY cell_type")
    expected_agg = fixed_sample_anndata.obs.groupby("cell_type", observed=False)["n_genes"].sum().reset_index(name="total_genes").sort_values("cell_type")
    pd.testing.assert_frame_equal(agg_result.reset_index(drop=True), expected_agg.reset_index(drop=True), check_dtype=False, check_categorical=False)


def test_execute_matrix_queries_fixed(fixed_sample_anndata):
    source = AnnDataSource(adata=fixed_sample_anndata)

    x_result = source.execute("SELECT * FROM X WHERE obs_id='cell_0' AND var_id='gene_A'")
    assert len(x_result) == 1
    x_value = fixed_sample_anndata["cell_0", "gene_A"].X
    if isinstance(x_value, np.ndarray):
        assert x_result["value"].iloc[0] == pytest.approx(x_value[0, 0])
    else:
        assert x_result["value"].iloc[0] == pytest.approx(x_value.toarray()[0, 0])

    layer_result = source.execute("SELECT * FROM layer_counts WHERE obs_id='cell_1' AND var_id='gene_B'")
    assert len(layer_result) == 1
    layer_value = fixed_sample_anndata["cell_1", "gene_B"].layers["counts"]
    if isinstance(layer_value, np.ndarray):
        assert layer_result["value"].iloc[0] == pytest.approx(layer_value[0, 0])
    else:
        assert layer_result["value"].iloc[0] == pytest.approx(layer_value.toarray()[0, 0])

    # Ensure table 'X' is now materialized
    assert "X" in source._materialized_tables
    assert "layer_counts" in source._materialized_tables


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

    # Get X after obs selection
    x_df_after_obs_filter = source.get("X")
    assert set(x_df_after_obs_filter["obs_id"].unique()) == set(expected_b_ids)
    assert len(x_df_after_obs_filter["var_id"].unique()) == fixed_sample_anndata.n_vars

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

    # Get AnnData for 'X' with obs filter, and var filter directly in query
    highly_var_gene_ids = fixed_sample_anndata.var_names[fixed_sample_anndata.var["highly_variable"]].tolist()
    filtered_adata = source.get("X", return_type="anndata", highly_variable=True)

    assert isinstance(filtered_adata, ad.AnnData)
    pd.testing.assert_index_equal(filtered_adata.obs_names.astype(str), pd.Index(b_cell_ids).astype(str))
    assert (filtered_adata.obs["cell_type"] == "B").all()
    pd.testing.assert_index_equal(filtered_adata.var_names.astype(str), pd.Index(highly_var_gene_ids).astype(str))
    assert filtered_adata.var["highly_variable"].all()

    if "cell_0" in filtered_adata.obs_names and "gene_A" in filtered_adata.var_names:
        assert filtered_adata["cell_0", "gene_A"].X.item() == pytest.approx(fixed_sample_anndata["cell_0", "gene_A"].X.item())

    no_obs_adata = source.get("X", return_type="anndata", cell_type="NonExistentType")
    assert no_obs_adata.n_obs == 0
    assert no_obs_adata.n_vars == fixed_sample_anndata.n_vars

    source.reset_selection()
    full_adata = source.get("X", return_type="anndata")
    assert full_adata.shape == fixed_sample_anndata.shape
    pd.testing.assert_frame_equal(full_adata.obs, fixed_sample_anndata.obs)
    pd.testing.assert_frame_equal(full_adata.var, fixed_sample_anndata.var)


def test_materialization_on_execute(sample_anndata):
    source = AnnDataSource(adata=sample_anndata)
    assert "X" not in source._materialized_tables
    source.execute("SELECT * FROM X LIMIT 1")
    assert "X" in source._materialized_tables
    assert "layer_normalized" not in source._materialized_tables
    source.execute("SELECT * FROM layer_normalized nl JOIN X x ON nl.obs_id = x.obs_id LIMIT 1")
    assert "layer_normalized" in source._materialized_tables


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


def test_execute_matrix_queries(sample_anndata):
    """Test executing queries on matrix components."""
    source = AnnDataSource(adata=sample_anndata)

    # Query the main expression matrix (X)
    x_result = source.execute("SELECT * FROM X LIMIT 5")
    assert "obs_id" in x_result.columns
    assert "var_id" in x_result.columns
    assert "value" in x_result.columns

    # Query a layer
    layer_result = source.execute("SELECT * FROM layer_normalized LIMIT 5")
    assert "obs_id" in layer_result.columns
    assert "var_id" in layer_result.columns
    assert "value" in layer_result.columns

    # Query a pairwise matrix (obsp)
    obsp_result = source.execute("SELECT * FROM obsp_distances LIMIT 5")
    assert "obs_id_1" in obsp_result.columns or "obs_id" in obsp_result.columns
    assert "value" in obsp_result.columns


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


def test_execute_uns_queries(sample_anndata):
    """Test executing queries on unstructured data."""
    source = AnnDataSource(adata=sample_anndata)

    # Query uns_keys
    uns_keys = source.execute("SELECT * FROM uns_keys")
    assert "uns_key" in uns_keys.columns
    assert "clustering_params" in uns_keys["uns_key"].values
    assert "metadata" in uns_keys["uns_key"].values

    # Query specific uns component
    colors = source.execute("SELECT * FROM uns_colors")
    assert "value" in colors.columns
    assert len(colors) == 3  # We had 3 colors


def test_execute_with_joins(sample_anndata):
    """Test executing queries with joins between tables."""
    source = AnnDataSource(adata=sample_anndata)

    # Join obs metadata with expression data
    join_result = source.execute("""
        SELECT o.cell_type, COUNT(*) as expr_count
        FROM X x
        JOIN obs o ON x.obs_id = o.obs_id
        WHERE x.value > 0
        GROUP BY o.cell_type
    """)

    assert "cell_type" in join_result.columns
    assert "expr_count" in join_result.columns

    # Join var metadata with expression data
    gene_expr = source.execute("""
        SELECT v.gene_type, AVG(x.value) as avg_expr
        FROM X x
        JOIN var v ON x.var_id = v.var_id
        GROUP BY v.gene_type
    """)

    assert "gene_type" in gene_expr.columns
    assert "avg_expr" in gene_expr.columns
    assert len(gene_expr) <= 3  # Should have at most 3 gene types


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

    # Get expression data with selection tracking from previous query
    # (This tests that the sample_id filtered obs_ids are used for X)
    expr_data = source.get("X")
    assert all(np.isin(expr_data["obs_id"], filtered["obs_id"]))

    # Verify internal state tracking
    assert source._obs_ids_selected is not None
    assert len(source._obs_ids_selected) == len(filtered)


def test_get_anndata(sample_anndata):
    """Test the get method returning AnnData."""
    source = AnnDataSource(adata=sample_anndata)

    # First filter the obs table to establish a selection
    source.get("obs", cell_type="T")

    # Get filtered AnnData
    filtered_adata = source.get("X", return_type="anndata")

    # Check if filtering was applied correctly
    assert isinstance(filtered_adata, ad.AnnData)
    assert np.all(filtered_adata.obs["cell_type"] == "T")
    assert filtered_adata.n_obs < sample_anndata.n_obs
    assert filtered_adata.n_vars == sample_anndata.n_vars  # Vars not filtered

    # Test filtering both obs and vars
    source.get("var", highly_variable=True)  # Establish var selection
    filtered_adata_2 = source.get("X", return_type="anndata")

    # Check if both filters were applied
    assert np.all(filtered_adata_2.obs["cell_type"] == "T")
    # All variables in the result should be highly variable
    assert filtered_adata_2.var["highly_variable"].all()
    assert filtered_adata_2.n_obs < sample_anndata.n_obs
    assert filtered_adata_2.n_vars < sample_anndata.n_vars

    # Reset selection and check
    source.reset_selection()
    unfiltered = source.get("X", return_type="anndata")
    assert unfiltered.n_obs == sample_anndata.n_obs
    assert unfiltered.n_vars == sample_anndata.n_vars


def test_reset_selection(sample_anndata):
    """Test the reset_selection method with direct ID filtering."""
    source = AnnDataSource(adata=sample_anndata)

    # 1. Test resetting obs selection
    # First get a subset of cell IDs
    cells = source.execute("SELECT obs_id FROM obs LIMIT 10")
    test_cell_ids = cells["obs_id"].tolist()

    # Apply direct ID filtering
    filtered_obs = source.get("obs", obs_id=test_cell_ids)
    assert len(filtered_obs) == len(test_cell_ids)

    # Verify selection is active by checking internal state
    assert source._obs_ids_selected is not None
    assert set(source._obs_ids_selected) == set(test_cell_ids)

    # Verify filtered X data only contains the selected obs_ids
    x_filtered = source.get("X")
    assert set(x_filtered["obs_id"].unique()) == set(test_cell_ids)

    # Reset obs selection only
    source.reset_selection(dim="obs")

    # Verify internal state is reset
    assert source._obs_ids_selected is None

    # Verify data is no longer filtered by obs
    x_after_reset = source.get("X")
    assert len(set(x_after_reset["obs_id"].unique())) == sample_anndata.n_obs

    # 2. Test resetting var selection
    # First get a subset of gene IDs
    genes = source.execute("SELECT var_id FROM var LIMIT 5")
    test_gene_ids = genes["var_id"].tolist()

    # Apply direct ID filtering
    filtered_var = source.get("var", var_id=test_gene_ids)
    assert len(filtered_var) == len(test_gene_ids)

    # Verify selection is active
    assert source._var_ids_selected is not None
    assert set(source._var_ids_selected) == set(test_gene_ids)

    # Verify filtered X data only contains the selected var_ids
    x_filtered = source.get("X")
    assert set(x_filtered["var_id"].unique()) == set(test_gene_ids)

    # Reset var selection
    source.reset_selection(dim="var")

    # Verify internal state is reset
    assert source._var_ids_selected is None

    # Verify data is no longer filtered by var
    x_after_reset = source.get("X")
    assert len(set(x_after_reset["var_id"].unique())) == sample_anndata.n_vars

    # 3. Test resetting both dimensions at once
    # Apply filtering on both dimensions
    source.get("obs", obs_id=test_cell_ids)
    source.get("var", var_id=test_gene_ids)

    # Verify both selections are active
    assert source._obs_ids_selected is not None
    assert source._var_ids_selected is not None

    # Verify filtered X contains only selected IDs from both dimensions
    x_filtered = source.get("X")
    assert set(x_filtered["obs_id"].unique()) == set(test_cell_ids)
    assert set(x_filtered["var_id"].unique()) == set(test_gene_ids)

    # Reset both dimensions
    source.reset_selection()

    # Verify both internal states are reset
    assert source._obs_ids_selected is None
    assert source._var_ids_selected is None

    # Verify data is completely unfiltered
    x_fully_reset = source.get("X")
    assert len(x_fully_reset["obs_id"].unique()) == sample_anndata.n_obs
    assert len(x_fully_reset["var_id"].unique()) == sample_anndata.n_vars

    # Verify that querying obs or var now returns all items
    all_obs = source.get("obs")
    assert len(all_obs) == sample_anndata.n_obs
    assert source._obs_ids_selected is not None  # get("obs") will re-select all if no filter
    assert len(source._obs_ids_selected) == sample_anndata.n_obs

    all_var = source.get("var")
    assert len(all_var) == sample_anndata.n_vars
    assert source._var_ids_selected is not None
    assert len(source._var_ids_selected) == sample_anndata.n_vars


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
    expr_data = source.get("X")

    # Verify through direct SQL to check
    verification = source.execute("""
        SELECT COUNT(*) as count FROM X x
        JOIN obs o ON x.obs_id = o.obs_id
        WHERE o.cell_type = 'T' AND o.sample_id = 'sample1'
    """)

    assert len(expr_data) == verification["count"].iloc[0]


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
