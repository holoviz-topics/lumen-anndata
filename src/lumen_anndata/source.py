"""Support AnnData datasets as a Lumen DuckDB source."""

from __future__ import annotations

import pathlib

from typing import (
    Any, Literal, Union, cast,
)

import anndata as ad
import numpy as np
import pandas as pd
import param
import scipy.sparse as sp

from anndata import AnnData
from lumen.sources.duckdb import DuckDBSource
from lumen.transforms import SQLFilter
from sqlglot import parse_one
from sqlglot.expressions import Table

ComponentInfo = dict[str, Union[Any, str, bool, None, pd.DataFrame, np.ndarray, sp.spmatrix]]
ComponentRegistry = dict[str, ComponentInfo]


class AnnDataSource(DuckDBSource):
    """AnnDataSource provides a Lumen DuckDB wrapper for AnnData datasets.

    Core principles:
    - `obs` and `var` tables are materialized immediately for metadata querying and filtering.
    - All other AnnData components are registered but lazily materialized into SQL
      tables only when directly queried for a pandas DataFrame.
    - ID-based filtering (`_obs_ids_selected`, `_var_ids_selected`) tracks selections.
    - Selections are updated *only* by direct queries on `obs` or `var` tables.
    - `return_type='anndata'` efficiently returns filtered AnnData objects (copies)
      by applying current selections and query filters directly to the AnnData object,
      avoiding unnecessary SQL materialization of large data matrices.
    - Use `reset_selection()` to clear current selection state.
    """

    adata = param.ClassSelector(
        class_=(AnnData, str, pathlib.Path),
        doc="An AnnData instance or path to a .h5ad file to load.",
    )

    dense_matrix_warning_threshold = param.Integer(
        default=1_000_000,
        doc="""Threshold (number of elements) above which to warn when materializing
             dense matrices to SQL. Set to 0 to disable warnings.""",
    )

    ephemeral = param.Boolean(default=True, doc="Always ephemeral (in-memory) by virtue of AnnDataSource.")

    filter_in_sql = param.Boolean(default=True, doc="Whether to apply filters in SQL or in-memory.")

    source_type = "anndata"

    def __init__(self, **params: Any):
        """Initialize AnnDataSource from an AnnData object or file path."""
        adata = params.get("adata")
        if adata is None:
            raise ValueError("Parameter 'adata' must be provided as an AnnData object or path to a .h5ad file.")

        # Initialize internal state from params if provided (for Lumen's state management)
        self._component_registry = {}
        self._materialized_tables = []
        self._obs_ids_selected = None
        self._var_ids_selected = None
        if isinstance(adata, (str, pathlib.Path)):
            self._adata_store = ad.read_h5ad(adata)
        elif isinstance(adata, AnnData):
            self._adata_store = adata.copy()
        else:
            raise ValueError("Invalid 'adata' parameter: must be AnnData instance or path to .h5ad file.")

        initial_mirrors = {}
        if self._adata_store:
            if not self._component_registry:  # Build registry if not loaded from state
                self._component_registry = self._build_component_registry_map()

            # Prepare obs table
            obs_df = self._adata_store.obs.copy()
            obs_df["obs_id"] = obs_df.index.astype(str).values
            initial_mirrors["obs"] = obs_df

            # Prepare var table
            var_df = self._adata_store.var.copy()
            var_df["var_id"] = var_df.index.astype(str).values
            initial_mirrors["var"] = var_df

        params["mirrors"] = initial_mirrors
        super().__init__(**params)

        if self._adata_store and self.connection and initial_mirrors:
            for table_name, df in initial_mirrors.items():
                self.connection.register(table_name, df)
                if table_name not in self._materialized_tables:
                    self._materialized_tables.append(table_name)

    @staticmethod
    def _get_adata_slice_labels(
        original_adata_index: pd.Index,
        selected_ids: pd.Series | np.ndarray | list[str] | None,
    ) -> Union[slice, list[str]]:
        """Convert selection IDs to a format suitable for AnnData slicing (sorted list of present string IDs)."""
        if selected_ids is None:
            return slice(None)

        if not isinstance(original_adata_index, pd.Index):
            original_adata_index = pd.Index(original_adata_index)

        original_str_index = original_adata_index.astype(str)

        if isinstance(selected_ids, (pd.Series, np.ndarray)):
            unique_selected_ids = pd.Index(selected_ids).unique().astype(str)
        elif isinstance(selected_ids, list):
            unique_selected_ids = pd.Index(list(set(selected_ids))).astype(str)

        present_ids = original_str_index.intersection(unique_selected_ids)
        return sorted(present_ids.to_list())

    def _build_component_registry_map(self) -> ComponentRegistry:
        """Create registry of all AnnData components that can be mirrored to SQL tables."""
        if not self._adata_store:
            return {}

        registry: ComponentRegistry = {}
        adata = self._adata_store

        registry["obs"] = {"obj_ref": adata.obs, "type": "obs", "adata_key": None}
        registry["var"] = {"obj_ref": adata.var, "type": "var", "adata_key": None}

        if adata.X is not None:
            registry["X"] = {
                "obj_ref": adata.X,
                "type": "matrix",
                "adata_key": None,
                "is_sparse": sp.issparse(adata.X),
                "row_dim": "obs",
                "col_dim": "var",
            }

        for key, layer in adata.layers.items():
            registry[f"layer_{key}"] = {
                "obj_ref": layer,
                "type": "matrix",
                "adata_key": key,
                "is_sparse": sp.issparse(layer),
                "row_dim": "obs",
                "col_dim": "var",
            }
        for key, mat in adata.obsp.items():
            registry[f"obsp_{key}"] = {
                "obj_ref": mat,
                "type": "matrix",
                "adata_key": key,
                "is_sparse": sp.issparse(mat),
                "row_dim": "obs",
                "col_dim": "obs",
            }
        for key, mat in adata.varp.items():
            registry[f"varp_{key}"] = {
                "obj_ref": mat,
                "type": "matrix",
                "adata_key": key,
                "is_sparse": sp.issparse(mat),
                "row_dim": "var",
                "col_dim": "var",
            }
        for key, arr in adata.obsm.items():
            registry[f"obsm_{key}"] = {
                "obj_ref": arr,
                "type": "multidim",
                "adata_key": key,
                "dim": "obs",
            }
        for key, arr in adata.varm.items():
            registry[f"varm_{key}"] = {
                "obj_ref": arr,
                "type": "multidim",
                "adata_key": key,
                "dim": "var",
            }
        if adata.uns:  # Only add uns_keys if uns is not empty
            registry["uns_keys"] = {"obj_ref": list(adata.uns.keys()), "type": "uns_keys", "adata_key": None}
            for key, item in adata.uns.items():
                if isinstance(item, (pd.DataFrame, np.ndarray, dict, list, tuple, str, int, float, bool)):  # Common serializable types
                    registry[f"uns_{key}"] = {"obj_ref": item, "type": "uns", "adata_key": key}
        return registry

    def _convert_component_to_sql_df(self, table_name: str) -> pd.DataFrame | None:
        """Convert an AnnData component to a DataFrame suitable for SQL querying."""
        if not self._adata_store:
            return None
        if table_name not in self._component_registry:
            raise ValueError(f"Component '{table_name}' not found in AnnData registry.")

        comp_info = self._component_registry[table_name]
        obj_data = comp_info["obj_ref"]
        obj_type = cast(str, comp_info["type"])

        if obj_type == "obs":
            df = cast(pd.DataFrame, obj_data).copy()
            df["obs_id"] = df.index.astype(str).values
            return df
        if obj_type == "var":
            df = cast(pd.DataFrame, obj_data).copy()
            df["var_id"] = df.index.astype(str).values
            return df

        if obj_type == "matrix":
            matrix = obj_data
            row_dim_type = cast(str, comp_info["row_dim"])  # 'obs' or 'var'
            col_dim_type = cast(str, comp_info["col_dim"])  # 'obs' or 'var'

            r_idx = (self._adata_store.obs_names if row_dim_type == "obs" else self._adata_store.var_names).astype(str)
            c_idx = (self._adata_store.var_names if col_dim_type == "var" else self._adata_store.obs_names).astype(str)

            r_name = (
                "obs_id"
                if row_dim_type == "obs" and col_dim_type == "var"
                else (
                    "var_id"
                    if row_dim_type == "var" and col_dim_type == "obs"
                    else (f"{row_dim_type}_id_1" if row_dim_type == col_dim_type else f"{row_dim_type}_id")
                )
            )
            c_name = (
                "var_id"
                if col_dim_type == "var" and row_dim_type == "obs"
                else (
                    "obs_id"
                    if col_dim_type == "obs" and row_dim_type == "var"
                    else (f"{col_dim_type}_id_2" if row_dim_type == col_dim_type else f"{col_dim_type}_id")
                )
            )

            if sp.issparse(matrix):
                coo = matrix.tocoo()
                return pd.DataFrame({r_name: r_idx[coo.row], c_name: c_idx[coo.col], "value": coo.data})
            else:  # Dense matrix
                matrix_np = cast(np.ndarray, matrix)
                row_indices, col_indices = np.meshgrid(np.arange(matrix_np.shape[0]), np.arange(matrix_np.shape[1]), indexing="ij")
                return pd.DataFrame(
                    {
                        r_name: r_idx[row_indices.ravel()],
                        c_name: c_idx[col_indices.ravel()],
                        "value": matrix_np.ravel(),
                    }
                )

        if obj_type == "multidim":
            array_like = obj_data
            adata_key = cast(str, comp_info["adata_key"])
            dim_type = cast(str, comp_info["dim"])  # 'obs' or 'var'
            id_col_name = f"{dim_type}_id"
            id_labels = (self._adata_store.obs_names if dim_type == "obs" else self._adata_store.var_names).astype(str)

            if isinstance(array_like, pd.DataFrame):
                df = array_like.copy()
            elif isinstance(array_like, np.ndarray):
                if array_like.ndim == 1:
                    df = pd.DataFrame({f"{adata_key}_0": array_like})
                elif array_like.ndim == 2:
                    df = pd.DataFrame(array_like, columns=[f"{adata_key}_{i}" for i in range(array_like.shape[1])])
                else:
                    return None  # Cannot easily represent >2D array as single SQL table
            else:
                return None

            df[id_col_name] = id_labels[: len(df)]
            return df.reset_index(drop=True)

        if obj_type == "uns_keys":
            return pd.DataFrame({"uns_key": cast(list[str], obj_data)})
        if obj_type == "uns":
            item = obj_data
            if isinstance(item, pd.DataFrame):
                return item.reset_index()
            if isinstance(item, np.ndarray):
                if item.ndim == 1:
                    return pd.DataFrame({"value": item})
                if item.ndim == 2:
                    return pd.DataFrame(item, columns=[f"col_{i}" for i in range(item.shape[1])])
            if isinstance(item, dict):
                return pd.DataFrame([item])
            if isinstance(item, (list, tuple)) and all(isinstance(i, (str, int, float, bool)) for i in item):
                return pd.DataFrame({"value": item})
            if isinstance(item, (str, int, float, bool)):
                return pd.DataFrame({"value": [item]})

        return None

    def _ensure_table_materialized(self, table_name: str):
        """Materialize an AnnData component into a DuckDB table if not already done."""
        if table_name in self._materialized_tables:
            return
        if table_name not in self._component_registry:
            if table_name not in self.get_tables():
                raise ValueError(f"Table '{table_name}' is not a known AnnData component or predefined table.")
            return

        comp_info = self._component_registry[table_name]
        if comp_info.get("type") == "matrix":
            matrix = comp_info["obj_ref"]
            size = matrix.shape[0] * matrix.shape[1]
            is_sparse = comp_info.get("is_sparse", False)
            if not is_sparse and self.dense_matrix_warning_threshold > 0 and size > self.dense_matrix_warning_threshold:
                self.param.warning(
                    f"Materializing dense matrix '{table_name}' ({matrix.shape[0]}x{matrix.shape[1]} = {size:,} elements) to SQL. "
                    f"This is MEMORY INTENSIVE. Consider using ID-based filtering with `return_type='anndata'`."
                )

        df = self._convert_component_to_sql_df(table_name)
        if df is not None:
            try:
                self.connection.register(table_name, df)
            except Exception as e:
                # Create empty table
                self.connection.execute(f"CREATE TABLE {table_name} AS SELECT * FROM (SELECT 1 AS dummy) WHERE 0")
                self.param.warning(f"Failed to register table '{table_name}' with DuckDB: {e}")
            self._materialized_tables.append(table_name)
        else:
            # Do not raise error here, as some 'uns' items might not be convertible.
            # Let subsequent SQL query fail if the table is truly needed and couldn't be made.
            self.param.warning(f"Component '{table_name}' conversion to DataFrame failed; cannot materialize for SQL.")

    def _has_column_in_sql_table(self, table_name: str, column_name: str) -> bool:
        """Check if a materialized SQL table has a specific column."""
        if table_name == "obs" and column_name == "obs_id":
            return True
        if table_name == "var" and column_name == "var_id":
            return True

        if table_name not in self._materialized_tables:
            # If table is not materialized, we can't check its columns via SQL describe.
            # Try to infer from component registry for unmaterialized components.
            if table_name in self._component_registry:
                comp_info = self._component_registry[table_name]
                comp_type = comp_info.get("type")
                if comp_type == "obs" and column_name in self._adata_store.obs.columns:
                    return True
                if comp_type == "var" and column_name in self._adata_store.var.columns:
                    return True
            return False

        schema_df = self.execute(f'PRAGMA table_info("{table_name}")')
        return column_name in schema_df["name"].astype(str).values

    def _prepare_anndata_slice_from_query(
        self,
        initial_obs_slice_labels: Union[slice, list[str]],
        initial_var_slice_labels: Union[slice, list[str]],
        query_filters: dict[str, Any],
    ) -> tuple[list[str], list[str]]:
        """
        Refine AnnData observation and variable slices based on query_filters.
        Applies filters to adata.obs and adata.var.
        """
        effective_obs_names = self._adata_store.obs_names if isinstance(initial_obs_slice_labels, slice) else pd.Index(initial_obs_slice_labels)
        effective_var_names = self._adata_store.var_names if isinstance(initial_var_slice_labels, slice) else pd.Index(initial_var_slice_labels)

        final_obs_keep_mask = pd.Series(True, index=effective_obs_names)
        final_var_keep_mask = pd.Series(True, index=effective_var_names)

        if not effective_obs_names.empty:
            for key, value in query_filters.items():
                if key in self._adata_store.obs.columns:
                    obs_column_data = self._adata_store.obs.loc[effective_obs_names, key]
                    condition = obs_column_data.isin(value) if isinstance(value, (list, tuple)) else (obs_column_data == value)
                    final_obs_keep_mask &= condition.reindex(effective_obs_names, fill_value=False)

        if not effective_var_names.empty:
            for key, value in query_filters.items():
                if key in self._adata_store.var.columns:
                    var_column_data = self._adata_store.var.loc[effective_var_names, key]
                    condition = var_column_data.isin(value) if isinstance(value, (list, tuple)) else (var_column_data == value)
                    final_var_keep_mask &= condition.reindex(effective_var_names, fill_value=False)

        final_obs_labels = effective_obs_names[final_obs_keep_mask].tolist()
        final_var_labels = effective_var_names[final_var_keep_mask].tolist()

        return final_obs_labels, final_var_labels

    def _get_as_anndata(self, query: dict[str, Any]) -> AnnData:
        """Return a filtered AnnData object based on current selections and query."""
        obs_slice_labels = self._get_adata_slice_labels(self._adata_store.obs_names, self._obs_ids_selected)
        var_slice_labels = self._get_adata_slice_labels(self._adata_store.var_names, self._var_ids_selected)

        final_obs_labels, final_var_labels = self._prepare_anndata_slice_from_query(obs_slice_labels, var_slice_labels, query)
        return self._adata_store[final_obs_labels, final_var_labels].copy()

    def _get_as_dataframe(self, table: str, query: dict[str, Any], sql_transforms: list) -> pd.DataFrame:
        """Get table data as DataFrame, materializing if necessary."""
        is_materialized = table in self._materialized_tables
        is_registered = table in self._component_registry

        if is_registered and not is_materialized:
            self._ensure_table_materialized(table)

        if table not in self._materialized_tables and table not in self.get_tables():
            raise ValueError(f"Table '{table}' could not be prepared for SQL query.")

        conditions = self._build_sql_conditions(table, query)

        current_sql_expr = self.get_sql_expr(table)
        applied_transforms = sql_transforms
        if self.filter_in_sql and conditions:
            applied_transforms = [SQLFilter(conditions=conditions)] + sql_transforms

        final_sql_expr = current_sql_expr
        for transform in applied_transforms:
            final_sql_expr = transform.apply(final_sql_expr)

        return self.execute(final_sql_expr)

    def _build_sql_conditions(self, table: str, query: dict) -> list:
        """Build conditions for SQL filtering from selections and query."""
        conditions = []

        if self._obs_ids_selected is not None and self._has_column_in_sql_table(table, "obs_id"):
            obs_ids = list(pd.Series(self._obs_ids_selected).unique().astype(str))
            if obs_ids:
                conditions.append(("obs_id", obs_ids))

        if self._var_ids_selected is not None and self._has_column_in_sql_table(table, "var_id"):
            var_ids = list(pd.Series(self._var_ids_selected).unique().astype(str))
            if var_ids:
                conditions.append(("var_id", var_ids))

        for key, value in query.items():
            if self._has_column_in_sql_table(table, key) or table not in self._component_registry:
                conditions.append((key, value))

        return conditions

    # @cached  # TODO: figure out what to do with this alongside reset_selection
    def get(self, table: str, **query: Any) -> Union[pd.DataFrame, AnnData]:
        """Get data from AnnData as DataFrame or filtered AnnData object.

        Parameters
        ----------
        table : str
            Name of the table to query (e.g., 'obs', 'var', 'X', etc.).
        query : dict
            Additional query parameters to filter the data, e.g. {'obs_id': ['cell1', 'cell2']}.

        Returns
        -------
        Union[pd.DataFrame, AnnData]
            DataFrame or AnnData object containing the queried data.
        """
        query.pop("__dask", None)  # Remove dask-specific parameter
        return_type = cast(Literal["pandas", "anndata"], query.pop("return_type", "pandas"))
        sql_transforms = query.pop("sql_transforms", [])

        if return_type == "anndata":
            return self._get_as_anndata(query)

        df_result = self._get_as_dataframe(table, query, sql_transforms)
        if table == "obs" and "obs_id" in df_result.columns:
            self._obs_ids_selected = df_result["obs_id"].unique()
        elif table == "var" and "var_id" in df_result.columns:
            self._var_ids_selected = df_result["var_id"].unique()
        return df_result

    def get_tables(self, materialized_only: bool = False) -> list[str]:
        """Get list of available tables."""
        all_tables = set(super().get_tables())
        if materialized_only:
            all_tables |= set(self._materialized_tables)
        else:
            all_tables |= set(self._component_registry.keys())
        return sorted(all_tables)

    def execute(self, sql_query: str, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Execute SQL query, automatically materializing referenced AnnData tables if needed."""
        parsed_query = parse_one(sql_query)
        if parsed_query:  # Ensure parsing was successful
            tables_in_query = {table.name for table in parsed_query.find_all(Table)}
            for table_name in tables_in_query:
                if table_name in self._component_registry and table_name not in self._materialized_tables:
                    self._ensure_table_materialized(table_name)
        return super().execute(sql_query, *args, **kwargs)

    def reset_selection(self, dim: str | None = None) -> "AnnDataSource":
        """Reset selection tracking for specified dimension(s).

        Parameters
        ----------
        dim : str or None
            Dimension to reset: 'obs', 'var', or None (resets both).

        Returns
        -------
        AnnDataSource
            The instance itself, for method chaining.
        """
        if dim is None or dim.lower() == "obs":
            self._obs_ids_selected = None
        if dim is None or dim.lower() == "var":
            self._var_ids_selected = None
        return self
