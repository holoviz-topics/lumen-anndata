"""Support AnnData datasets as a Lumen DuckDB source."""

from __future__ import annotations

import tempfile

from copy import deepcopy
from pathlib import Path
from typing import (
    Any, Literal, Union, cast,
)

import anndata as ad
import numpy as np
import pandas as pd
import panel as pn
import param

from anndata import AnnData
from lumen.config import config
from lumen.serializers import Serializer
from lumen.sources.base import cached
from lumen.sources.duckdb import DuckDBSource
from lumen.transforms import SQLFilter
from lumen.util import resolve_module_reference
from sqlglot import parse_one
from sqlglot.expressions import Table

ComponentInfo = dict[str, Union[Any, str, bool, None, pd.DataFrame, np.ndarray]]
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
      by applying current selections and query filters directly to the AnnData object.
    """

    adata = param.ClassSelector(
        class_=(AnnData, str, Path),
        doc="""
        AnnData object or path to a .h5ad file. This parameter is used only to initialize the source.
        To retrieve the up-to-date data, use the `get` method, which returns a DataFrame or
        filtered AnnData object depending on the `return_type` argument.
        """,
    )

    filter_in_sql = param.Boolean(default=True, doc="Whether to apply filters in SQL or in-memory.")

    operations = param.HookList(default=[], doc="""
        Operations to apply to the AnnData object
        ONLY when getting data with return_type='anndata'.""")

    uploaded_filename = param.String(
        default=None, doc="""
        If provided, will persist the uploaded AnnData file to this filename after the session ends,
        saved under the `.lumen_anndata_cache` directory,  or in `/tmp` if there is a PermissionError.
        """
    )

    source_type = "anndata"

    _opened = {}  # Track files: {filename: (adata_object, is_temporary)}

    def __init__(self, **params: Any):
        """Initialize AnnDataSource from an AnnData object or file path."""
        connection = params.pop('_connection', None)
        adata = params.get("adata")
        if adata is None:
            raise ValueError("Parameter 'adata' must be provided as an AnnData object or path to a .h5ad file.")

        # Initialize internal state from params if provided (for Lumen's state management)
        self._component_registry = params.pop('_component_registry', {})
        self._materialized_tables = params.pop('_materialized_tables', [])
        self._obs_ids_selected = params.pop('_obs_ids_selected', None)
        self._var_ids_selected = params.pop('_var_ids_selected', None)
        self._lumen_filename = None
        self._prepare_adata(adata)

        initial_mirrors = {}
        if self._adata_store:
            # Build registry only if not already provided (e.g., from create_sql_expr_source)
            if not self._component_registry:
                self._component_registry = self._build_component_registry_map()

            # Prepare obs and var tables using utility method
            initial_mirrors = self._prepare_obs_var_tables()

        params["mirrors"] = initial_mirrors
        if connection:
            params['_connection'] = connection
        super().__init__(**params)

        if self._adata_store and self.connection and initial_mirrors:
            self._register_tables(initial_mirrors)

        if self.tables is None:
            self.tables = {}
        self.tables.update({
            table: table for table in self._component_registry.keys()
        })
        if not self.uploaded_filename:
            pn.state.on_session_destroyed(self._cleanup_temp_files)

    def _prepare_obs_var_tables(self, adata: AnnData | None = None) -> dict[str, pd.DataFrame]:
        """Prepare obs and var tables with ID columns for SQL registration.

        Parameters
        ----------
        adata : AnnData, optional
            AnnData object to use. If None, uses self._adata_store.

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary with 'obs' and 'var' keys containing prepared DataFrames.
        """
        target_adata = adata or self._adata_store
        if not target_adata:
            return {}

        # Prepare obs table
        obs_df = target_adata.obs.copy()
        obs_df["obs_id"] = obs_df.index.astype(str).values

        # Prepare var table
        var_df = target_adata.var.copy()
        var_df["var_id"] = var_df.index.astype(str).values

        return {"obs": obs_df, "var": var_df}

    def _register_tables(self, tables: dict[str, pd.DataFrame]) -> None:
        """Register tables with the DuckDB connection and update materialized tables list.

        Parameters
        ----------
        tables : dict[str, pd.DataFrame]
            Dictionary mapping table names to DataFrames to register.
        """
        for table_name, df in tables.items():
            self.connection.register(table_name, df)
            if table_name not in self._materialized_tables:
                self._materialized_tables.append(table_name)

    def _prepare_adata(self, adata):
        """Prepare AnnData object from file path or AnnData instance."""
        if isinstance(adata, (str, Path)):
            adata_path = str(adata)
            adata_obj = None  # Will be loaded from file
            is_temp = False
        elif isinstance(adata, AnnData):
            adata_obj = adata
            adata_path = adata.filename or self._create_temp_file(adata)
            is_temp = adata.filename is None
        else:
            raise ValueError("Invalid 'adata' parameter: must be AnnData instance or path to .h5ad file.")

        # Used to access the *RAW* data saved on file; DO NOT use `get` method to retrieve the processed data!
        if adata_path in self._opened:
            self._adata_store = self._opened[adata_path][0]
        else:
            if isinstance(adata_obj, AnnData):
                self._adata_store = adata_obj
            else:
                self._adata_store = ad.read_h5ad(adata_path)
            self._opened[adata_path] = (self._adata_store, is_temp)
        self._lumen_filename = adata_path

    def _create_temp_file(self, adata: AnnData) -> str:
        """Create a temporary file for AnnData if no filename is set."""
        if self._lumen_filename:
            return self._lumen_filename

        try:
            cache_dir = Path(".lumen_anndata_cache")
            cache_dir.mkdir(exist_ok=True)
        except PermissionError:
            cache_dir = None

        with tempfile.NamedTemporaryFile(
            dir=cache_dir, suffix=".h5ad", delete=False, mode="wb"
        ) as temp_file:
            if self.uploaded_filename:
                filename = str(Path(temp_file.name).parent / self.uploaded_filename)
            else:
                filename = temp_file.name
            if not Path(filename).exists():
                adata.write_h5ad(filename)
                self.param.warning(
                    "AnnDataSource was created from an in-memory AnnData object. "
                    f"Saved to a temporary file {filename} for serialization. "
                    "Consider using backed='r' to avoid this."
                )
        return filename

    @classmethod
    def _cleanup_temp_files(cls, session_context):
        """Clean up all temporary files when session is destroyed."""
        # Create a list of items to avoid modifying dictionary during iteration
        items_to_process = list(cls._opened.items())
        for filename, (_, is_temp) in items_to_process:
            if not is_temp:
                cls._opened.pop(filename, None)
                continue
            Path(filename).unlink(missing_ok=True)
            cls._opened.pop(filename, None)

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

        df = self._convert_component_to_sql_df(table_name)
        if df is not None:
            try:
                self._register_tables({table_name: df})
            except Exception as e:
                # Create empty table
                self.connection.execute(f"CREATE TABLE {table_name} AS SELECT * FROM (SELECT 1 AS dummy) WHERE 0")
                self.param.warning(f"Failed to register table '{table_name}' with DuckDB: {e}")
                # Still mark as materialized even if it failed
                if table_name not in self._materialized_tables:
                    self._materialized_tables.append(table_name)
        else:
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

    def _refine_ids_from_df(self, df: pd.DataFrame, obs_ids: Any = None, var_ids: Any = None) -> tuple[Any, Any]:
        """Refine obs and var IDs based on what's present in a dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to check for obs_id and var_id columns
        obs_ids : array-like, optional
            Current obs IDs selection
        var_ids : array-like, optional
            Current var IDs selection

        Returns
        -------
        tuple[Any, Any]
            Refined (obs_ids, var_ids)
        """
        # Refine obs_ids selection based on what's present
        if "obs_id" in df.columns and len(df) > 0:
            current_obs_ids = df["obs_id"].unique()
            if obs_ids is None:
                obs_ids = pd.Series(current_obs_ids)
            else:
                # Keep only obs_ids that exist in the new table
                obs_ids_series = pd.Series(obs_ids)
                obs_ids = obs_ids_series[obs_ids_series.isin(current_obs_ids)]

        # Refine var_ids selection based on what's present
        if "var_id" in df.columns and len(df) > 0:
            current_var_ids = df["var_id"].unique()
            if var_ids is None:
                var_ids = pd.Series(current_var_ids)
            else:
                # Keep only var_ids that exist in the new table
                var_ids_series = pd.Series(var_ids)
                var_ids = var_ids_series[var_ids_series.isin(current_var_ids)]

        return obs_ids, var_ids

    def _set_ids_from_series_or_array(self, obs_ids: Any = None, var_ids: Any = None) -> None:
        """Set selection state from pandas Series, arrays, or None.

        Parameters
        ----------
        obs_ids : pd.Series, array-like, or None
            Observation IDs to set
        var_ids : pd.Series, array-like, or None
            Variable IDs to set
        """
        # Update obs_ids selection state
        if obs_ids is not None:
            if isinstance(obs_ids, pd.Series):
                self._obs_ids_selected = obs_ids.values if len(obs_ids) > 0 else None
            else:
                self._obs_ids_selected = obs_ids if len(obs_ids) > 0 else None
        else:
            self._obs_ids_selected = None

        # Update var_ids selection state
        if var_ids is not None:
            if isinstance(var_ids, pd.Series):
                self._var_ids_selected = var_ids.values if len(var_ids) > 0 else None
            else:
                self._var_ids_selected = var_ids if len(var_ids) > 0 else None
        else:
            self._var_ids_selected = None

    def _apply_operations(self, adata: AnnData) -> AnnData:
        """Apply all operations to the AnnData object and return the modified object."""
        if not self.operations:
            return adata

        required_tables = []
        for operation in self.operations:
            required_tables.extend(operation.requires)

        # Materialize required tables that aren't already materialized
        for table in required_tables:
            if table and table in self._component_registry:
                self._ensure_table_materialized(table)

        for operation in self.operations:
            adata = operation(adata)
        return adata

    def _get_as_anndata(self, query: dict[str, Any], table: str | None = None) -> AnnData:
        """Return a filtered AnnData object based on current selections and query.

        Parameters
        ----------
        query : dict
            Query parameters for filtering
        table : str, optional
            Table name with potential SQL expression to execute for getting IDs

        Returns
        -------
        AnnData
            Filtered AnnData object
        """
        obs_ids = self._obs_ids_selected
        var_ids = self._var_ids_selected

        # If selections are None and we have a table with SQL, execute it to get IDs
        if (obs_ids is None or var_ids is None) and table and table in self.tables:
            sql_expr = self.get_sql_expr(table)
            # Only execute if it's not just a simple table reference
            if sql_expr != self.sql_expr.format(table=f'"{table}"'):
                try:
                    df = self.execute(sql_expr)
                    obs_ids, var_ids = self._refine_ids_from_df(df, obs_ids, var_ids)
                except Exception as e:
                    self.param.warning(f"Could not extract IDs from table '{table}': {e}")

        obs_slice_labels = self._get_adata_slice_labels(self._adata_store.obs_names, obs_ids)
        var_slice_labels = self._get_adata_slice_labels(self._adata_store.var_names, var_ids)

        final_obs_labels, final_var_labels = self._prepare_anndata_slice_from_query(obs_slice_labels, var_slice_labels, query)
        adata = self._adata_store[final_obs_labels, final_var_labels]

        if self.operations:
            adata = self._apply_operations(adata)

        return adata

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
        try:
            return self.execute(final_sql_expr)
        except Exception as e:
            self.param.warning(f"SQL execution failed: {e}")
            return pd.DataFrame()

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

    def _serialize_tables(self) -> dict[str, Any]:
        """Serialize the tables for storage or transmission."""
        tables = {}
        for t in self.get_tables(materialized_only=True):
            tdf = self.get(t)
            serializer = Serializer._get_type(config.serializer)()
            tables[t] = serializer.serialize(tdf)
        return tables

    @cached
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

        if table == "anndata" or return_type == "anndata":
            return self._get_as_anndata(query, table)

        df_result = self._get_as_dataframe(table, query, sql_transforms)
        if table == "obs" and "obs_id" in df_result.columns:
            self._obs_ids_selected = df_result["obs_id"].unique()
        elif table == "var" and "var_id" in df_result.columns:
            self._var_ids_selected = df_result["var_id"].unique()
        return df_result

    def get_tables(self, materialized_only: bool = False) -> list[str]:
        """Get list of available tables."""
        all_tables = set({table for table in self.tables if not self._is_table_excluded(table)})
        if materialized_only:
            all_tables -= set(self._component_registry.keys()) - set(self._materialized_tables)
        else:
            all_tables |= set(self._component_registry)
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

    def create_sql_expr_source(
        self, tables: dict[str, str], materialize: bool = True, adata: AnnData | None = None, **kwargs
    ):
        """
        Creates a new SQL Source given a set of table names and
        corresponding SQL expressions, preserving and refining selection state.

        Arguments
        ---------
        tables: dict[str, str]
            Mapping from table name to SQL expression.
        materialize: bool
            Whether to materialize new tables
        adata: AnnData | None
            AnnData object to use for the new source, if any.
        kwargs: any
            Additional keyword arguments.

        Returns
        -------
        source: AnnDataSource
        """
        # Prepare parameters for the new source
        params = dict(self.param.values(), **kwargs)
        params.pop('name', None)  # Remove name to avoid conflicts
        params.pop('tables', None)  # Remove tables to avoid conflicts

        # Pass internal state to the new source
        params['_component_registry'] = self._component_registry
        # Only pass materialized tables that are not component registry tables
        # Component registry tables will be re-materialized on demand
        params['_materialized_tables'] = [table for table in self._materialized_tables
                                         if table not in self._component_registry]
        params['_obs_ids_selected'] = self._obs_ids_selected
        params['_var_ids_selected'] = self._var_ids_selected

        # Reuse connection unless it has changed OR we have a new AnnData object
        if 'uri' not in kwargs and 'initializers' not in kwargs and adata is None:
            params['_connection'] = self._connection

        sql_expr_tables = {
            table: sql_expr for table, sql_expr in tables.items()
            if table not in self._component_registry
        }

        # Create the new source using parent's method
        source = super().create_sql_expr_source(sql_expr_tables.copy(), materialize, **params)

        # Update the new source's tables with SQL expressions from all registered components
        source.tables.update({
            table: self.get_sql_expr(table)
            for table in self._component_registry.keys()
        })

        # Ensure component registry is properly set for the new source
        source._component_registry = self._component_registry

        # Ensure the new source has access to the AnnData store
        source._adata_store = adata or self._adata_store
        if adata is not None:
            source._component_registry = source._build_component_registry_map()
            # Re-register obs and var tables with the new AnnData's data using utility method
            new_tables = source._prepare_obs_var_tables(adata)
            source._register_tables(new_tables)
            is_temp = self._opened[source._lumen_filename][1]
            self._opened[source._lumen_filename] = (adata, is_temp)

        # Refine selections based on what's actually present in the new tables
        obs_ids = self._obs_ids_selected
        var_ids = self._var_ids_selected

        for table_name in tables:
            try:
                # Get the data from the new source without updating its selections
                # We're just checking what's present, not making a selection
                sql_expr = source.get_sql_expr(table_name)
                df = source.execute(sql_expr)

                # Use helper method to refine IDs
                obs_ids, var_ids = self._refine_ids_from_df(df, obs_ids, var_ids)

            except Exception as e:
                # If we can't get the table, skip refining selections for it
                self.param.warning(f"Could not refine selections for table '{table_name}': {e}")
                continue

        # Update the new source's selection state using helper method
        source._set_ids_from_series_or_array(obs_ids, var_ids)

        return source

    def to_spec(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        filename = self._lumen_filename
        spec = super().to_spec(context)
        spec["adata"] = filename

        # Handle operations serialization
        operations = spec.pop("operations", None)
        if not operations:
            return spec

        spec["operations"] = []
        for operation in operations:
            op_spec = {"type": f"{operation.__module__}.{type(operation).__name__}"}
            for k, v in operation.param.values().items():
                # Get the default value from the operation's class parameter
                param_obj = getattr(type(operation).param, k, None)
                if param_obj is None:
                    continue
                default = param_obj.default
                try:
                    is_equal = default is v
                    if not is_equal:
                        is_equal = default == v
                except Exception:
                    is_equal = False
                if k == 'name' or is_equal:
                    continue
                else:
                    op_spec[k] = v
            spec['operations'].append(op_spec)
        return spec

    @classmethod
    def from_spec(cls, spec: dict[str, Any] | str) -> "AnnDataSource":
        """Create AnnDataSource from specification.

        Parameters
        ----------
        spec : dict or str
            Source specification

        Returns
        -------
        AnnDataSource
            Instantiated source
        """
        if isinstance(spec, str):
            # If spec is a string, assume it's a file path
            return cls(adata=spec)

        spec = deepcopy(spec)

        # Handle operations deserialization
        operation_specs = spec.pop("operations", [])
        if operation_specs:
            operations = []
            for op in operation_specs:
                if isinstance(op, dict):
                    # Need to instantiate from spec
                    op_spec = deepcopy(op)
                    op_type = op_spec.pop('type')
                    op_class = resolve_module_reference(op_type)
                    if hasattr(op_class, "instance"):
                        operations.append(op_class.instance(**op_spec))
                    else:
                        operations.append(op_class(**op_spec))
                else:
                    # Already instantiated
                    operations.append(op)
            spec["operations"] = operations

        # Use parent class from_spec for everything else
        return super().from_spec(spec)
