import param
import pathlib

from anndata import AnnData
from lumen.sources.duckdb import DuckDBSource, cached
from lumen.transforms import SQLFilter, SQLColumns, SQLLimit


class AnnDataSource(DuckDBSource):
    """
    AnnDataSource provides a Lumen DuckDB wrapper for AnnData datasets.

    It works by mirroring the `obs` and `var` tables into DuckDB and
    tracking selections along the obs and var dimensions.

    The `.get` method extends the default behavior of a DuckDB source
    by making it possible to return the AnnData object instead of the
    requested table. By setting `return_type="anndata"` you can request
    the AnnData object with the current selections applied.
    """

    adata = param.ClassSelector(class_=(AnnData, str, pathlib.Path), doc="""
        An AnnData instance or path to a .h5ad file to load.""")

    ephemeral = param.Boolean(default=True, doc="""
        Always ephemeral (in-memory) by virtue of AnnSQL.""")

    filter_in_sql = param.Boolean(default=True, doc="""
        Whether to apply filters in SQL or in-memory.""")

    source_type = "anndata"

    def __init__(self, **params: Any):
        adata = params.get("adata")
        _tables = params.pop("_tables", None)
        if '_adata' in params:
            self._adata = params.pop('_adata')
        else:
            if isinstance(adata, (pathlib.Path, str)) and os.path.exists(adata):
                self._adata = ad.read_h5ad(adata)
            elif isinstance(adata, AnnData):
                self._adata = adata
            else:
                raise ValueError("`adata` must be an AnnData or path to an .h5ad file.")
            obs = self._adata.obs.copy()
            obs['obs_id'] = obs.index.values
            var = self._adata.var.copy()
            var['var_id'] = var.index.values
            params['mirrors'] = {
                'obs': obs,
                'var': var
            }
        super().__init__(**params)
        if _tables is not None:
            self._tables = _tables
        else:
            self._tables = None
            with self.param.update(tables=None):
                self._tables = self.get_tables()
        self._obs_ids = params.get('_obs_ids', None)
        self._var_ids = params.get('_var_ids', None)

    def create_sql_expr_source(
        self, tables: dict[str, str], materialize: bool = True, **kwargs
    ):
        params = dict(self.param.values(), **kwargs)
        params.pop('tables')
        params['_adata'] = self._adata
        params['_tables'] = self._tables
        source = super().create_sql_expr_source(tables.copy(), materialize, **params)
        source.tables.update({table: self.get_sql_expr(table) for table in self._tables})
        obs_ids = self._obs_ids
        var_ids = self._var_ids
        for table in tables:
            df = source.get(table)
            if "obs_id" in df.columns:
                if obs_ids is None:
                    obs_ids = df["obs_id"]
                else:
                    obs_ids = obs_ids[obs_ids.isin(df["obs_id"])]
            if "var_id" in df.columns:
                if var_ids is None:
                    var_ids = df["var_id"]
                else:
                    var_ids = var_ids[var_ids.isin(df["var_id"])]
        source._obs_ids = obs_ids
        source._var_ids = var_ids
        return source

    def get_tables(self):
        if self.tables is None and self._tables:
            return self._tables.copy()
        return super().get_tables()

    def _has(self, sql_expr, column='obs_id'):
        try:
            test_expr = sql_expr
            for t in (SQLColumns(columns=["obs_id"]), SQLLimit(limit=1)):
                test_expr = t.apply(test_expr)
            self.execute(test_expr)
        except Exception:
            return False
        else:
             return True

    @cached
    def get(self, table, **query):
        query.pop('__dask', None)
        return_type = query.pop('return_type', 'pandas')
        sql_expr = self.get_sql_expr(table)
        sql_transforms = query.pop('sql_transforms', [])
        conditions = list(query.items())
        if self._obs_ids is not None and self._has(sql_expr, 'obs_id'):
            # ALERT: This is potentially terrible since obs_ids can be huge
            #        But since the SQL table
            conditions.append(("obs_id", list(self._obs_ids)))
        if self._var_ids is not None and self._has(sql_expr, 'var_id'):
            # ALERT: This is potentially terrible since obs_ids can be huge
            #        But since the SQL table
            conditions.append(("var_id", list(self._var_ids)))
        if self.filter_in_sql:
            sql_transforms = [SQLFilter(conditions=conditions)] + sql_transforms
        for st in sql_transforms:
            sql_expr = st.apply(sql_expr)
        df = self.execute(sql_expr)
        if return_type == 'adata':
            adata = self._adata
            obs_slice = df["obs_id"] if "obs_id" in df.columns else slice(None)
            var_slice = df["var_id"] if "var_id" in df.columns else slice(None)
            return adata[obs_slice, var_slice]
        return df
