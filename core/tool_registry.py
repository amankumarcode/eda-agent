"""
All callable tools for the EDA agent system.

Every tool exists in two forms:
1. A plain Python function — used directly in tests and agent nodes.
2. A LangChain StructuredTool — bound to create_react_agent in each node.

Agent nodes import their tool subsets from PROFILING_TOOLS, STATISTICAL_TOOLS,
VIZ_TOOLS, DATETIME_TOOLS, or REPL_TOOLS. They never define tool functions
themselves.

Important: all outputs are plain dicts with native Python types. Numpy scalars
are cast via .item() so every return value is JSON-serialisable.
"""

import io
import json
from typing import Any, Callable, Optional

import pandas as pd
from scipy import stats as scipy_stats
import plotly.express as px
from langchain_core.tools import StructuredTool
from langchain_experimental.tools import PythonREPLTool

from core.schema import REPLOutput

# Single REPL instance shared across all tool calls in this process.
_repl = PythonREPLTool()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _native(val: Any) -> Any:
    """Convert numpy scalars → native Python; Timestamps → str; NaN → None."""
    if val is None:
        return None
    if isinstance(val, pd.Timestamp):
        return str(val)
    if hasattr(val, "item"):           # numpy scalar
        v = val.item()
        return None if (isinstance(v, float) and v != v) else v
    if isinstance(val, float) and val != val:   # plain float NaN
        return None
    return val


# ---------------------------------------------------------------------------
# Group 1 — Profiling
# ---------------------------------------------------------------------------

def get_schema(df: pd.DataFrame) -> dict:
    """Return schema info for the DataFrame: column names, dtypes as strings,
    shape as [rows, cols], numeric_cols list, and categorical_cols list."""
    return {
        "columns":          df.columns.tolist(),
        "dtypes":           {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape":            list(df.shape),
        "numeric_cols":     df.select_dtypes(include="number").columns.tolist(),
        "categorical_cols": df.select_dtypes(exclude="number").columns.tolist(),
    }


def get_null_report(df: pd.DataFrame) -> dict:
    """Return null counts and percentages for every column that has at least
    one null. Returns {"nulls": {}} when the DataFrame is fully populated."""
    total = len(df)
    if total == 0:
        return {"nulls": {}}
    nulls: dict[str, Any] = {}
    for col in df.columns:
        n = int(df[col].isnull().sum())
        if n > 0:
            nulls[col] = {"count": n, "pct": round(n / total * 100, 2)}
    return {"nulls": nulls}


def get_distribution_stats(df: pd.DataFrame, column: str) -> dict:
    """Return descriptive statistics for a numeric column: mean, median, std,
    min, max, skew, kurtosis, q25, q50, q75.
    Raises ValueError if column is not numeric."""
    if column not in df.select_dtypes(include="number").columns:
        raise ValueError(f"Column '{column}' is not numeric.")
    s = df[column].dropna()
    return {
        "mean":     float(s.mean()),
        "median":   float(s.median()),
        "std":      float(s.std()),
        "min":      float(s.min()),
        "max":      float(s.max()),
        "skew":     float(s.skew()),
        "kurtosis": float(s.kurtosis()),
        "q25":      float(s.quantile(0.25)),
        "q50":      float(s.quantile(0.50)),
        "q75":      float(s.quantile(0.75)),
    }


def get_cardinality(df: pd.DataFrame, column: str) -> dict:
    """Return unique value count and top-5 most frequent values (descending)
    for the given column."""
    vc = df[column].value_counts()
    top_5 = [{"value": _native(v), "count": int(c)} for v, c in vc.head(5).items()]
    return {"unique_count": int(df[column].nunique()), "top_5": top_5}


def get_sample_rows(df: pd.DataFrame, n: int = 5) -> list:
    """Return the first n rows as a list of dicts. Timestamps are converted to
    strings; numpy scalars to native Python types."""
    return [
        {col: _native(val) for col, val in row.items()}
        for _, row in df.head(n).iterrows()
    ]


# ---------------------------------------------------------------------------
# Group 2 — Statistical
# ---------------------------------------------------------------------------

def get_correlation_matrix(df: pd.DataFrame) -> dict:
    """Return the Pearson correlation matrix as a nested dict for all numeric
    columns, computed after dropping rows that contain any null."""
    corr = df.select_dtypes(include="number").dropna().corr()
    return {
        col: {other: float(corr.loc[col, other]) for other in corr.columns}
        for col in corr.index
    }


def detect_outliers(df: pd.DataFrame, column: str) -> dict:
    """Detect outliers in a numeric column using the IQR method.
    Returns q1, q3, iqr, lower_bound, upper_bound, outlier_count,
    outlier_pct (%), and up to 10 example outlier values."""
    s = df[column].dropna()
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = s[(s < lower) | (s > upper)]
    total = len(s)
    return {
        "q1":            q1,
        "q3":            q3,
        "iqr":           float(iqr),
        "lower_bound":   float(lower),
        "upper_bound":   float(upper),
        "outlier_count": int(len(outliers)),
        "outlier_pct":   float(round(len(outliers) / total * 100, 2)) if total else 0.0,
        "examples":      [float(v) for v in outliers.head(10).tolist()],
    }


def run_normality_test(df: pd.DataFrame, column: str) -> dict:
    """Run the Shapiro-Wilk normality test on a numeric column.
    Returns stat, p_value, and is_normal (True when p_value > 0.05).
    Raises ValueError if column is not numeric."""
    if column not in df.select_dtypes(include="number").columns:
        raise ValueError(f"Column '{column}' is not numeric.")
    s = df[column].dropna()
    stat, p_value = scipy_stats.shapiro(s)
    return {
        "stat":      float(stat),
        "p_value":   float(p_value),
        "is_normal": bool(p_value > 0.05),
    }


def get_feature_importance_proxy(df: pd.DataFrame, target_column: str) -> dict:
    """Rank all other numeric columns by their absolute Pearson correlation
    with target_column in descending order.
    Returns {"ranking": [{"feature": str, "importance": float}]}.
    Raises ValueError if target_column is not numeric."""
    numeric = df.select_dtypes(include="number").dropna()
    if target_column not in numeric.columns:
        raise ValueError(f"Column '{target_column}' is not numeric or not found.")
    corr = numeric.corr()[target_column].drop(target_column).abs()
    return {
        "ranking": [
            {"feature": col, "importance": float(val)}
            for col, val in corr.sort_values(ascending=False).items()
        ]
    }


def get_skewness_report(df: pd.DataFrame) -> dict:
    """Return skewness for every numeric column.
    highly_skewed is True when abs(skewness) > 1."""
    numeric = df.select_dtypes(include="number")
    result = {}
    for col in numeric.columns:
        s = float(numeric[col].dropna().skew())
        result[col] = {"skewness": s, "highly_skewed": bool(abs(s) > 1)}
    return result


def get_kurtosis_report(df: pd.DataFrame) -> dict:
    """Return excess kurtosis (Fisher's definition, normal = 0) for every
    numeric column. heavy_tailed is True when kurtosis > 3."""
    numeric = df.select_dtypes(include="number")
    result = {}
    for col in numeric.columns:
        k = float(numeric[col].dropna().kurtosis())
        result[col] = {"kurtosis": k, "heavy_tailed": bool(k > 3)}
    return result


# ---------------------------------------------------------------------------
# Group 3 — Categorical
# ---------------------------------------------------------------------------

def get_value_counts(df: pd.DataFrame, column: str) -> dict:
    """Return value counts for a column as a list of {value, count, pct} dicts
    sorted descending by count."""
    total = len(df)
    vc = df[column].value_counts()
    return {
        "values": [
            {
                "value": _native(v),
                "count": int(c),
                "pct":   float(round(int(c) / total * 100, 2)) if total else 0.0,
            }
            for v, c in vc.items()
        ]
    }


def get_categorical_summary(df: pd.DataFrame) -> dict:
    """Return {unique_count, mode, top_frequency} for every categorical column."""
    total = len(df)
    cat_df = df.select_dtypes(exclude="number")
    result = {}
    for col in cat_df.columns:
        vc = cat_df[col].value_counts()
        mode = str(vc.index[0]) if len(vc) > 0 else None
        top_freq = float(round(int(vc.iloc[0]) / total * 100, 2)) if (len(vc) > 0 and total) else 0.0
        result[col] = {
            "unique_count":  int(cat_df[col].nunique()),
            "mode":          mode,
            "top_frequency": top_freq,
        }
    return result


def get_crosstab(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """Return a frequency crosstab between two categorical columns as a nested
    dict: {col1_value: {col2_value: count}}."""
    ct = pd.crosstab(df[col1], df[col2])
    return {
        str(row): {str(col): int(ct.loc[row, col]) for col in ct.columns}
        for row in ct.index
    }


# ---------------------------------------------------------------------------
# Group 4 — Data quality
# ---------------------------------------------------------------------------

def get_duplicate_report(df: pd.DataFrame) -> dict:
    """Return duplicate row count, duplicate percentage, and up to 5 sample
    duplicate rows as a list of dicts."""
    total = len(df)
    dupes = df[df.duplicated()]
    count = int(len(dupes))
    samples = [
        {col: _native(val) for col, val in row.items()}
        for _, row in dupes.head(5).iterrows()
    ]
    return {
        "duplicate_count":   count,
        "duplicate_pct":     float(round(count / total * 100, 2)) if total else 0.0,
        "sample_duplicates": samples,
    }


def get_constant_columns(df: pd.DataFrame) -> dict:
    """Return a list of column names where every row has the same value
    (nunique == 1)."""
    return {
        "constant_columns": [col for col in df.columns if df[col].nunique() == 1]
    }


def get_high_cardinality_columns(df: pd.DataFrame, threshold: int = 50) -> dict:
    """Return categorical columns whose unique value count exceeds threshold.
    Each entry includes the column name and its unique count."""
    cat_df = df.select_dtypes(exclude="number")
    return {
        "high_cardinality": [
            {"column": col, "unique_count": int(cat_df[col].nunique())}
            for col in cat_df.columns
            if cat_df[col].nunique() > threshold
        ]
    }


# ---------------------------------------------------------------------------
# Group 5 — Viz  (all return fig.to_dict())
# ---------------------------------------------------------------------------

def make_histogram_spec(df: pd.DataFrame, column: str) -> dict:
    """Return a Plotly histogram figure spec (fig.to_dict()) for column."""
    fig = px.histogram(df, x=column, title=f"Distribution of {column}")
    return fig.to_dict()


def make_scatter_spec(df: pd.DataFrame, x: str, y: str) -> dict:
    """Return a Plotly scatter plot spec of x vs y with labelled axes."""
    fig = px.scatter(df, x=x, y=y, title=f"{y} vs {x}", labels={x: x, y: y})
    return fig.to_dict()


def make_correlation_heatmap_spec(df: pd.DataFrame) -> dict:
    """Return a Plotly heatmap spec of the Pearson correlation matrix using
    RdBu colorscale, clamped to [-1, 1]."""
    corr = df.select_dtypes(include="number").dropna().corr()
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu",
        title="Correlation Heatmap",
        zmin=-1,
        zmax=1,
    )
    return fig.to_dict()


def make_boxplot_spec(df: pd.DataFrame, column: str) -> dict:
    """Return a Plotly box plot spec for column."""
    fig = px.box(df, y=column, title=f"Box Plot of {column}")
    return fig.to_dict()


def make_missing_value_heatmap_spec(df: pd.DataFrame) -> dict:
    """Return a Plotly heatmap spec showing missing values:
    1 = missing, 0 = present, with labelled axes."""
    missing = df.isnull().astype(int)
    fig = px.imshow(
        missing,
        title="Missing Value Heatmap",
        labels={"x": "Column", "y": "Row Index"},
    )
    return fig.to_dict()


def make_pairwise_scatter_spec(df: pd.DataFrame) -> dict:
    """Return a Plotly scatter matrix (SPLOM) spec for all numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    fig = px.scatter_matrix(numeric_df, title="Pairwise Scatter Matrix")
    return fig.to_dict()


def make_timeseries_spec(df: pd.DataFrame, x: str, y: str) -> dict:
    """Return a Plotly line chart spec of y over x. x should be a datetime
    column for meaningful time-series display."""
    fig = px.line(df, x=x, y=y, title=f"{y} over Time")
    return fig.to_dict()


# ---------------------------------------------------------------------------
# Group 6 — Datetime
# ---------------------------------------------------------------------------

def get_datetime_features(df: pd.DataFrame, column: str) -> dict:
    """Extract year, month, and day_of_week lists from a datetime column and
    attempt to infer the series frequency via pd.infer_freq (None if not
    inferrable). Raises ValueError if column cannot be parsed as datetime."""
    try:
        dt = pd.to_datetime(df[column])
    except Exception as exc:
        raise ValueError(
            f"Column '{column}' cannot be parsed as datetime: {exc}"
        ) from exc
    freq: Optional[str] = pd.infer_freq(pd.DatetimeIndex(dt))
    return {
        "year":               dt.dt.year.tolist(),
        "month":              dt.dt.month.tolist(),
        "day_of_week":        dt.dt.day_of_week.tolist(),
        "detected_frequency": freq,
    }


# ---------------------------------------------------------------------------
# Group 7 — Grounded PythonREPL
# ---------------------------------------------------------------------------

# Module-level DataFrame reference set by inject_dataframe.
# VIZ_TOOLS bound wrappers read this instead of accepting df as a parameter,
# so create_react_agent can generate valid JSON schemas for them.
_current_df: Optional[pd.DataFrame] = None


def inject_dataframe(df: pd.DataFrame) -> None:
    """Store df for bound viz tool wrappers and prime the REPL namespace.
    Call this from the agent node before running the ReAct loop."""
    global _current_df
    _current_df = df
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    csv_str = buf.getvalue()
    _repl.run(
        f"import pandas as pd, io\n"
        f"df = pd.read_csv(io.StringIO('''{csv_str}'''))\n"
    )


# ---------------------------------------------------------------------------
# Group 5b — Bound viz wrappers (no df parameter → JSON-schema friendly)
# ---------------------------------------------------------------------------

def _bound_make_histogram_spec(column: str) -> dict:
    """Return a Plotly histogram spec for column using the injected DataFrame."""
    assert _current_df is not None, "Call inject_dataframe before using viz tools."
    return make_histogram_spec(_current_df, column)


def _bound_make_scatter_spec(x: str, y: str) -> dict:
    """Return a Plotly scatter plot spec of x vs y using the injected DataFrame."""
    assert _current_df is not None, "Call inject_dataframe before using viz tools."
    return make_scatter_spec(_current_df, x, y)


def _bound_make_correlation_heatmap_spec() -> dict:
    """Return a Plotly correlation heatmap spec using the injected DataFrame."""
    assert _current_df is not None, "Call inject_dataframe before using viz tools."
    return make_correlation_heatmap_spec(_current_df)


def _bound_make_boxplot_spec(column: str) -> dict:
    """Return a Plotly box plot spec for column using the injected DataFrame."""
    assert _current_df is not None, "Call inject_dataframe before using viz tools."
    return make_boxplot_spec(_current_df, column)


def _bound_make_missing_value_heatmap_spec() -> dict:
    """Return a Plotly missing value heatmap spec using the injected DataFrame."""
    assert _current_df is not None, "Call inject_dataframe before using viz tools."
    return make_missing_value_heatmap_spec(_current_df)


def _bound_make_pairwise_scatter_spec() -> dict:
    """Return a Plotly scatter matrix spec using the injected DataFrame."""
    assert _current_df is not None, "Call inject_dataframe before using viz tools."
    return make_pairwise_scatter_spec(_current_df)


def _bound_make_timeseries_spec(x: str, y: str) -> dict:
    """Return a Plotly line chart spec of y over x using the injected DataFrame."""
    assert _current_df is not None, "Call inject_dataframe before using viz tools."
    return make_timeseries_spec(_current_df, x, y)


def run_python_repl(code: str, description: str) -> dict:
    """Execute ad-hoc Python code in a persistent REPL and return structured output.

    Rules for the code you write:
    - Always end your code with: print(json.dumps(your_result))
    - Import json at the top of your code
    - The DataFrame is available as `df` after inject_dataframe() has been called
      by the node before the agent loop starts.
    - If you forget to print valid JSON the result will be {"raw": <your output>}.

    Returns a dict with keys: success (bool), result (dict), stdout_raw (str),
    error (str | None).
    """
    try:
        stdout_raw: str = _repl.run(code)
        try:
            parsed = json.loads(stdout_raw.strip())
            result: dict[str, Any] = parsed if isinstance(parsed, dict) else {"value": parsed}
        except (json.JSONDecodeError, ValueError):
            result = {"raw": stdout_raw}
        return REPLOutput(
            success=True,
            result=result,
            stdout_raw=stdout_raw,
            error=None,
        ).model_dump()
    except Exception as exc:
        return REPLOutput(
            success=False,
            result={},
            stdout_raw="",
            error=str(exc),
        ).model_dump()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Callable] = {
    "get_schema":                      get_schema,
    "get_null_report":                 get_null_report,
    "get_distribution_stats":          get_distribution_stats,
    "get_cardinality":                 get_cardinality,
    "get_sample_rows":                 get_sample_rows,
    "get_correlation_matrix":          get_correlation_matrix,
    "detect_outliers":                 detect_outliers,
    "run_normality_test":              run_normality_test,
    "get_feature_importance_proxy":    get_feature_importance_proxy,
    "get_skewness_report":             get_skewness_report,
    "get_kurtosis_report":             get_kurtosis_report,
    "get_value_counts":                get_value_counts,
    "get_categorical_summary":         get_categorical_summary,
    "get_crosstab":                    get_crosstab,
    "get_duplicate_report":            get_duplicate_report,
    "get_constant_columns":            get_constant_columns,
    "get_high_cardinality_columns":    get_high_cardinality_columns,
    "make_histogram_spec":             make_histogram_spec,
    "make_scatter_spec":               make_scatter_spec,
    "make_correlation_heatmap_spec":   make_correlation_heatmap_spec,
    "make_boxplot_spec":               make_boxplot_spec,
    "make_missing_value_heatmap_spec": make_missing_value_heatmap_spec,
    "make_pairwise_scatter_spec":      make_pairwise_scatter_spec,
    "make_timeseries_spec":            make_timeseries_spec,
    "get_datetime_features":           get_datetime_features,
    "run_python_repl":                 run_python_repl,
    "inject_dataframe":                inject_dataframe,
}

PROFILING_TOOLS: list[StructuredTool] = [
    StructuredTool.from_function(f) for f in [
        get_schema, get_null_report, get_distribution_stats,
        get_cardinality, get_sample_rows, get_categorical_summary,
        get_duplicate_report, get_constant_columns,
        get_high_cardinality_columns, run_python_repl,
    ]
]

STATISTICAL_TOOLS: list[StructuredTool] = [
    StructuredTool.from_function(f) for f in [
        get_correlation_matrix, detect_outliers, run_normality_test,
        get_feature_importance_proxy, get_skewness_report,
        get_kurtosis_report, get_value_counts, get_crosstab,
        run_python_repl,
    ]
]

VIZ_TOOLS: list[StructuredTool] = [
    StructuredTool.from_function(f) for f in [
        _bound_make_histogram_spec,
        _bound_make_scatter_spec,
        _bound_make_correlation_heatmap_spec,
        _bound_make_boxplot_spec,
        _bound_make_missing_value_heatmap_spec,
        _bound_make_pairwise_scatter_spec,
        _bound_make_timeseries_spec,
        run_python_repl,
    ]
]

DATETIME_TOOLS: list[StructuredTool] = [
    StructuredTool.from_function(get_datetime_features),
]

REPL_TOOLS: list[StructuredTool] = [
    StructuredTool.from_function(run_python_repl),
]
