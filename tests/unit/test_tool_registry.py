import pandas as pd
import pytest

from core.tool_registry import (
    DATETIME_TOOLS,
    PROFILING_TOOLS,
    REPL_TOOLS,
    STATISTICAL_TOOLS,
    TOOL_REGISTRY,
    VIZ_TOOLS,
    detect_outliers,
    get_cardinality,
    get_categorical_summary,
    get_constant_columns,
    get_correlation_matrix,
    get_crosstab,
    get_datetime_features,
    get_distribution_stats,
    get_duplicate_report,
    get_feature_importance_proxy,
    get_high_cardinality_columns,
    get_kurtosis_report,
    get_null_report,
    get_sample_rows,
    get_schema,
    get_skewness_report,
    get_value_counts,
    inject_dataframe,
    make_boxplot_spec,
    make_correlation_heatmap_spec,
    make_histogram_spec,
    make_missing_value_heatmap_spec,
    make_pairwise_scatter_spec,
    make_scatter_spec,
    make_timeseries_spec,
    run_normality_test,
    run_python_repl,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age":    [25, 32, 47, 51, 23, 38, 61, 29, 44, 35],
        "income": [40000, 58000, 92000, 105000, 36000, 74000, 120000, 52000, 88000, 67000],
        "score":  [72, 85, 91, 78, 65, 88, 95, 70, 82, 79],
        "region": ["N", "S", "E", "W", "N", "E", "S", "W", "N", "E"],
    })


@pytest.fixture
def df_with_nulls(sample_df: pd.DataFrame) -> pd.DataFrame:
    df = sample_df.copy()
    df.loc[0, "age"] = None
    df.loc[2, "income"] = None
    return df


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def test_get_schema_keys(sample_df: pd.DataFrame) -> None:
    result = get_schema(sample_df)
    assert set(result.keys()) == {"columns", "dtypes", "shape", "numeric_cols", "categorical_cols"}
    assert "region" not in result["numeric_cols"]
    assert "region" in result["categorical_cols"]
    assert result["shape"] == [10, 4]


def test_get_null_report_empty_on_clean_df(sample_df: pd.DataFrame) -> None:
    result = get_null_report(sample_df)
    assert result == {"nulls": {}}


def test_get_null_report_reports_nulls(df_with_nulls: pd.DataFrame) -> None:
    result = get_null_report(df_with_nulls)
    assert "age" in result["nulls"]
    assert "income" in result["nulls"]
    assert result["nulls"]["age"]["count"] == 1
    assert result["nulls"]["income"]["count"] == 1


def test_get_distribution_stats_all_keys(sample_df: pd.DataFrame) -> None:
    result = get_distribution_stats(sample_df, "income")
    expected = {"mean", "median", "std", "min", "max", "skew", "kurtosis", "q25", "q50", "q75"}
    assert expected.issubset(set(result.keys()))
    assert isinstance(result["mean"], float)


def test_get_distribution_stats_raises_on_categorical(sample_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="not numeric"):
        get_distribution_stats(sample_df, "region")


def test_get_cardinality_region(sample_df: pd.DataFrame) -> None:
    result = get_cardinality(sample_df, "region")
    assert result["unique_count"] == 4
    assert isinstance(result["top_5"], list)
    assert all("value" in item and "count" in item for item in result["top_5"])


def test_get_sample_rows_returns_list_of_dicts(sample_df: pd.DataFrame) -> None:
    result = get_sample_rows(sample_df, n=3)
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(row, dict) for row in result)
    # All values must be native Python types (JSON-serialisable)
    import json
    json.dumps(result)  # raises if not serialisable


# ---------------------------------------------------------------------------
# Statistical
# ---------------------------------------------------------------------------

def test_get_correlation_matrix_nested_dict(sample_df: pd.DataFrame) -> None:
    result = get_correlation_matrix(sample_df)
    assert isinstance(result, dict)
    assert "age" in result
    assert "income" in result["age"]
    # Categorical column must not appear
    assert "region" not in result
    # Self-correlation is 1.0
    assert result["age"]["age"] == pytest.approx(1.0)


def test_detect_outliers_required_keys(sample_df: pd.DataFrame) -> None:
    result = detect_outliers(sample_df, "income")
    required = {"q1", "q3", "iqr", "lower_bound", "upper_bound", "outlier_count", "outlier_pct", "examples"}
    assert required.issubset(set(result.keys()))
    assert isinstance(result["outlier_count"], int)
    assert isinstance(result["examples"], list)


def test_run_normality_test_keys(sample_df: pd.DataFrame) -> None:
    result = run_normality_test(sample_df, "score")
    assert "stat" in result
    assert "p_value" in result
    assert "is_normal" in result
    assert isinstance(result["is_normal"], bool)
    assert 0.0 <= result["p_value"] <= 1.0


def test_run_normality_test_raises_on_categorical(sample_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="not numeric"):
        run_normality_test(sample_df, "region")


def test_get_feature_importance_proxy_sorted_descending(sample_df: pd.DataFrame) -> None:
    result = get_feature_importance_proxy(sample_df, "income")
    ranking = result["ranking"]
    assert isinstance(ranking, list)
    features = [r["feature"] for r in ranking]
    # Target must not appear in its own ranking
    assert "income" not in features
    # Sorted descending
    importances = [r["importance"] for r in ranking]
    assert importances == sorted(importances, reverse=True)


def test_get_skewness_report_bools(sample_df: pd.DataFrame) -> None:
    result = get_skewness_report(sample_df)
    assert "age" in result and "income" in result
    for info in result.values():
        assert isinstance(info["highly_skewed"], bool)
        assert isinstance(info["skewness"], float)


def test_get_kurtosis_report_bools(sample_df: pd.DataFrame) -> None:
    result = get_kurtosis_report(sample_df)
    assert "age" in result
    for info in result.values():
        assert isinstance(info["heavy_tailed"], bool)
        assert isinstance(info["kurtosis"], float)


# ---------------------------------------------------------------------------
# Categorical
# ---------------------------------------------------------------------------

def test_get_value_counts_sorted_descending(sample_df: pd.DataFrame) -> None:
    result = get_value_counts(sample_df, "region")
    counts = [v["count"] for v in result["values"]]
    assert counts == sorted(counts, reverse=True)
    assert all("value" in v and "count" in v and "pct" in v for v in result["values"])


def test_get_categorical_summary_region_present(sample_df: pd.DataFrame) -> None:
    result = get_categorical_summary(sample_df)
    assert "region" in result
    assert "unique_count" in result["region"]
    assert "mode" in result["region"]
    assert "top_frequency" in result["region"]
    assert result["region"]["unique_count"] == 4


def test_get_crosstab_nested_dict(sample_df: pd.DataFrame) -> None:
    df_cat = sample_df.copy()
    df_cat["grade"] = ["A", "B", "A", "B", "C", "A", "B", "C", "A", "B"]
    result = get_crosstab(df_cat, "region", "grade")
    assert isinstance(result, dict)
    for val in result.values():
        assert isinstance(val, dict)
        for count in val.values():
            assert isinstance(count, int)


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------

def test_get_duplicate_report_clean_df(sample_df: pd.DataFrame) -> None:
    result = get_duplicate_report(sample_df)
    assert result["duplicate_count"] == 0
    assert result["duplicate_pct"] == 0.0
    assert result["sample_duplicates"] == []


def test_get_constant_columns_detects_constant(sample_df: pd.DataFrame) -> None:
    df_const = sample_df.copy()
    df_const["constant"] = 42
    result = get_constant_columns(df_const)
    assert "constant" in result["constant_columns"]
    assert "age" not in result["constant_columns"]


def test_get_high_cardinality_empty_with_high_threshold(sample_df: pd.DataFrame) -> None:
    # region has 4 unique values — threshold=50 means none qualify
    result = get_high_cardinality_columns(sample_df, threshold=50)
    assert result["high_cardinality"] == []


def test_get_high_cardinality_detects_with_low_threshold(sample_df: pd.DataFrame) -> None:
    # region has 4 unique values > threshold=3
    result = get_high_cardinality_columns(sample_df, threshold=3)
    cols = [item["column"] for item in result["high_cardinality"]]
    assert "region" in cols


# ---------------------------------------------------------------------------
# Viz — all specs must have "data" and "layout" keys
# ---------------------------------------------------------------------------

def _assert_plotly_spec(spec: dict) -> None:
    assert "data" in spec, f"missing 'data' key in spec"
    assert "layout" in spec, f"missing 'layout' key in spec"


def test_make_histogram_spec(sample_df: pd.DataFrame) -> None:
    _assert_plotly_spec(make_histogram_spec(sample_df, "age"))


def test_make_scatter_spec(sample_df: pd.DataFrame) -> None:
    _assert_plotly_spec(make_scatter_spec(sample_df, "age", "income"))


def test_make_correlation_heatmap_spec(sample_df: pd.DataFrame) -> None:
    _assert_plotly_spec(make_correlation_heatmap_spec(sample_df))


def test_make_boxplot_spec(sample_df: pd.DataFrame) -> None:
    _assert_plotly_spec(make_boxplot_spec(sample_df, "income"))


def test_make_missing_value_heatmap_spec(sample_df: pd.DataFrame) -> None:
    _assert_plotly_spec(make_missing_value_heatmap_spec(sample_df))


def test_make_pairwise_scatter_spec(sample_df: pd.DataFrame) -> None:
    _assert_plotly_spec(make_pairwise_scatter_spec(sample_df))


def test_make_timeseries_spec_with_datetime_column() -> None:
    df_ts = pd.DataFrame({
        "date":  pd.date_range("2024-01-01", periods=10, freq="D"),
        "value": range(10),
    })
    _assert_plotly_spec(make_timeseries_spec(df_ts, "date", "value"))


# ---------------------------------------------------------------------------
# PythonREPL
# ---------------------------------------------------------------------------

def test_run_python_repl_success_json() -> None:
    code = "import json\nprint(json.dumps({'answer': 42}))"
    result = run_python_repl(code, "compute answer")
    assert result["success"] is True
    assert result["result"] == {"answer": 42}
    assert result["error"] is None


def test_run_python_repl_plain_string_gives_raw() -> None:
    code = "print('hello world')"
    result = run_python_repl(code, "print string")
    assert result["success"] is True
    assert "raw" in result["result"]


def test_run_python_repl_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import core.tool_registry as tr
    from unittest.mock import MagicMock

    # PythonREPLTool is a frozen Pydantic model; replace the module-level
    # _repl with a plain mock so run_python_repl picks it up via its global ref.
    mock_repl = MagicMock()
    mock_repl.run.side_effect = RuntimeError("Execution failed")
    monkeypatch.setattr(tr, "_repl", mock_repl)

    result = run_python_repl("bad code", "test failure")
    assert result["success"] is False
    assert result["error"] is not None
    assert "Execution failed" in result["error"]


def test_inject_dataframe_no_error(sample_df: pd.DataFrame) -> None:
    inject_dataframe(sample_df)  # must not raise


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_tool_registry_has_27_keys() -> None:
    assert len(TOOL_REGISTRY) == 27


def test_profiling_tools_contains_repl() -> None:
    names = [t.name for t in PROFILING_TOOLS]
    assert "run_python_repl" in names


def test_statistical_tools_contains_repl() -> None:
    names = [t.name for t in STATISTICAL_TOOLS]
    assert "run_python_repl" in names


def test_viz_tools_contains_repl() -> None:
    names = [t.name for t in VIZ_TOOLS]
    assert "run_python_repl" in names


def test_repl_tools_has_exactly_one() -> None:
    assert len(REPL_TOOLS) == 1
    assert REPL_TOOLS[0].name == "run_python_repl"
