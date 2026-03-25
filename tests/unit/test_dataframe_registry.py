"""Tests for core/dataframe_registry.py."""
import pandas as pd
import pytest

from core.dataframe_registry import clear, get, register, remove


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})


def test_register_and_get(sample_df):
    register("sess-1", sample_df)
    result = get("sess-1")
    assert result is not None
    assert result.shape == sample_df.shape


def test_get_missing_returns_none():
    assert get("nonexistent") is None


def test_remove(sample_df):
    register("sess-2", sample_df)
    remove("sess-2")
    assert get("sess-2") is None


def test_remove_missing_is_no_op():
    remove("never-registered")  # must not raise


def test_clear(sample_df):
    register("sess-a", sample_df)
    register("sess-b", sample_df)
    clear()
    assert get("sess-a") is None
    assert get("sess-b") is None


def test_register_overwrites_existing(sample_df):
    df2 = pd.DataFrame({"z": [9, 8, 7]})
    register("sess-3", sample_df)
    register("sess-3", df2)
    result = get("sess-3")
    assert list(result.columns) == ["z"]


def test_multiple_sessions_independent(sample_df):
    df2 = pd.DataFrame({"a": [10, 20]})
    register("alpha", sample_df)
    register("beta", df2)
    assert get("alpha").shape == sample_df.shape
    assert get("beta").shape == df2.shape


def test_get_returns_same_object(sample_df):
    register("sess-4", sample_df)
    assert get("sess-4") is sample_df
