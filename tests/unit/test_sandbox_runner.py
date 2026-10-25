import json
import sys

import pandas as pd
import pytest

from dva.sandbox.runner import child_env, run_code, truncate
from dva.workspace import session_workspace

EMIT_ONE = (
    "emit([{'field': 'quantity', 'issue_type': 'other', 'severity': 'low', "
    "'detail': 'note', 'count': 3}], ['quantity:units'])"
)


@pytest.fixture
def ws(tmp_path):
    return session_workspace(tmp_path, "s1").ensure()


def test_emit_returns_findings_and_stdout(ws, settings):
    result = run_code(f"print('hello')\n{EMIT_ONE}", ws, {}, settings.sandbox)
    assert result.ok and result.error == ""
    assert result.stdout.strip() == "hello"
    assert result.emitted.findings[0].count == 3
    assert result.emitted.checks_completed == ["quantity:units"]


def test_code_reads_parquet_and_saves_artifacts(ws, settings):
    pd.DataFrame({"quantity": [1.0, 2.0, 3.0]}).to_parquet(ws.table_path("sales"))
    code = (
        "df = pd.read_parquet(DATA['sales'])\n"
        "df.to_csv(f'{ARTIFACTS_DIR}/copy.csv', index=False)\n"
        "print(int(df['quantity'].sum()))\nemit([], [])"
    )
    result = run_code(code, ws, {"sales": str(ws.table_path("sales"))}, settings.sandbox)
    assert result.ok and result.stdout.strip() == "6"
    assert result.artifacts == [str(ws.artifacts_dir / "copy.csv")]


def test_policy_violation_never_runs(ws, settings):
    code = "pd.DataFrame().to_csv(f'{ARTIFACTS_DIR}/ran.csv')\nimport os"
    result = run_code(code, ws, {}, settings.sandbox)
    assert not result.ok and "rejected by the sandbox policy" in result.error
    assert not (ws.artifacts_dir / "ran.csv").exists()
    assert not (ws.sandbox_dir / "analysis.py").exists()


def test_exception_returns_truncated_traceback_with_code_lines(ws, settings):
    result = run_code("x = {}\ny = x['missing']\n", ws, {}, settings.sandbox)
    assert not result.ok
    assert 'File "analysis.py", line 2' in result.error and "KeyError" in result.error
    assert "bootstrap" not in result.error


def test_timeout_stops_execution(ws, settings):
    fast = settings.sandbox.model_copy(update={"timeout_seconds": 1})
    result = run_code("while True:\n    pass\n", ws, {}, fast)
    assert not result.ok and result.timed_out
    assert result.error == "Timed out after 1 s" and result.duration_s < 10


def test_stdout_is_truncated(ws, settings):
    result = run_code(f"print('x' * 10000)\n{EMIT_ONE}", ws, {}, settings.sandbox)
    assert result.ok
    assert "truncated" in result.stdout
    assert len(result.stdout) < settings.sandbox.output_char_limit + 100


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("x = 1", "without calling emit"),
        (
            "emit([{'field': 'a', 'issue_type': 'bogus', 'severity': 'low', 'detail': 'd', "
            "'count': 1}], [])",
            "emit() payload is invalid",
        ),
    ],  # fmt: skip
)
def test_missing_or_invalid_emit_is_an_error(ws, settings, code, message):
    result = run_code(code, ws, {}, settings.sandbox)
    assert not result.ok and message in result.error


def test_stale_emit_from_an_earlier_run_is_not_reused(ws, settings):
    assert run_code(EMIT_ONE, ws, {}, settings.sandbox).ok
    assert not run_code("x = 1", ws, {}, settings.sandbox).ok


def test_child_environment_has_no_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
    env = child_env(tmp_path)
    assert set(env) == {"PATH", "HOME", "LANG", "PYTHONHASHSEED", "PYTHONDONTWRITEBYTECODE"}
    assert "sk-ant" not in json.dumps(env)


def test_memory_limit_status_is_reported(ws, settings):
    result = run_code(EMIT_ONE, ws, {}, settings.sandbox)
    if sys.platform.startswith("linux"):
        assert result.memory_limit == "applied"
    else:  # macOS refuses RLIMIT_AS; Windows has no resource module
        assert result.memory_limit.startswith("not")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS enforced on Linux")
def test_memory_limit_stops_large_allocations(ws, settings):
    result = run_code("x = bytearray(3 * 1024 ** 3)\nemit([], [])", ws, {}, settings.sandbox)
    assert not result.ok and "MemoryError" in result.error


def test_truncate_keeps_head_or_tail():
    assert truncate("abcdef", 10) == "abcdef"
    assert truncate("abcdef", 3).startswith("abc") and "truncated 3" in truncate("abcdef", 3)
    assert truncate("abcdef", 3, keep="tail").endswith("def")
