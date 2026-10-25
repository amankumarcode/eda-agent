"""Run generated analysis code in a separate Python process.

The code is checked by the AST policy first and never runs if it fails. It runs with the
session folder as working directory, a timeout, a memory limit where the OS allows one,
and an environment with no secrets. Results come back only through the preloaded
``emit(findings, checks_completed)`` helper and truncated stdout.

This is a guardrail, not a security boundary.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, ValidationError

from dva.config import SandboxSettings, session_logger
from dva.sandbox.policy import check_code
from dva.state import IssueType, Severity
from dva.workspace import SessionWorkspace

CODE_FILENAME = "analysis.py"
BOOTSTRAP_FILENAME = "bootstrap.py"
CONFIG_FILENAME = "run_config.json"
STATUS_FILENAME = "status.json"

# Our own launcher; the AST policy applies to the generated code only. It applies the
# memory limit to its own process, preloads pd/np/DATA/ARTIFACTS_DIR/emit, then runs
# analysis.py so that tracebacks point at the generated code's own line numbers.
BOOTSTRAP = """\
import json, sys, traceback
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
status = {"memory_limit": "not supported on this platform"}
try:
    import resource
    limit = config["memory_limit_mb"] * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, resource.getrlimit(resource.RLIMIT_AS)[1]))
    status["memory_limit"] = "applied"
except Exception as exc:
    status["memory_limit"] = f"not applied: {type(exc).__name__}: {exc}"
Path(config["status_path"]).write_text(json.dumps(status))

import numpy as np
import pandas as pd


def emit(findings, checks_completed):
    payload = {"findings": findings, "checks_completed": checks_completed}
    Path(config["emit_path"]).write_text(json.dumps(payload, default=str))


namespace = {"pd": pd, "np": np, "DATA": config["data"], "ARTIFACTS_DIR": config["artifacts_dir"],
             "emit": emit, "__name__": "__main__"}
code_path = config["code_path"]
try:
    exec(compile(Path(code_path).read_text(), "analysis.py", "exec"), namespace)
except BaseException as exc:
    traceback.print_exception(type(exc), exc, exc.__traceback__.tb_next)
    sys.exit(1)
"""


class EmittedFinding(BaseModel):
    """One finding as the generated code reports it (source is set by the EDA agent)."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1)
    issue_type: IssueType
    severity: Severity
    detail: str = Field(min_length=1)
    count: NonNegativeInt
    evidence_ref: str | None = None


class EmitPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[EmittedFinding]
    checks_completed: list[str]


class SandboxResult(BaseModel):
    """Outcome of one run. ``error`` holds policy violations, a truncated traceback, a
    timeout message, or an emit problem; it is empty when ``ok``."""

    ok: bool
    emitted: EmitPayload | None = None
    stdout: str = ""
    error: str = ""
    timed_out: bool = False
    policy_violations: list[str] = []
    memory_limit: str = ""
    duration_s: float = 0.0
    artifacts: list[str] = []


def truncate(text: str, limit: int, *, keep: str = "head") -> str:
    """Cut text to ``limit`` characters, noting how much was dropped."""
    if len(text) <= limit:
        return text
    note = f"\n...[truncated {len(text) - limit} characters]...\n"
    return text[:limit] + note if keep == "head" else note + text[-limit:]


def child_env(home: Path) -> dict[str, str]:
    """A minimal environment for the child: no API keys or other secrets are passed on."""
    return {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def run_code(
    code: str, workspace: SessionWorkspace, data_refs: dict[str, str], settings: SandboxSettings
) -> SandboxResult:
    """Check and run generated code.

    Reads: the Parquet files the code opens. Writes: analysis.py, bootstrap.py,
    run_config.json, status.json and emit.json under the session's sandbox folder, plus
    whatever the code saves under artifacts/. Returns: the run outcome.
    """
    policy = check_code(code, settings)
    if not policy.allowed:
        return SandboxResult(ok=False, policy_violations=policy.violations,
                             error="Code rejected by the sandbox policy:\n- "
                             + "\n- ".join(policy.violations))  # fmt: skip
    config_path = _prepare(code, workspace, data_refs, settings)
    before = set(workspace.artifacts_dir.iterdir())
    started = time.monotonic()
    stdout, stderr, returncode, timed_out = _execute(workspace, config_path, settings)
    result = _collect(workspace, settings, stdout, stderr, returncode, timed_out)
    new_files = sorted(str(p) for p in set(workspace.artifacts_dir.iterdir()) - before)
    return result.model_copy(update={"duration_s": round(time.monotonic() - started, 3),
                                     "artifacts": new_files})  # fmt: skip


def _prepare(
    code: str, workspace: SessionWorkspace, data_refs: dict[str, str], settings: SandboxSettings
) -> Path:
    workspace.ensure()
    folder = workspace.sandbox_dir
    for stale in (workspace.emit_path, folder / STATUS_FILENAME):
        stale.unlink(missing_ok=True)
    (folder / CODE_FILENAME).write_text(code, encoding="utf-8")
    (folder / BOOTSTRAP_FILENAME).write_text(BOOTSTRAP, encoding="utf-8")
    config = {
        "data": data_refs,
        "artifacts_dir": str(workspace.artifacts_dir),
        "emit_path": str(workspace.emit_path),
        "status_path": str(folder / STATUS_FILENAME),
        "code_path": str(folder / CODE_FILENAME),
        "memory_limit_mb": settings.memory_limit_mb,
    }
    config_path = folder / CONFIG_FILENAME
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _execute(
    workspace: SessionWorkspace, config_path: Path, settings: SandboxSettings
) -> tuple[str, str, int | None, bool]:
    command = [sys.executable, "-I", str(workspace.sandbox_dir / BOOTSTRAP_FILENAME),
               str(config_path)]  # fmt: skip
    process = subprocess.Popen(
        command, cwd=workspace.session_dir, env=child_env(workspace.session_dir),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=os.name == "posix",
    )  # fmt: skip
    try:
        stdout, stderr = process.communicate(timeout=settings.timeout_seconds)
        return stdout, stderr, process.returncode, False
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        stdout, stderr = process.communicate()
        return stdout, stderr, None, True


def _collect(
    workspace: SessionWorkspace,
    settings: SandboxSettings,
    stdout: str,
    stderr: str,
    returncode: int | None,
    timed_out: bool,
) -> SandboxResult:
    limit = settings.output_char_limit
    memory = _memory_status(workspace)
    base = {"stdout": truncate(stdout, limit), "memory_limit": memory, "timed_out": timed_out}
    if timed_out:
        return SandboxResult(
            ok=False, error=f"Timed out after {settings.timeout_seconds} s", **base
        )
    if returncode != 0:
        error = truncate(stderr.strip() or f"exit code {returncode}", limit, keep="tail")
        return SandboxResult(ok=False, error=error, **base)
    if not workspace.emit_path.exists():
        return SandboxResult(ok=False, error="The code finished without calling emit(...)", **base)
    try:
        payload = EmitPayload.model_validate_json(workspace.emit_path.read_text())
    except ValidationError as exc:
        return SandboxResult(ok=False, error=truncate(f"emit() payload is invalid: {exc}", limit),
                             **base)  # fmt: skip
    return SandboxResult(ok=True, emitted=payload, **base)


def _memory_status(workspace: SessionWorkspace) -> str:
    path = workspace.sandbox_dir / STATUS_FILENAME
    status = json.loads(path.read_text()).get("memory_limit", "") if path.exists() else "unknown"
    if status != "applied":
        log = session_logger(__name__, workspace.session_id, "sandbox")
        log.warning("sandbox memory limit %s; running without it", status)
    return status
