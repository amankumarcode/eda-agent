"""AST policy for generated code: an import allowlist and a list of blocked names.

This is a guardrail against mistakes and casual misuse, not a security boundary. Code
that passes can still, for example, read or write files through pandas.
"""

from __future__ import annotations

import ast

from pydantic import BaseModel

from dva.config import SandboxSettings


class PolicyResult(BaseModel):
    violations: list[str]

    @property
    def allowed(self) -> bool:
        return not self.violations


def check_code(code: str, settings: SandboxSettings) -> PolicyResult:
    """Check generated code before it runs. Reads nothing from disk. Returns violations.

    Rejects: imports outside the allowlist (and relative imports), any use of a blocked
    call name or blocked module name (also as an attribute, e.g. ``pd.io.common.os``), and
    dunder attribute access such as ``__class__``.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return PolicyResult(violations=[f"syntax error on line {exc.lineno}: {exc.msg}"])
    violations: list[str] = []
    for node in ast.walk(tree):
        violations += _check_node(node, settings)
    return PolicyResult(violations=list(dict.fromkeys(violations)))


def _check_node(node: ast.AST, settings: SandboxSettings) -> list[str]:
    allowed = set(settings.allowed_imports)
    line = getattr(node, "lineno", "?")
    if isinstance(node, ast.Import):
        return [f"line {line}: import of {a.name!r} is not allowed"
                for a in node.names if a.name.split(".")[0] not in allowed]  # fmt: skip
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if node.level or module.split(".")[0] not in allowed:
            return [f"line {line}: import from {'.' * node.level + module!r} is not allowed"]
        return []
    if isinstance(node, ast.Name) and node.id in settings.blocked_calls:
        return [f"line {line}: use of {node.id!r} is not allowed"]
    if isinstance(node, ast.Name) and node.id in settings.blocked_modules:
        return [f"line {line}: access to module {node.id!r} is not allowed"]
    if isinstance(node, ast.Attribute) and node.attr in settings.blocked_modules:
        return [f"line {line}: access to module {node.attr!r} is not allowed"]
    if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
        return [f"line {line}: dunder attribute {node.attr!r} is not allowed"]
    return []
