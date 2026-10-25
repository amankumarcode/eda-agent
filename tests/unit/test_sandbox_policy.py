import pytest

from dva.sandbox.policy import check_code


@pytest.mark.parametrize(
    "code",
    [
        "import pandas as pd\nimport numpy as np\nimport json, math, statistics, re",
        "from collections import Counter\nfrom itertools import chain\nimport datetime as dt",
        "import pandas.api.types as t",
        "df = pd.read_parquet(DATA['sales'])\nemit([], [])",
    ],
)
def test_allowed_code_passes(settings, code):
    assert check_code(code, settings.sandbox).allowed


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("import os", "import of 'os'"),
        ("import subprocess", "import of 'subprocess'"),
        ("import socket", "import of 'socket'"),
        ("import pathlib", "import of 'pathlib'"),
        ("from os import system", "import from 'os'"),
        ("from . import x", "import from '.'"),
        ("__import__('os')", "use of '__import__'"),
        ("eval('1+1')", "use of 'eval'"),
        ("exec('x=1')", "use of 'exec'"),
        ("compile('1', 'f', 'eval')", "use of 'compile'"),
        ("f = eval", "use of 'eval'"),
        ("getattr(pd, 'read_csv')", "use of 'getattr'"),
        ("open('x.txt', 'w')", "use of 'open'"),
        ("os.system('ls')", "access to module 'os'"),
        ("sys.exit(0)", "access to module 'sys'"),
        ("pd.io.common.os.system('ls')", "access to module 'os'"),
        ("().__class__.__subclasses__()", "dunder attribute '__class__'"),
        ("def f(:", "syntax error"),
    ],
)
def test_blocked_code_is_rejected(settings, code, message):
    result = check_code(code, settings.sandbox)
    assert not result.allowed
    assert any(message in v for v in result.violations), result.violations


def test_violations_name_the_line_and_are_deduplicated(settings):
    result = check_code("x = 1\nimport os\nimport os", settings.sandbox)
    assert result.violations == ["line 2: import of 'os' is not allowed",
                                 "line 3: import of 'os' is not allowed"]  # fmt: skip
