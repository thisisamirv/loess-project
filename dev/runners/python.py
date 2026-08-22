from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time

from .base import RunResult, Snippet

_PYTHON_PREAMBLE = "import fastloess as fl\nimport numpy as np\n"

# Set to a venv python that has fastloess installed (overridden by main()).
PYTHON_BIN: str = sys.executable


def skip_reason(snippet: Snippet) -> str | None:
    code = snippet.code
    if re.search(r"total_points\s*=\s*[1-9][0-9]{4,}", code):
        return "large synthetic dataset (too slow for CI)"
    return None


def run_python(snippet: Snippet, timeout: int) -> RunResult:
    code = _PYTHON_PREAMBLE + snippet.code
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp = f.name
    try:
        t0 = time.monotonic()
        proc = subprocess.run(
            [PYTHON_BIN, tmp],
            capture_output=True,
            check=False,
            timeout=timeout,
            text=True,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        dur = time.monotonic() - t0
        return RunResult(
            snippet=snippet,
            runner="python",
            passed=(proc.returncode == 0),
            duration=dur,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            snippet=snippet,
            runner="python",
            passed=False,
            duration=timeout,
            stderr=f"Timed out after {timeout}s",
        )
    finally:
        os.unlink(tmp)
