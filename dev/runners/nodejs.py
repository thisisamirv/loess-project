"""Node.js snippet runner."""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from pathlib import Path

from .base import REPO_ROOT, RunResult, Snippet, _find_exe


def skip_reason(snippet: Snippet) -> str | None:
    return None


_NODEJS_DIR = REPO_ROOT / "bindings" / "nodejs"


def _ensure_nodejs_selflink(nodejs_dir: Path) -> None:
    """Create node_modules/fastloess shim so require('fastloess') resolves locally."""
    nm_fastloess = nodejs_dir / "node_modules" / "fastloess"
    if nm_fastloess.exists():
        return
    nm_fastloess.mkdir(parents=True, exist_ok=True)
    (nm_fastloess / "index.js").write_text(
        "module.exports = require('../../');\n", encoding="utf-8"
    )
    (nm_fastloess / "package.json").write_text(
        '{"name":"fastloess","main":"index.js","version":"0.0.0"}\n',
        encoding="utf-8",
    )


def run_nodejs(snippet: Snippet, timeout: int) -> RunResult:
    node_bin = _find_exe("node")
    if node_bin is None:
        return RunResult(
            snippet=snippet,
            runner="nodejs",
            skipped=True,
            skip_reason="node not found in PATH",
        )

    cwd = str(_NODEJS_DIR) if _NODEJS_DIR.exists() else str(REPO_ROOT)
    if _NODEJS_DIR.exists():
        _ensure_nodejs_selflink(_NODEJS_DIR)

    tmp_name = f"_snippet_{uuid.uuid4().hex}.js"
    tmp = str(Path(cwd) / tmp_name)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(snippet.code)

    try:
        t0 = time.monotonic()
        proc = subprocess.run(
            [node_bin, tmp],
            capture_output=True,
            check=False,
            timeout=timeout,
            text=True,
            cwd=cwd,
        )
        dur = time.monotonic() - t0
        return RunResult(
            snippet=snippet,
            runner="nodejs",
            passed=(proc.returncode == 0),
            duration=dur,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            snippet=snippet,
            runner="nodejs",
            passed=False,
            duration=timeout,
            stderr=f"Timed out after {timeout}s",
        )
    finally:
        os.unlink(tmp)
