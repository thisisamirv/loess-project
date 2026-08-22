"""R snippet runner."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import textwrap
import time

from .base import REPO_ROOT, RunResult, Snippet, _find_exe


def skip_reason(snippet: Snippet) -> str | None:
    code = snippet.code
    if re.search(r"\binstall\.packages\b|\bdevtools::", code):
        return "package installation snippet"
    if not any(c in code for c in ["<-", "=", "(", "library"]):
        return "no executable R statements"
    return None


_R_PREAMBLE = textwrap.dedent("""\
    suppressMessages({{
        .libPaths(c(
            normalizePath(file.path(
                Sys.getenv("LOESS_REPO_ROOT", "{repo_root}"),
                "bindings", "r", ".r-lib"
            ), mustWork = FALSE),
            .libPaths()
        ))
        library(rfastloess)
    }})
    suppressWarnings(pdf(NULL))
    plot.new()
""").format(repo_root=str(REPO_ROOT).replace("\\", "/"))


def run_r(snippet: Snippet, timeout: int) -> RunResult:
    rscript = _find_exe("Rscript")
    if rscript is None:
        return RunResult(
            snippet=snippet,
            runner="r",
            skipped=True,
            skip_reason="Rscript not found in PATH",
        )

    code = _R_PREAMBLE + snippet.code
    with tempfile.NamedTemporaryFile(
        suffix=".R", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp = f.name

    try:
        t0 = time.monotonic()
        proc = subprocess.run(
            [rscript, "--vanilla", tmp],
            capture_output=True,
            check=False,
            timeout=timeout,
            text=True,
            env={**os.environ, "LOESS_REPO_ROOT": str(REPO_ROOT)},
        )
        dur = time.monotonic() - t0
        return RunResult(
            snippet=snippet,
            runner="r",
            passed=(proc.returncode == 0),
            duration=dur,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            snippet=snippet,
            runner="r",
            passed=False,
            duration=timeout,
            stderr=f"Timed out after {timeout}s",
        )
    finally:
        os.unlink(tmp)
