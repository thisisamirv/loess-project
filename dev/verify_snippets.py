"""Extract fenced code blocks from docs/ and run them to verify correctness.

Each snippet is prefixed with language-specific boilerplate that provides common
sample data and imports, so fragment-style doc examples can execute end-to-end.

Usage
-----
    python dev/verify_snippets.py                    # run all supported languages
    python dev/verify_snippets.py --lang python      # Python only
    python dev/verify_snippets.py --lang nodejs      # Node.js only
    python dev/verify_snippets.py --lang julia       # Julia only
    python dev/verify_snippets.py --file docs/api/python.md
    python dev/verify_snippets.py --dry-run          # list snippets, don't run
    python dev/verify_snippets.py --verbose          # show snippet source on failure
    python dev/verify_snippets.py --output out.json  # also write JSON report
    python dev/verify_snippets.py --timeout 60       # per-snippet timeout (seconds)
    python dev/verify_snippets.py --stop-on-fail     # exit after first failure
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"

# ---------------------------------------------------------------------------
# Terminal colours (disabled on non-TTY or Windows without colour support)
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and os.name != "nt" or os.environ.get("FORCE_COLOR")


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def green(t: str) -> str:
    return _c("32", t)


def red(t: str) -> str:
    return _c("31", t)


def yellow(t: str) -> str:
    return _c("33", t)


def cyan(t: str) -> str:
    return _c("36", t)


def bold(t: str) -> str:
    return _c("1", t)


# ---------------------------------------------------------------------------
# Python executable detection (prefer venv where fastloess is installed)
# ---------------------------------------------------------------------------

_PYTHON_BIN: str = sys.executable  # may be replaced in main()


def _find_python_with_fastloess() -> str:
    """Return the best Python executable that has fastloess installed."""
    candidates = [
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",  # Windows root venv
        REPO_ROOT / ".venv" / "bin" / "python",  # Unix root venv
        REPO_ROOT / "bindings" / "python" / ".venv" / "Scripts" / "python.exe",
        REPO_ROOT / "bindings" / "python" / ".venv" / "bin" / "python",
    ]
    for c in candidates:
        if c.exists():
            try:
                r = subprocess.run(
                    [str(c), "-c", "import fastloess"],
                    capture_output=True,
                    timeout=10,
                )
                if r.returncode == 0:
                    return str(c)
            except Exception:
                pass
    return sys.executable


# ---------------------------------------------------------------------------
# Language-specific boilerplate injected before every snippet
# ---------------------------------------------------------------------------

# Tab labels in the docs that map to each runner
_TAB_ALIASES: dict[str, set[str]] = {
    "python": {"Python"},
    "julia": {"Julia"},
    "nodejs": {"Node.js"},
    # These are not run by default (need external toolchain)
    "wasm": {"WebAssembly"},
    "r": {"R"},
    "cpp": {"C++"},
    "rust": {
        "Rust",
        "Rust (fastLoess)",
        "loess-rs (no_std compatible)",
        "fastLoess (parallel)",
    },
}

# Code-block language tags for each runner
_LANG_TAGS: dict[str, set[str]] = {
    "python": {"python"},
    "julia": {"julia"},
    "nodejs": {"javascript", "js", "typescript", "ts"},
    "wasm": {"javascript", "js"},
    "r": {"r"},
    "cpp": {"cpp", "c++"},
    "rust": {"rust"},
}

_PYTHON_PREAMBLE = textwrap.dedent("""\
    # --- snippet preamble: suppress display back-end -------------------------
    import sys as _sys, os as _os
    try:
        import matplotlib as _mpl
        _mpl.use("Agg")
        import matplotlib.pyplot as plt
        plt.show = lambda *a, **kw: None
    except ImportError:
        pass

    # --- snippet preamble: common imports ------------------------------------
    import numpy as np
    import fastloess as fl

    # --- snippet preamble: sample data ---------------------------------------
    np.random.seed(42)
    _n = 100
    x = np.linspace(0.0, 10.0, _n)
    y = np.sin(x) + np.random.normal(0, 0.1, _n)

    # Aliases used in various tutorials
    t = x.copy()
    t_irregular = x.copy()
    y_irregular = y.copy()
    positions = x.copy()
    observed = y.copy()
    times = x.copy()
    temperatures = y.copy()
    hours = x.copy()
    expression = y.copy()
    coverage = np.abs(y.copy()) * 10 + 5

    # Multivariate input variables (lat/lon/x1/x2/x3 for dimensions.md examples)
    lat = x.copy(); lon = x * 0.5
    x1 = x.copy(); x2 = x * 0.5; x3 = x * 0.25
    z = np.sin(x) + np.cos(x * 0.5)
    # Python binding requires flattened 1D for multi-dim input
    x2d = np.column_stack([x, x * 0.5]).ravel()    # (200,) flat row-major
    x3d = np.column_stack([x, x * 0.5, x * 0.25]).ravel()  # (300,) flat

    # Outlier / weight examples
    y_with_outlier = y.copy();  y_with_outlier[50] = 100.0
    weights = np.ones(_n)

    # Streaming / chunk examples
    chunk_size, overlap = 50, 10
    chunk1_x, chunk1_y = x[:50].copy(), y[:50].copy()
    chunk2_x, chunk2_y = x[50:].copy(), y[50:].copy()
    x_chunk, y_chunk = x[:50].copy(), y[:50].copy()

    # Sliding-window examples
    data_x = list(x[:30])
    data_y = list(y[:30])

    # API doc pseudocode helpers
    fastloess = fl   # allow docs that use bare "fastloess.X(...)"
    kwargs = {'fraction': 0.3}
    model = fl.Loess()
    stream = fl.StreamingLoess()
    online = fl.OnlineLoess()

    # -------------------------------------------------------------------------
""")

_JULIA_PREAMBLE = textwrap.dedent("""\
    # --- snippet preamble ----------------------------------------------------
    using FastLOESS
    using Random, Printf, Statistics
    Random.seed!(42)

    _n = 100
    x  = collect(LinRange(0.0, 10.0, _n))
    y  = sin.(x) .+ randn(_n) .* 0.1

    t           = copy(x)
    t_irregular = copy(x)
    y_irregular = copy(y)
    positions   = copy(x)
    observed    = copy(y)
    times       = copy(x)
    temperatures = copy(y)
    hours       = copy(x)
    expression  = copy(y)
    coverage    = abs.(y) .* 10.0 .+ 5.0

    x2d = hcat(x, x .* 0.5)
    x3d = hcat(x, x .* 0.5, x .* 0.25)
    z   = sin.(x) .+ cos.(x .* 0.5)
    lat = copy(x); lon = x .* 0.5
    x1  = copy(x); x2 = x .* 0.5; x3 = x .* 0.25

    y_with_outlier = copy(y); y_with_outlier[50] = 100.0
    weights = ones(Float64, _n)

    chunk1_x, chunk1_y = x[1:50], y[1:50]
    chunk2_x, chunk2_y = x[51:end], y[51:end]
    x_chunk, y_chunk   = x[1:50], y[1:50]

    data_x = copy(x[1:30])
    data_y = copy(y[1:30])

    # API doc placeholders (method-signature examples use these as variables)
    model  = Loess()
    stream = StreamingLoess()
    online = OnlineLoess()
    kwargs = (fraction=0.3,)
    # -------------------------------------------------------------------------
""")

_NODEJS_PREAMBLE = textwrap.dedent("""\
    // --- snippet preamble ----------------------------------------------------
    'use strict';
    const fl = (() => { try { return require('fastloess'); } catch (e) { return null; } })();
    if (!fl) { console.error('fastloess not found — skip'); process.exit(0); }
    const { Loess, StreamingLoess, OnlineLoess } = fl;
    const fastloess = fl;

    const _n = 100;
    const x = new Float64Array(_n).map((_, i) => i * 0.1);
    const y = new Float64Array(x.map(xi => Math.sin(xi) + (Math.random() - 0.5) * 0.2));

    const t            = new Float64Array(x);
    const tIrregular   = new Float64Array(x);
    const yIrregular   = new Float64Array(y);
    const positions    = new Float64Array(x);
    const observed     = new Float64Array(y);
    const times        = new Float64Array(x);
    const temperatures = new Float64Array(y);
    const hours        = new Float64Array(x);
    const expression   = new Float64Array(y);
    const coverage     = new Float64Array(y.map(yi => Math.abs(yi) * 10 + 5));

    const x2d = { x: Array.from(x), z: Array.from(x.map(xi => xi * 0.5)) };
    const z = new Float64Array(y.map((yi, i) => yi + Math.cos(x[i] * 0.5)));

    const yWithOutlier = new Float64Array(y); yWithOutlier[50] = 100.0;
    const weights = new Float64Array(_n).fill(1.0);

    const chunk1_x = x.slice(0, 50); const chunk1_y = y.slice(0, 50);
    const chunk2_x = x.slice(50);    const chunk2_y = y.slice(50);
    let data_x = Array.from(x.slice(0, 30));
    let data_y = Array.from(y.slice(0, 30));
    const xArr = new Float64Array(x); const yArr = new Float64Array(y);
    let windowX = Array.from(x.slice(0, 20));
    let windowY = Array.from(y.slice(0, 20));
    // -------------------------------------------------------------------------
""")

PREAMBLES: dict[str, str] = {
    "python": _PYTHON_PREAMBLE,
    "julia": _JULIA_PREAMBLE,
    "nodejs": _NODEJS_PREAMBLE,
}

# ---------------------------------------------------------------------------
# Snippet data class
# ---------------------------------------------------------------------------


@dataclass
class Snippet:
    file: Path
    line: int  # 1-based line number of the opening fence
    lang_tag: str  # code-block language tag (e.g. "python")
    tab: Optional[str]  # nearest === "Tab" label, or None
    code: str

    @property
    def runner(self) -> Optional[str]:
        """Return which runner handles this snippet, or None to skip."""
        for runner, tags in _LANG_TAGS.items():
            if self.lang_tag.lower() in tags:
                # For JS: distinguish Node.js from WASM by tab label
                if runner in ("nodejs", "wasm"):
                    if self.tab in _TAB_ALIASES["wasm"]:
                        return "wasm"
                    if self.tab in _TAB_ALIASES["nodejs"]:
                        return "nodejs"
                    # No tab: fall back to nodejs if no WASM markers
                    if (
                        "fastloess-wasm" not in self.code
                        and "import {" not in self.code[:80]
                    ):
                        return "nodejs"
                    return "wasm"
                return runner
        return None

    @property
    def label(self) -> str:
        tab = f" [{self.tab}]" if self.tab else ""
        return f"{self.file.relative_to(REPO_ROOT)}:{self.line}{tab}"


# ---------------------------------------------------------------------------
# Markdown parser
# ---------------------------------------------------------------------------

_TAB_RE = re.compile(r'^[ \t]*===\s+"([^"]+)"', re.MULTILINE)
_FENCE_RE = re.compile(r"^([ \t]*)```(\w+)", re.MULTILINE)


def extract_snippets(md_file: Path) -> List[Snippet]:
    """Extract all fenced code blocks from a markdown file."""
    text = md_file.read_text(encoding="utf-8")
    lines = text.splitlines()
    result: List[Snippet] = []

    current_tab: Optional[str] = None
    i = 0
    while i < len(lines):
        line = lines[i]

        # Track tab context
        m = _TAB_RE.match(line)
        if m:
            current_tab = m.group(1)
            i += 1
            continue

        # Detect fence opening: optional leading whitespace then ```lang
        m = re.match(r"^([ \t]*)```(\w+)\s*$", line)
        if m:
            fence_indent = m.group(1)
            lang_tag = m.group(2)
            start_line = i + 1  # 1-based
            code_lines: List[str] = []
            i += 1
            while i < len(lines):
                close = lines[i]
                # Closing fence: same indent + ```
                if re.match(r"^" + re.escape(fence_indent) + r"```\s*$", close):
                    i += 1
                    break
                # Strip the common fence indent
                stripped = (
                    close[len(fence_indent) :]
                    if close.startswith(fence_indent)
                    else close
                )
                code_lines.append(stripped)
                i += 1
            code = "\n".join(code_lines)
            result.append(
                Snippet(
                    file=md_file,
                    line=start_line,
                    lang_tag=lang_tag,
                    tab=current_tab,
                    code=code,
                )
            )
            # A tab label covers only the next block (reset after capture)
            current_tab = None
            continue

        # Reset tab on section headers or dividers
        if line.startswith("#") or line.strip() == "---":
            current_tab = None

        i += 1

    return result


def should_skip(snippet: Snippet, runner: str) -> Optional[str]:
    """Return a skip reason string, or None if the snippet should be run."""
    code = snippet.code

    # MkDocs file-include directives are not runnable
    if "--8<--" in code:
        return "file-include (--8<--)"

    # Empty or whitespace-only
    if not code.strip():
        return "empty"

    # Skip snippets that reference variables or packages we can't supply
    if runner == "python":
        # Genomic-specific heavy I/O (would need real files)
        if any(s in code for s in ["read_csv", "open(", "glob(", "argparse"]):
            return "file I/O"
        # Lines that are obviously just output examples (no executable Python)
        if not any(c in code for c in ["=", "(", "import", "print"]):
            return "no executable statements"

    if runner == "julia":
        # Skip package-management / installation snippets
        if re.search(r"\bPkg\.(add|develop|clone|rm|pin)\s*\(", code):
            return "Pkg management snippet"
    if runner == "nodejs":
        # TypeScript-only syntax (type annotations)
        if ": SmoothOptions" in code or ": LoessResult" in code:
            return "TypeScript (not Node.js)"

    return None


# ---------------------------------------------------------------------------
# Node.js: strip redeclarations that conflict with the preamble
# ---------------------------------------------------------------------------

# Variables the Node.js preamble already declares at the top level
_NODEJS_PREAMBLE_VARS: frozenset = frozenset(
    {
        "fl",
        "fastloess",
        "Loess",
        "StreamingLoess",
        "OnlineLoess",
        # data arrays — snippets often redeclare these with small sample data
        "x",
        "y",
        "z",
        "t",
        "weights",
        "yWithOutlier",
        "positions",
        "observed",
        "times",
        "temperatures",
        "hours",
        "expression",
        "coverage",
        "chunk1_x",
        "chunk1_y",
        "chunk2_x",
        "chunk2_y",
        "data_x",
        "data_y",
        "xArr",
        "yArr",
        "windowX",
        "windowY",
    }
)


def _strip_nodejs_redeclarations(code: str) -> str:
    """Remove declarations that would shadow preamble const bindings.

    Handles single-line and multi-line declarations by tracking bracket depth.
    """
    result = []
    skipping = False  # inside a multi-line declaration being stripped
    depth = 0  # net ( [ { depth while skipping

    for line in code.splitlines():
        s = line.strip()

        if skipping:
            for ch in line:
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
            if depth <= 0:
                skipping = False
                depth = 0
            continue  # drop continuation line

        # Detect lines to strip
        strip_line = False
        if re.match(
            r"""(?:const|let|var)\s+\S.*=\s*require\(\s*['"]fastloess['"]\s*\)""", s
        ):
            strip_line = True
        elif re.match(
            r"""(?:const|let|var)\s+\{[^}]+\}\s*=\s*(?:fl|fastloess)\s*;?\s*$""", s
        ):
            strip_line = True
        else:
            m = re.match(r"(?:const|let|var)\s+(\w+)\s*=", s)
            if m and m.group(1) in _NODEJS_PREAMBLE_VARS:
                strip_line = True

        if strip_line:
            for ch in line:
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
            if depth > 0:
                skipping = True  # multi-line declaration
            else:
                depth = 0
            continue

        result.append(line)

    return "\n".join(result)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    snippet: Snippet
    runner: str
    skipped: bool = False
    skip_reason: str = ""
    passed: bool = False
    duration: float = 0.0
    stdout: str = ""
    stderr: str = ""
    returncode: int = -1


def run_python(snippet: Snippet, timeout: int) -> RunResult:
    code = PREAMBLES["python"] + snippet.code
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp = f.name
    try:
        t0 = time.monotonic()
        proc = subprocess.run(
            [_PYTHON_BIN, tmp],
            capture_output=True,
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


def run_julia(snippet: Snippet, timeout: int) -> RunResult:
    code = PREAMBLES["julia"] + snippet.code
    with tempfile.NamedTemporaryFile(
        suffix=".jl", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp = f.name

    julia_bin = _find_exe("julia")
    if julia_bin is None:
        os.unlink(tmp)
        return RunResult(
            snippet=snippet,
            runner="julia",
            skipped=True,
            skip_reason="julia not found in PATH",
        )

    # Find the Julia project for the bindings
    julia_project = REPO_ROOT / "bindings" / "julia" / "julia"
    env = {**os.environ}
    if julia_project.exists():
        env["JULIA_PROJECT"] = str(julia_project)

    try:
        t0 = time.monotonic()
        proc = subprocess.run(
            [julia_bin, "--startup-file=no", tmp],
            capture_output=True,
            timeout=timeout,
            text=True,
            env=env,
        )
        dur = time.monotonic() - t0
        return RunResult(
            snippet=snippet,
            runner="julia",
            passed=(proc.returncode == 0),
            duration=dur,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            snippet=snippet,
            runner="julia",
            passed=False,
            duration=timeout,
            stderr=f"Timed out after {timeout}s",
        )
    finally:
        os.unlink(tmp)


def run_nodejs(snippet: Snippet, timeout: int) -> RunResult:
    code = PREAMBLES["nodejs"] + _strip_nodejs_redeclarations(snippet.code)
    with tempfile.NamedTemporaryFile(
        suffix=".js", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp = f.name

    node_bin = _find_exe("node")
    if node_bin is None:
        os.unlink(tmp)
        return RunResult(
            snippet=snippet,
            runner="nodejs",
            skipped=True,
            skip_reason="node not found in PATH",
        )

    # Run from the nodejs binding directory so require('fastloess') resolves
    nodejs_dir = REPO_ROOT / "bindings" / "nodejs"
    cwd = str(nodejs_dir) if nodejs_dir.exists() else str(REPO_ROOT)

    try:
        t0 = time.monotonic()
        proc = subprocess.run(
            [node_bin, tmp],
            capture_output=True,
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


_RUNNERS = {
    "python": run_python,
    "julia": run_julia,
    "nodejs": run_nodejs,
}


def _find_exe(name: str) -> Optional[str]:
    import shutil

    return shutil.which(name)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def iter_md_files(root: Path, file_filter: Optional[str]) -> Iterator[Path]:
    if file_filter:
        p = Path(file_filter)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.is_file():
            yield p
            return
        # Treat as glob
        yield from sorted(REPO_ROOT.glob(file_filter))
        return
    yield from sorted(root.rglob("*.md"))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lang",
        choices=["python", "julia", "nodejs", "all"],
        default="all",
        help="Which language runner to use (default: all)",
    )
    parser.add_argument(
        "--file",
        metavar="PATH_OR_GLOB",
        help="Restrict to a specific file or glob (relative to repo root)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List snippets without running them"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print snippet source and full output on failure",
    )
    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Stop after the first failing snippet",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Per-snippet timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--output", metavar="FILE", help="Write JSON report to this file"
    )
    args = parser.parse_args(argv)

    active_runners: set[str] = (
        {"python", "julia", "nodejs"} if args.lang == "all" else {args.lang}
    )

    # Detect best Python executable
    global _PYTHON_BIN
    _PYTHON_BIN = _find_python_with_fastloess()

    # ---- Collect snippets ---------------------------------------------------
    snippets: List[Snippet] = []
    for md in iter_md_files(DOCS_DIR, args.file):
        snippets.extend(extract_snippets(md))

    total_found = len(snippets)

    # Filter to only snippets we can handle
    runnable: List[Tuple[Snippet, str]] = []  # (snippet, runner)
    for s in snippets:
        r = s.runner
        if r is None or r not in active_runners:
            continue
        reason = should_skip(s, r)
        if reason:
            continue
        runnable.append((s, r))

    print(bold("\nfastLoess doc snippet verifier"))
    print(f"Docs dir : {DOCS_DIR}")
    print(f"Runners  : {', '.join(sorted(active_runners))}")
    print(f"Snippets : {len(runnable)} runnable / {total_found} total")
    if args.dry_run:
        print()
        for s, r in runnable:
            print(f"  {cyan(r):20s}  {s.label}")
        print()
        return 0
    print()

    # ---- Run snippets -------------------------------------------------------
    results: List[RunResult] = []
    n_pass = n_fail = n_skip = 0

    for s, runner in runnable:
        label = s.label
        sys.stdout.write(f"  {cyan(runner):20s}  {label} … ")
        sys.stdout.flush()

        run_fn = _RUNNERS.get(runner)
        if run_fn is None:
            print(yellow("SKIP (no runner)"))
            n_skip += 1
            results.append(
                RunResult(
                    snippet=s,
                    runner=runner,
                    skipped=True,
                    skip_reason="no runner implementation",
                )
            )
            continue

        res = run_fn(s, args.timeout)

        if res.skipped:
            print(yellow(f"SKIP ({res.skip_reason})"))
            n_skip += 1
        elif res.passed:
            print(green(f"PASS ({res.duration:.2f}s)"))
            n_pass += 1
        else:
            print(red(f"FAIL ({res.duration:.2f}s, exit {res.returncode})"))
            n_fail += 1
            if args.verbose:
                _print_failure(s, res)

        results.append(res)

        if args.stop_on_fail and n_fail > 0:
            print(red("\nStopped after first failure (--stop-on-fail)."))
            break

    # ---- Summary ------------------------------------------------------------
    print()
    print("-" * 60)
    print(bold("Summary"))
    print(
        f"  {green(f'PASS: {n_pass}'):30s}  {yellow(f'SKIP: {n_skip}'):30s}  {red(f'FAIL: {n_fail}')}"
    )
    print()

    # Print failures in verbose mode (already shown inline) or always in brief
    failures = [r for r in results if not r.passed and not r.skipped]
    if failures and not args.verbose:
        print(bold("Failed snippets:"))
        for r in failures:
            print(f"  {red('FAIL')} {r.snippet.label}")
            # Show first 5 lines of stderr
            if r.stderr.strip():
                for line in r.stderr.strip().splitlines()[:5]:
                    print(f"      {line}")
        print()

    # ---- JSON output --------------------------------------------------------
    if args.output:
        report = {
            "summary": {"pass": n_pass, "fail": n_fail, "skip": n_skip},
            "snippets": [
                {
                    "file": str(r.snippet.file.relative_to(REPO_ROOT)),
                    "line": r.snippet.line,
                    "lang": r.snippet.lang_tag,
                    "tab": r.snippet.tab,
                    "runner": r.runner,
                    "status": "skip" if r.skipped else ("pass" if r.passed else "fail"),
                    "skip_reason": r.skip_reason if r.skipped else None,
                    "returncode": r.returncode if not r.skipped else None,
                    "duration": round(r.duration, 3),
                    "stderr": r.stderr[:2000] if r.stderr else "",
                }
                for r in results
            ],
        }
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written to {args.output}")

    return 1 if n_fail > 0 else 0


def _print_failure(snippet: Snippet, res: RunResult) -> None:
    """Print detailed failure information."""
    sep = "-" * 56
    print()
    print(f"  {sep}")
    print(f"  {bold('File:')} {snippet.label}")
    if snippet.tab:
        print(f"  {bold('Tab:')}  {snippet.tab}")
    print(f"  {bold('Code:')}")
    for line in snippet.code.splitlines()[:20]:
        print(f"    {line}")
    if len(snippet.code.splitlines()) > 20:
        print(f"    ... ({len(snippet.code.splitlines())} lines total)")
    if res.stderr.strip():
        print(f"  {bold('Stderr:')}")
        for line in res.stderr.strip().splitlines()[-20:]:
            print(f"    {line}")
    if res.stdout.strip():
        print(f"  {bold('Stdout:')}")
        for line in res.stdout.strip().splitlines()[-10:]:
            print(f"    {line}")
    print(f"  {sep}")
    print()


if __name__ == "__main__":
    sys.exit(main())
