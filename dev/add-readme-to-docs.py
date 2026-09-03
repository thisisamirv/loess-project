#!/usr/bin/env python3
"""Embed a binding's top-level README.md as its docs-site home page.

Auto-detects the docs-site flavor from what exists in the binding directory:
  - Starlight (bindings/nodejs, bindings/wasm): rewrites the body of the
    already-existing src/content/docs/index.md, preserving its own
    hand-authored frontmatter (hero, etc.) above it.
  - Sphinx (bindings/python): rewrites docs/index.md with the README body,
    preserving the existing hidden `:::{toctree}` block at the end (Sphinx's
    site navigation is defined inline in the root doc, unlike Starlight's
    separate nav file).

In both cases the README's redundant top-level `# LOESS Project` heading is
stripped, since the page's own title (Starlight hero / Sphinx toctree owner)
already covers it.

Usage:
    python dev/add-readme-to-docs.py <binding_dir>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

BINDING_DIR = (
    Path(sys.argv[1]).resolve()
    if len(sys.argv) > 1
    else REPO_ROOT / "bindings" / "nodejs"
)
README_PATH = BINDING_DIR / "README.md"

# Drops the leading "# LOESS Project" (optionally preceded by an HTML
# comment, e.g. a markdownlint-disable directive) so the README's own H1
# isn't duplicated below the page's title.
H1_RE = re.compile(r"^(<!--[^\n]*-->\n)?# .+\n\n?")


def _read_readme() -> str:
    return README_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")


def _embed_starlight(index_path: Path) -> None:
    existing = index_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    fm_match = re.match(r"^---\n[\s\S]*?\n---\n", existing)
    if not fm_match:
        raise ValueError(f"No frontmatter block found in {index_path}")
    frontmatter = fm_match.group(0)

    body = H1_RE.sub(lambda m: (m.group(1) or "") + "\n", _read_readme(), count=1)
    index_path.write_text(f"{frontmatter}\n{body}", encoding="utf-8")


def _embed_sphinx(index_path: Path) -> None:
    existing = index_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    toctree_match = re.search(r":::\{toctree\}[\s\S]*?\n:::\n?", existing)
    if not toctree_match:
        raise ValueError(f"No toctree block found in {index_path}")
    toctree = toctree_match.group(0)

    # Unlike Starlight, Sphinx has no separate title mechanism (frontmatter/
    # hero) that renders into the page body, so the README's own H1 is kept
    # as the page's visible title instead of being stripped.
    body = _read_readme()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(f"{body.rstrip()}\n\n{toctree}", encoding="utf-8")


def main() -> None:
    starlight_index = BINDING_DIR / "src" / "content" / "docs" / "index.md"
    sphinx_conf = BINDING_DIR / "docs" / "conf.py"

    if starlight_index.exists():
        index_path = starlight_index
        _embed_starlight(index_path)
    elif sphinx_conf.exists():
        index_path = BINDING_DIR / "docs" / "index.md"
        _embed_sphinx(index_path)
    else:
        sys.exit(
            f"Don't know how to embed README for {BINDING_DIR}: neither "
            f"{starlight_index} nor {sphinx_conf} exist."
        )

    print(f"Embedded README.md into {index_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
