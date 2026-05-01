"""Local import bootstrap for src-layout development.

Python automatically imports `sitecustomize` on startup (after `site`).
Keeping this at repo root makes `src/` discoverable when commands are run
directly from the project directory without an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    package = src / "portfolio_toolkit"
    if package.exists():
        src_str = str(src)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


_ensure_src_on_path()
