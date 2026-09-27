"""Where the pipeline keeps its working files.

Run bundles go wherever the user asks (the repo's `output/` by default). Pass-one work
databases are written continuously during processing, so they live in a local cache
directory instead: a folder synced by OneDrive or Dropbox slows SQLite writes and can
lock the file mid-run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

CACHE_ENV = "SAFETY_TWIN_CACHE_DIR"


def cache_root() -> Path:
    override = os.environ.get(CACHE_ENV)
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "safety-twin"


def run_work_dir(run_id: str) -> Path:
    return cache_root() / "runs" / run_id
