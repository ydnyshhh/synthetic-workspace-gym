from __future__ import annotations

from contextlib import contextmanager
import shutil
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TEMP_ROOT = ROOT / ".tmp-tests"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@contextmanager
def workspace_tempdir():
    path = TEMP_ROOT / f"tmp-{uuid4().hex[:10]}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield str(path)
    finally:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
