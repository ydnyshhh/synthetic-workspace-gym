from __future__ import annotations

import sys
from pathlib import Path

try:
    from synthetic_workspace_gym import load_environment
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from synthetic_workspace_gym import load_environment

__all__ = ["load_environment"]
