from __future__ import annotations

import sys
import types

if sys.platform == "win32":
    fcntl = types.ModuleType("fcntl")
    fcntl.LOCK_EX = 2
    fcntl.LOCK_UN = 8
    fcntl.flock = lambda _fd, _operation: None
    sys.modules.setdefault("fcntl", fcntl)
