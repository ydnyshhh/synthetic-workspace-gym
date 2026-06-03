from __future__ import annotations

from typing import Any

try:
    import verifiers as vf  # type: ignore[import-not-found]
except ImportError:
    vf = None  # type: ignore[assignment]


class VerifiersUnavailableError(ImportError):
    """Raised when native Verifiers support is requested without the package."""


def is_verifiers_available() -> bool:
    return vf is not None


def require_verifiers() -> Any:
    if vf is None:
        raise VerifiersUnavailableError(
            "Native verifiers integration requires the optional `verifiers` package. "
            "Install with `uv sync --extra verifiers` or `pip install verifiers`."
        )
    return vf
