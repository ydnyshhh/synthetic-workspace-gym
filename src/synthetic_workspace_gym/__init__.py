"""Synthetic Workspace Gym."""

from .hub import load_environment
from .provenance import ENVIRONMENT_VERSION

__version__ = ENVIRONMENT_VERSION

__all__ = ["__version__", "load_environment"]
