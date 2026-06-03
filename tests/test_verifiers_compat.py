from __future__ import annotations

import unittest

import test_support  # noqa: F401

import synthetic_workspace_gym.verifiers as swg_verifiers
from synthetic_workspace_gym.verifiers.compat import (
    VerifiersUnavailableError,
    is_verifiers_available,
    require_verifiers,
)


class VerifiersCompatTests(unittest.TestCase):
    def test_public_api_imports_without_verifiers_dependency(self) -> None:
        self.assertTrue(hasattr(swg_verifiers, "SyntheticWorkspaceVerifiersEnv"))
        self.assertTrue(hasattr(swg_verifiers, "SWGVerifiersDataset"))
        self.assertTrue(hasattr(swg_verifiers, "compute_reward"))

    def test_is_verifiers_available_returns_bool(self) -> None:
        self.assertIsInstance(is_verifiers_available(), bool)

    def test_require_verifiers_raises_clear_error_when_unavailable(self) -> None:
        if is_verifiers_available():
            self.assertIsNotNone(require_verifiers())
        else:
            with self.assertRaisesRegex(VerifiersUnavailableError, "uv sync --extra verifiers"):
                require_verifiers()


if __name__ == "__main__":
    unittest.main()
