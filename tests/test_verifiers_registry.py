from __future__ import annotations

import unittest

from synthetic_workspace_gym.verifiers.compat import is_verifiers_available
from synthetic_workspace_gym.verifiers.env import SyntheticWorkspaceVerifiersEnv, adapt_to_verifiers, make_verifiers_env
from synthetic_workspace_gym.verifiers.registry import (
    get_environment_config,
    list_environments,
    make_environment,
    register_with_verifiers,
)


class VerifiersRegistryTests(unittest.TestCase):
    def test_list_environments_returns_known_ids(self) -> None:
        env_ids = list_environments()

        self.assertIn("swg.script_repair.csv_schema_drift", env_ids)
        self.assertIn("swg.tabular.monthly_segment_report", env_ids)

    def test_get_environment_config_and_make_environment(self) -> None:
        config = get_environment_config("swg.script_repair.csv_schema_drift")
        env = make_environment("swg.script_repair.csv_schema_drift", difficulty=1, seed=7)
        try:
            self.assertEqual(config["family"], "script_repair")
            self.assertIsInstance(env, SyntheticWorkspaceVerifiersEnv)
        finally:
            env.close()

    def test_register_with_verifiers_does_not_crash(self) -> None:
        self.assertIsInstance(register_with_verifiers(), bool)

    @unittest.skipUnless(is_verifiers_available(), "verifiers is unavailable")
    def test_make_verifiers_env_requires_native_package(self) -> None:
        env = make_verifiers_env(family="script_repair", scenario="csv_schema_drift", difficulty=1, seed=7)
        self.assertIsNotNone(env)

    @unittest.skipUnless(is_verifiers_available(), "verifiers is unavailable")
    def test_adapt_to_verifiers_returns_object(self) -> None:
        base = SyntheticWorkspaceVerifiersEnv(family="script_repair", scenario="csv_schema_drift", difficulty=1, seed=7)
        try:
            self.assertIsNotNone(adapt_to_verifiers(base))
        finally:
            base.close()


if __name__ == "__main__":
    unittest.main()
