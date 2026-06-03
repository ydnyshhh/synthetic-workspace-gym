from __future__ import annotations

import unittest

from synthetic_workspace_gym.splits import default_split_policy, normalize_split_name
from synthetic_workspace_gym.splits.policy import (
    heldout_scenarios_for_family,
    in_distribution_scenarios_for_family,
    scenario_pool_for_family,
)


class SplitPolicyTests(unittest.TestCase):
    def test_normalize_split_aliases(self) -> None:
        self.assertEqual(normalize_split_name("val"), "validation")
        self.assertEqual(normalize_split_name("dev"), "validation")
        self.assertEqual(normalize_split_name("heldout"), "heldout")

    def test_default_policy_seed_and_difficulty_ranges(self) -> None:
        specs = default_split_policy(families=("script_repair",))

        self.assertEqual(specs["train"].difficulties, [1, 2, 3])
        self.assertEqual(specs["validation"].seeds, list(range(80, 90)))
        self.assertEqual(specs["test"].seeds, list(range(90, 100)))
        self.assertEqual(specs["heldout"].seeds, list(range(100, 120)))

    def test_default_policy_uses_scenario_heldout(self) -> None:
        in_distribution_list = in_distribution_scenarios_for_family("script_repair")
        heldout_list = heldout_scenarios_for_family("script_repair")
        in_distribution = set(in_distribution_list)
        heldout = set(heldout_list)

        self.assertIn("csv_schema_drift", in_distribution)
        self.assertIn("team_roster_export", heldout)
        self.assertFalse(in_distribution.intersection(heldout))
        self.assertEqual(scenario_pool_for_family("script_repair"), in_distribution_list + heldout_list)


if __name__ == "__main__":
    unittest.main()
