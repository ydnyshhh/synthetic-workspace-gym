from __future__ import annotations

import unittest

import test_support  # noqa: F401

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

        self.assertEqual(set(specs), {"train", "validation", "test", "heldout"})
        self.assertEqual(specs["train"].difficulties, [1, 2, 3])
        self.assertEqual(specs["validation"].seeds, list(range(80, 90)))
        self.assertEqual(specs["test"].seeds, list(range(90, 100)))
        self.assertEqual(specs["heldout"].seeds, list(range(100, 120)))
        for spec in specs.values():
            self.assertTrue(all(1 <= difficulty <= 5 for difficulty in spec.difficulties))

    def test_default_policy_seed_ranges_are_disjoint(self) -> None:
        specs = default_split_policy(families=("script_repair",))
        seed_sets = {split: set(spec.seeds) for split, spec in specs.items()}

        self.assertFalse(seed_sets["train"].intersection(seed_sets["validation"]))
        self.assertFalse(seed_sets["train"].intersection(seed_sets["test"]))
        self.assertFalse(seed_sets["validation"].intersection(seed_sets["test"]))
        self.assertFalse(seed_sets["heldout"].intersection(seed_sets["train"]))

    def test_default_policy_uses_scenario_heldout(self) -> None:
        in_distribution_list = in_distribution_scenarios_for_family("script_repair")
        heldout_list = heldout_scenarios_for_family("script_repair")
        in_distribution = set(in_distribution_list)
        heldout = set(heldout_list)

        self.assertIn("csv_schema_drift", in_distribution)
        self.assertIn("team_roster_export", heldout)
        self.assertFalse(in_distribution.intersection(heldout))
        self.assertEqual(scenario_pool_for_family("script_repair"), in_distribution_list + heldout_list)

    def test_heldout_scenarios_disjoint_from_train_for_each_family(self) -> None:
        specs = default_split_policy()
        for family in specs["train"].families:
            train_scenarios = set(specs["train"].scenarios[family])
            heldout_scenarios = set(specs["heldout"].scenarios[family])

            self.assertFalse(train_scenarios.intersection(heldout_scenarios), family)


if __name__ == "__main__":
    unittest.main()
