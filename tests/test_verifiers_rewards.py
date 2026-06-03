from __future__ import annotations

import unittest

from synthetic_workspace_gym.verifiers.rewards import compute_reward, to_verifiers_info, to_verifiers_reward


class VerifiersRewardTests(unittest.TestCase):
    def test_score_mode(self) -> None:
        self.assertEqual(compute_reward({"score": 0.4, "success": False}, mode="score"), 0.4)

    def test_binary_mode(self) -> None:
        self.assertEqual(compute_reward({"score": 0.4, "success": False}, mode="binary"), 0.0)
        self.assertEqual(compute_reward({"score": 1.0, "success": True}, mode="binary"), 1.0)

    def test_subscore_mode(self) -> None:
        payload = {"score": 0.5, "subscores": {"files": 0.25}}

        self.assertEqual(compute_reward(payload, mode="subscore:files"), 0.25)

    def test_weighted_mode(self) -> None:
        payload = {"score": 0.0, "subscores": {"a": 0.5, "b": 0.25}}

        self.assertEqual(compute_reward(payload, mode="weighted", weights={"a": 2.0, "b": 1.0}), 1.25)

    def test_missing_fields_are_defensive(self) -> None:
        self.assertEqual(to_verifiers_reward({}), 0.0)
        info = to_verifiers_info({})

        self.assertFalse(info["success"])
        self.assertEqual(info["score"], 0.0)
        self.assertEqual(info["subscores"], {})


if __name__ == "__main__":
    unittest.main()
