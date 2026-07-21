from __future__ import annotations

import unittest

from synthetic_workspace_gym.generators.d5_quality import (
    summarize_atomic_oracle,
    summarize_composition_oracle,
    validate_atomic_oracle,
    validate_composition_oracle,
    validate_d5_realization,
)
from synthetic_workspace_gym.generators.difficulty_primitives import (
    CompositionSpec,
    CompositionStage,
    DefectBundle,
    coerce_defect_bundle,
)


class DifficultyPrimitiveTests(unittest.TestCase):
    def defect_bundle(self) -> DefectBundle:
        return DefectBundle(
            bundle_id="schema_chain",
            defect_ids=("version", "precedence", "normalize", "serialize"),
            dependency_edges=(
                ("version", "precedence"),
                ("precedence", "normalize"),
                ("normalize", "serialize"),
            ),
            capability_groups={
                "contract": ("version",),
                "transformation": ("precedence", "normalize"),
                "integration": ("serialize",),
            },
            required_files=(
                "config/schema.json",
                "src/parser.py",
                "src/normalize.py",
                "src/serializer.py",
            ),
        )

    def composition_spec(self) -> CompositionSpec:
        return CompositionSpec(
            stages=(
                CompositionStage(
                    stage_id="resolve_contract",
                    required_inputs=("docs/schema-v4.md",),
                    produced_artifacts=("artifacts/account_map.json",),
                    capability="contract_resolution",
                ),
                CompositionStage(
                    stage_id="apply_contract",
                    required_inputs=(
                        "artifacts/account_map.json",
                        "data/events.csv",
                    ),
                    produced_artifacts=("outputs/report.json",),
                    capability="integration",
                ),
            ),
            dependencies=(("resolve_contract", "apply_contract"),),
        )

    def test_defect_bundle_round_trips_legacy_mapping(self) -> None:
        bundle = self.defect_bundle()
        payload = bundle.to_dict()
        restored = coerce_defect_bundle(payload)

        self.assertEqual(restored.bundle_id, bundle.bundle_id)
        self.assertEqual(restored.defect_ids, bundle.defect_ids)
        self.assertEqual(restored.required_files, bundle.required_files)
        self.assertEqual(payload["semantic_dependency_depth"], 4)

    def test_defect_bundle_rejects_uncovered_defect(self) -> None:
        with self.assertRaisesRegex(ValueError, "cover every defect"):
            DefectBundle(
                bundle_id="invalid",
                defect_ids=("a", "b"),
                dependency_edges=(("a", "b"),),
                capability_groups={"contract": ("a",)},
                required_files=("src/a.py", "src/b.py"),
            )

    def test_composition_requires_downstream_artifact_consumption(self) -> None:
        with self.assertRaisesRegex(ValueError, "produced upstream"):
            CompositionSpec(
                stages=(
                    CompositionStage(
                        "stage_a",
                        ("docs/spec.md",),
                        ("artifacts/map.json",),
                        "retrieval",
                    ),
                    CompositionStage(
                        "stage_b",
                        ("data/input.csv",),
                        ("outputs/report.json",),
                        "transformation",
                    ),
                ),
                dependencies=(("stage_a", "stage_b"),),
            )

    def test_atomic_oracle_profile_enforces_quality_thresholds(self) -> None:
        labels = ("a", "b", "c", "d")
        scores = {
            frozenset(): 0.10,
            frozenset(("a",)): 0.20,
            frozenset(("b",)): 0.25,
            frozenset(("c",)): 0.30,
            frozenset(("d",)): 0.35,
            frozenset(("a", "b")): 0.40,
            frozenset(("a", "c")): 0.45,
            frozenset(("a", "d")): 0.50,
            frozenset(("b", "c")): 0.55,
            frozenset(("b", "d")): 0.60,
            frozenset(("c", "d")): 0.65,
            frozenset(("a", "b", "c")): 0.70,
            frozenset(("a", "b", "d")): 0.75,
            frozenset(("a", "c", "d")): 0.80,
            frozenset(("b", "c", "d")): 0.85,
            frozenset(labels): 1.0,
        }
        profile = summarize_atomic_oracle(scores, labels)
        validated = validate_atomic_oracle(profile)

        self.assertTrue(validated["valid"])
        self.assertEqual(validated["single_fix_max_reward"], 0.35)
        self.assertEqual(validated["pair_fix_max_reward"], 0.65)

    def test_composition_oracle_and_structural_gate(self) -> None:
        composition = self.composition_spec()
        oracle = summarize_composition_oracle(
            unmodified_reward=0.10,
            stage_a_only_reward=0.30,
            stage_b_only_reward=0.20,
            stage_a_b_partial_reward=0.60,
            reference_solution_reward=1.0,
        )
        validated = validate_composition_oracle(oracle, composition)
        gate = validate_d5_realization(
            {
                "capability_count": 5,
                "touched_file_count": 3,
                "semantic_dependency_depth": 4,
            },
            atomic_oracle={
                "unmodified_reward": 0.10,
                "single_fix_max_reward": 0.35,
                "pair_fix_max_reward": 0.60,
                "all_but_one_reward": 0.80,
                "reference_solution_reward": 1.0,
            },
            composition_oracle=oracle,
            composition_spec=composition,
        )

        self.assertTrue(validated["valid"])
        self.assertEqual(validated["coupling_margin"], 0.70)
        self.assertTrue(gate["valid"])


if __name__ == "__main__":
    unittest.main()
