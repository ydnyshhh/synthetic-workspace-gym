from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from synthetic_workspace_gym.generators.registry import list_generators

from .policy import scenario_pool_for_family
from .schemas import VALID_SPLITS, SplitManifest


def validate_split_manifest(manifest: SplitManifest) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    counts = Counter(assignment.split for assignment in manifest.assignments)
    known_families = set(list_generators())
    seen_task_ids: set[str] = set()
    seen_tuples: dict[tuple[str, str | None, int, int], str] = {}
    scenarios_by_split: dict[tuple[str, str], set[str | None]] = defaultdict(set)
    seeds_by_key: dict[tuple[str, str, str | None, int], set[int]] = defaultdict(set)

    for assignment in manifest.assignments:
        if assignment.split not in VALID_SPLITS:
            errors.append(f"Invalid split name: {assignment.split}")
        if assignment.task_id in seen_task_ids:
            errors.append(f"Duplicate task_id: {assignment.task_id}")
        seen_task_ids.add(assignment.task_id)

        exact = (assignment.family, assignment.scenario, assignment.difficulty, assignment.seed)
        previous_split = seen_tuples.get(exact)
        if previous_split is not None and previous_split != assignment.split:
            errors.append(f"Duplicate environment tuple across splits: {exact}")
        seen_tuples[exact] = assignment.split

        if assignment.family not in known_families:
            errors.append(f"Unknown family: {assignment.family}")
        elif assignment.scenario is not None and assignment.scenario not in scenario_pool_for_family(assignment.family):
            errors.append(f"Unknown scenario for {assignment.family}: {assignment.scenario}")
        if not 1 <= assignment.difficulty <= 5:
            errors.append(f"Invalid difficulty for {assignment.task_id}: {assignment.difficulty}")
        if assignment.seed < 0:
            errors.append(f"Invalid seed for {assignment.task_id}: {assignment.seed}")

        scenarios_by_split[(assignment.family, assignment.split)].add(assignment.scenario)
        seeds_by_key[(assignment.split, assignment.family, assignment.scenario, assignment.difficulty)].add(assignment.seed)

    for family in known_families:
        heldout = scenarios_by_split.get((family, "heldout"), set())
        in_distribution = set()
        for split in ("train", "validation", "test"):
            in_distribution.update(scenarios_by_split.get((family, split), set()))
        overlap = heldout.intersection(in_distribution)
        if overlap:
            errors.append(f"Heldout scenarios appear in train/validation/test for {family}: {sorted(str(item) for item in overlap)}")

    for key, train_seeds in list(seeds_by_key.items()):
        split, family, scenario, difficulty = key
        if split != "train":
            continue
        validation_overlap = train_seeds.intersection(seeds_by_key.get(("validation", family, scenario, difficulty), set()))
        test_overlap = train_seeds.intersection(seeds_by_key.get(("test", family, scenario, difficulty), set()))
        if validation_overlap:
            errors.append(f"Validation overlaps train seeds for {family}/{scenario}/d{difficulty}: {sorted(validation_overlap)}")
        if test_overlap:
            errors.append(f"Test overlaps train seeds for {family}/{scenario}/d{difficulty}: {sorted(test_overlap)}")

    for key, validation_seeds in list(seeds_by_key.items()):
        split, family, scenario, difficulty = key
        if split != "validation":
            continue
        test_overlap = validation_seeds.intersection(seeds_by_key.get(("test", family, scenario, difficulty), set()))
        if test_overlap:
            errors.append(f"Test overlaps validation seeds for {family}/{scenario}/d{difficulty}: {sorted(test_overlap)}")

    for split in VALID_SPLITS:
        if counts.get(split, 0) == 0:
            warnings.append(f"Split has no assignments: {split}")

    return {
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors,
        "warnings": warnings,
        "counts": {split: counts.get(split, 0) for split in ("train", "validation", "test", "heldout")},
    }
