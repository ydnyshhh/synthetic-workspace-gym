#!/usr/bin/env bash
set -euo pipefail

# Build the deterministic script_repair-only split manifest for the first SWG RL
# improvement experiment. This uses SWG's built-in default split policy:
# train d1-3, validation d2-4, test d3-5, and scenario-heldout d3-5.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MANIFEST="splits/swg_rl_exp1_script_repair_manifest.json"
ASSIGNMENTS="splits/swg_rl_exp1_script_repair_assignments.jsonl"
EXPORT_DIR="prime_exports/swg_rl_exp1_script_repair"
OVERWRITE=0

if [[ "${1:-}" == "--overwrite" ]]; then
  OVERWRITE=1
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--overwrite]" >&2
  exit 2
fi

mkdir -p splits prime_exports

if [[ "${OVERWRITE}" -ne 1 ]]; then
  if [[ -e "${MANIFEST}" || -e "${ASSIGNMENTS}" || -e "${EXPORT_DIR}" ]]; then
    echo "Refusing to overwrite existing split artifacts." >&2
    echo "Re-run with --overwrite to refresh the manifest or export pack." >&2
    exit 1
  fi
fi

uv run swg splits build \
  --output "${MANIFEST}" \
  --assignments-output "${ASSIGNMENTS}" \
  --families script_repair \
  --shuffle \
  --shuffle-seed 42

uv run swg splits validate \
  --manifest "${MANIFEST}"

uv run swg splits stats \
  --manifest "${MANIFEST}"

EXPORT_ARGS=(
  swg prime export-splits
  --split-manifest "${MANIFEST}"
  --output-dir "${EXPORT_DIR}"
  --export-name swg_rl_exp1_script_repair
)

if [[ "${OVERWRITE}" -eq 1 ]]; then
  EXPORT_ARGS+=(--overwrite)
fi

uv run "${EXPORT_ARGS[@]}"
