#!/usr/bin/env bash
set -euo pipefail

# Post-RL hosted evals for a trained checkpoint/model id.
# Change ENV_ID if you pushed the environment under another owner or namespace.

if [[ $# -lt 1 || -z "${1:-}" ]]; then
  echo "usage: $0 <trained-checkpoint-id>" >&2
  exit 2
fi

MODEL_ID="$1"
ENV_ID="${ENV_ID:-ydnyshhh/synthetic-workspace-gym}"
MAX_EXAMPLES="${MAX_EXAMPLES:-40}"
MAX_TURNS="${MAX_TURNS:-8}"

prime eval run "${ENV_ID}" \
  --hosted \
  -m "${MODEL_ID}" \
  -n "${MAX_EXAMPLES}" \
  -r 1 \
  -a "{\"split\":\"train\",\"family\":\"script_repair\",\"difficulties\":\"1,2,3\",\"max_examples\":${MAX_EXAMPLES},\"max_turns\":${MAX_TURNS},\"reward_mode\":\"score\",\"sample_strategy\":\"balanced\",\"shuffle\":true,\"shuffle_seed\":42}" \
  --follow

prime eval run "${ENV_ID}" \
  --hosted \
  -m "${MODEL_ID}" \
  -n "${MAX_EXAMPLES}" \
  -r 1 \
  -a "{\"split\":\"validation\",\"family\":\"script_repair\",\"difficulties\":\"2,3,4\",\"max_examples\":${MAX_EXAMPLES},\"max_turns\":${MAX_TURNS},\"reward_mode\":\"score\",\"sample_strategy\":\"balanced\",\"shuffle\":true,\"shuffle_seed\":42}" \
  --follow

prime eval run "${ENV_ID}" \
  --hosted \
  -m "${MODEL_ID}" \
  -n "${MAX_EXAMPLES}" \
  -r 1 \
  -a "{\"split\":\"test\",\"family\":\"script_repair\",\"difficulties\":\"3,4,5\",\"max_examples\":${MAX_EXAMPLES},\"max_turns\":${MAX_TURNS},\"reward_mode\":\"score\",\"sample_strategy\":\"balanced\",\"shuffle\":true,\"shuffle_seed\":42}" \
  --follow

prime eval run "${ENV_ID}" \
  --hosted \
  -m "${MODEL_ID}" \
  -n "${MAX_EXAMPLES}" \
  -r 1 \
  -a "{\"split\":\"heldout\",\"family\":\"script_repair\",\"difficulties\":\"3,4,5\",\"max_examples\":${MAX_EXAMPLES},\"max_turns\":${MAX_TURNS},\"reward_mode\":\"score\",\"sample_strategy\":\"balanced\",\"shuffle\":true,\"shuffle_seed\":42}" \
  --follow
