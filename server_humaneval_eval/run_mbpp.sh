#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TASKS="${TASKS:-mbpp}"
export TASK_LABEL="${TASK_LABEL:-mbpp}"
export MAX_LENGTH_GENERATION="${MAX_LENGTH_GENERATION:-1024}"
export METRIC_OUTPUT_PATH="${METRIC_OUTPUT_PATH:-outputs/eval_results_mbpp_step_00037918.json}"
export SAVE_GENERATIONS_PATH="${SAVE_GENERATIONS_PATH:-outputs/generations_step_00037918.json}"
export STDOUT_LOG_FILE="${STDOUT_LOG_FILE:-outputs/eval_stdout_mbpp_step_00037918.log}"

exec "$SCRIPT_DIR/run_humaneval.sh"
