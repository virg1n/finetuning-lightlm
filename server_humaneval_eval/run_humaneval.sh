#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HF_REPO="${HF_REPO:-Virg1n/lightlm-ffn-code-cpt}"
HF_FILENAME="${HF_FILENAME:-checkpoints/cpt_step_00037918.pt}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/cpt_step_00037918.pt}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
TASKS="${TASKS:-humaneval}"
TASK_LABEL="${TASK_LABEL:-${TASKS//,/__}}"
MODEL_DIR="${MODEL_DIR:-$OUTPUT_DIR/eval_model_step_00037918}"
METRIC_OUTPUT_PATH="${METRIC_OUTPUT_PATH:-$OUTPUT_DIR/eval_results_${TASK_LABEL}_step_00037918.json}"
SAVE_GENERATIONS_PATH="${SAVE_GENERATIONS_PATH:-$OUTPUT_DIR/generations_step_00037918.json}"
STDOUT_LOG_FILE="${STDOUT_LOG_FILE:-$OUTPUT_DIR/eval_stdout_${TASK_LABEL}_step_00037918.log}"

HARNESS_DIR="${HARNESS_DIR:-bigcode-evaluation-harness}"
HARNESS_REPO="${HARNESS_REPO:-https://github.com/bigcode-project/bigcode-evaluation-harness.git}"
HARNESS_COMMIT="${HARNESS_COMMIT:-8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd}"

MAX_LENGTH_GENERATION="${MAX_LENGTH_GENERATION:-1024}"
PRECISION="${PRECISION:-bf16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
N_SAMPLES="${N_SAMPLES:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_K="${TOP_K:-0}"
TOP_P="${TOP_P:-1.0}"
DO_SAMPLE="${DO_SAMPLE:-False}"
SEED="${SEED:-0}"

mkdir -p "$OUTPUT_DIR"

if [ "${SKIP_EXPORT:-0}" != "1" ]; then
  if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Downloading checkpoint to $CHECKPOINT_PATH"
    python download_checkpoint.py \
      --repo-id "$HF_REPO" \
      --filename "$HF_FILENAME" \
      --local-dir .
  else
    echo "Using existing checkpoint: $CHECKPOINT_PATH"
  fi

  echo "Exporting native checkpoint to HF model directory: $MODEL_DIR"
  python export_checkpoint.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --output-dir "$MODEL_DIR" \
    --use-cache true
else
  echo "Skipping export; using existing model dir: $MODEL_DIR"
fi

if [ ! -f "$HARNESS_DIR/main.py" ]; then
  echo "Cloning BigCode evaluation harness into $HARNESS_DIR"
  git clone "$HARNESS_REPO" "$HARNESS_DIR"
fi

if [ -n "$HARNESS_COMMIT" ]; then
  git -C "$HARNESS_DIR" checkout "$HARNESS_COMMIT"
fi

export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1

echo "Running tasks '$TASKS' with model: $MODEL_DIR"
accelerate launch \
  --num_processes "$NUM_PROCESSES" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  "$HARNESS_DIR/main.py" \
  --model "$MODEL_DIR" \
  --trust_remote_code \
  --precision "$PRECISION" \
  --tasks "$TASKS" \
  --max_length_generation "$MAX_LENGTH_GENERATION" \
  --temperature "$TEMPERATURE" \
  --n_samples "$N_SAMPLES" \
  --batch_size "$BATCH_SIZE" \
  --metric_output_path "$METRIC_OUTPUT_PATH" \
  --top_k "$TOP_K" \
  --top_p "$TOP_P" \
  --do_sample "$DO_SAMPLE" \
  --seed "$SEED" \
  --save_generations \
  --save_generations_path "$SAVE_GENERATIONS_PATH" \
  --allow_code_execution \
  2>&1 | tee "$STDOUT_LOG_FILE"

echo "Metric JSON: $METRIC_OUTPUT_PATH"
for task in ${TASKS//,/ }; do
  echo "Generation JSON for $task: ${SAVE_GENERATIONS_PATH%.*}_${task}.json"
done
