# SFT Phase for LightLM Code Alignment

This SFT phase starts from the current CPT best checkpoint:

```text
cpt_runs/lightlm-ffn-code-cpt/checkpoints/cpt_best.pt
```

It trains with a 4096-token context window and a 70/30 mixture:

- 70% instruction/dialogue/code-feedback data
- 30% raw completion data, with Python examples sometimes converted into docstring-driven completion examples

The SFT config is:

```text
config_sft_ffn.json
```

The trainer is:

```text
sft_train.py
```

## Required One-Time Setup

From the repo root on the 4x48GB GPU server:

```bash
export HF_TOKEN="hf_..."

# BigCode harness used during periodic HumanEval/MBPP eval.
if [ ! -f server_humaneval_eval/bigcode-evaluation-harness/main.py ]; then
  git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git \
    server_humaneval_eval/bigcode-evaluation-harness
  git -C server_humaneval_eval/bigcode-evaluation-harness checkout 8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd
fi
```

The config expects a local non-overlap synthetic HumanEval-format JSONL file:

```text
data/sft/humaneval_synthetic_nonoverlap.jsonl
```

Each row can be either prompt/completion:

```json
{"prompt": "Write a function ...", "completion": "def ..."}
```

or chat-style:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

If you do not have this file yet, set this source to `"enabled": false` in `config_sft_ffn.json`.

## Run SFT

Use `torchrun` or `accelerate`. With 4 GPUs:

```bash
torchrun --nproc_per_node=4 sft_train.py --config config_sft_ffn.json
```

Equivalent with Accelerate:

```bash
accelerate launch --num_processes 4 sft_train.py --config config_sft_ffn.json
```

Important defaults in `config_sft_ffn.json`:

- `model.context_len = 4096`
- `training.max_seq_len = 4096`
- `training.per_device_batch_size = 2`
- `training.gradient_accumulation_steps = 16`
- `training.max_steps = 2000`
- periodic eval every `250` SFT steps
- eval tasks: `humaneval,mbpp`
- primary metric: `humaneval/pass@1`
- tie-break metric: `mbpp/pass@1`

For a smoke test:

```bash
cp config_sft_ffn.json config_sft_smoke.json
# edit config_sft_smoke.json:
# training.max_steps = 2
# evaluation.bigcode_harness.enabled = false
# evaluation.bigcode_harness.initial_eval_enabled = false
# evaluation.bigcode_harness.final_eval_enabled = false
# evaluation.bigcode_harness.interval_steps = 0
torchrun --nproc_per_node=4 sft_train.py --config config_sft_smoke.json
```

## Outputs

SFT outputs go to:

```text
sft_runs/lightlm-ffn-code-sft/
```

Checkpoints:

```text
sft_runs/lightlm-ffn-code-sft/checkpoints/sft_step_XXXXXXXX.pt
sft_runs/lightlm-ffn-code-sft/checkpoints/sft_best.pt
sft_runs/lightlm-ffn-code-sft/checkpoints/sft_best_humaneval.pt
sft_runs/lightlm-ffn-code-sft/checkpoints/sft_best_mbpp.pt
```

`sft_best.pt` and `sft_best_humaneval.pt` track the best HumanEval score. If a later checkpoint ties the best HumanEval score, `sft_best_mbpp.pt` is updated only when MBPP improves among that HumanEval-tied set.

Logs and eval outputs:

```text
sft_runs/lightlm-ffn-code-sft/logs/train.jsonl
sft_runs/lightlm-ffn-code-sft/logs/eval_stdout_step_*.log
sft_runs/lightlm-ffn-code-sft/eval_results_step_*.json
sft_runs/lightlm-ffn-code-sft/generations_step_*_humaneval.json
sft_runs/lightlm-ffn-code-sft/generations_step_*_mbpp.json
```

HumanEval and MBPP scoring execute generated Python code. Run on a server/container where executing benchmark code is acceptable.
