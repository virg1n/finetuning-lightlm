# LightLM Code Benchmark Server Evaluation

This folder is a self-contained bundle for evaluating `Virg1n/lightlm-ffn-code-cpt` checkpoint `cpt_step_00037918.pt` on HumanEval and MBPP on a Linux GPU server.

It exports the native training checkpoint to a local Hugging Face model directory, then runs `bigcode-evaluation-harness` with code execution enabled.

## Files

- `model.py`, `hf_lightlm.py`: LightLM model code required by `trust_remote_code`.
- `lightlm_ffn_code_eval_config.json`: architecture/tokenizer settings for this checkpoint.
- `download_checkpoint.py`: downloads `checkpoints/cpt_step_00037918.pt` from Hugging Face.
- `export_checkpoint.py`: exports the native checkpoint to HF `model.safetensors`.
- `run_humaneval.sh`: one-command Linux HumanEval runner. It also accepts `TASKS=mbpp`.
- `run_mbpp.sh`: one-command Linux MBPP runner.
- `score_humaneval_generations.py`: optional scorer for already-saved HumanEval or MBPP generations.
- `requirements.txt`: Python packages except CUDA PyTorch.

## Server Setup

Run these commands on the Linux GPU server from inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install CUDA PyTorch for your server. For a CUDA 12.6 driver/wheel:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch
```

Then install the rest:

```bash
python -m pip install -r requirements.txt
```

Set your Hugging Face token if the checkpoint repo is private:

```bash
export HF_TOKEN="hf_..."
```

Run HumanEval:

```bash
chmod +x run_humaneval.sh
./run_humaneval.sh
```

Run MBPP:

```bash
chmod +x run_humaneval.sh run_mbpp.sh
./run_mbpp.sh
```

Run both tasks in one harness call:

```bash
TASKS=humaneval,mbpp TASK_LABEL=humaneval_mbpp ./run_humaneval.sh
```

By default this writes:

- `outputs/eval_results_humaneval_step_00037918.json`
- `outputs/generations_step_00037918_humaneval.json`
- `outputs/eval_stdout_humaneval_step_00037918.log`
- `outputs/eval_model_step_00037918/`

MBPP writes:

- `outputs/eval_results_mbpp_step_00037918.json`
- `outputs/generations_step_00037918_mbpp.json`
- `outputs/eval_stdout_mbpp_step_00037918.log`

The expected local result from this machine was:

```json
{
  "results": {
    "humaneval": {
      "pass@1": 0.06097560975609756
    }
  }
}
```

## Manual Checkpoint Upload

If the server cannot download from Hugging Face, upload the checkpoint yourself and place it here:

```text
server_humaneval_eval/checkpoints/cpt_step_00037918.pt
```

Then run:

```bash
SKIP_EXPORT=0 ./run_humaneval.sh
```

## Useful Overrides

You can override defaults with environment variables:

```bash
MAX_LENGTH_GENERATION=1024 BATCH_SIZE=1 NUM_PROCESSES=1 ./run_humaneval.sh
```

Use `MAX_LENGTH_GENERATION=1024` unless you intentionally want longer generations. This model has `context_len=1536`; 1024 leaves enough completion room for HumanEval and MBPP while avoiding very slow no-stop generations.

To use an already exported model:

```bash
SKIP_EXPORT=1 MODEL_DIR=outputs/eval_model_step_00037918 ./run_humaneval.sh
```

To use a pre-cloned harness:

```bash
HARNESS_DIR=/path/to/bigcode-evaluation-harness ./run_humaneval.sh
```

To skip the harness commit pin:

```bash
HARNESS_COMMIT= ./run_humaneval.sh
```

## Score Saved Generations Only

If you already have a generations file:

```bash
export HF_ALLOW_CODE_EVAL=1
python score_humaneval_generations.py \
  --task humaneval \
  --generations-path outputs/generations_step_00037918_humaneval.json \
  --output-path outputs/eval_results_from_generations.json \
  --harness-dir bigcode-evaluation-harness \
  --allow-code-execution
```

For MBPP:

```bash
export HF_ALLOW_CODE_EVAL=1
python score_humaneval_generations.py \
  --task mbpp \
  --generations-path outputs/generations_step_00037918_mbpp.json \
  --output-path outputs/eval_results_mbpp_from_generations.json \
  --harness-dir bigcode-evaluation-harness \
  --allow-code-execution
```

HumanEval and MBPP scoring execute model-generated Python code. Run them only on a server/container where you are comfortable executing untrusted benchmark code.
