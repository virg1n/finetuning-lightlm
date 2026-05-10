import argparse
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clean_distributed_env() -> Dict[str, str]:
    env = os.environ.copy()
    for key in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_ERROR_FILE",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_USE_AGENT_STORE",
    ):
        env.pop(key, None)
    return env


def latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(checkpoint_dir.glob("cpt_step_*.pt"))
    return checkpoints[-1] if checkpoints else None


def scalar_items(mapping: Dict[str, Any]) -> Iterable[Tuple[str, Any]]:
    for key, value in sorted(mapping.items()):
        if isinstance(value, (str, int, float, bool)) or value is None:
            yield key, value


def print_results(path: Path) -> None:
    if not path.exists():
        print(f"results: missing ({path})")
        return
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"results: unreadable ({exc})")
        return

    results = payload.get("results", payload) if isinstance(payload, dict) else payload
    if not isinstance(results, dict):
        print(f"results: {type(results).__name__}")
        return

    print("results:")
    for task, task_result in sorted(results.items()):
        print(f"  {task}:")
        if not isinstance(task_result, dict):
            print(f"    {task_result}")
            continue
        printed = False
        for key, value in scalar_items(task_result):
            print(f"    {key}: {value}")
            printed = True
        if not printed:
            print(f"    keys: {', '.join(sorted(task_result.keys()))}")


def get_command_arg(command: List[str], name: str, default: str) -> str:
    try:
        idx = command.index(name)
    except ValueError:
        return default
    if idx + 1 >= len(command):
        return default
    return str(command[idx + 1])


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return cleaned[:120] or "eval"


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def export_checkpoint_model(config: Dict[str, Any], checkpoint: str, device_name: str) -> str:
    import torch

    from cont_pretrain import build_tokenizer_and_model, export_eval_model

    export_config = copy.deepcopy(config)
    if checkpoint != "base":
        export_config["model"]["resume_checkpoint"] = checkpoint
    else:
        export_config["model"]["resume_checkpoint"] = ""

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--export-device cuda was requested, but CUDA is not available.")
    device = torch.device(device_name)
    tokenizer, model, _ = build_tokenizer_and_model(export_config, device)
    model.eval()
    with torch.no_grad():
        return export_eval_model(export_config, tokenizer, model)


def resolve_eval_model(args: argparse.Namespace, config: Dict[str, Any]) -> str:
    if args.model:
        path = Path(args.model)
        return str(path.resolve()) if path.exists() else args.model

    output_dir = resolve_path(config["paths"]["output_dir"], REPO_ROOT)
    exported_model = output_dir / "eval_model"
    checkpoint_dir = resolve_path(config["paths"]["checkpoint_dir"], REPO_ROOT)

    checkpoint = args.checkpoint
    if checkpoint == "auto":
        latest = latest_checkpoint(checkpoint_dir)
        if latest is not None:
            checkpoint = str(latest)
        elif (exported_model / "config.json").exists():
            return str(exported_model.resolve())
        else:
            raise RuntimeError(
                "No checkpoint or exported eval model found. Pass --model, "
                "--checkpoint path/to/cpt_step_*.pt, or --checkpoint base."
            )
    elif checkpoint == "latest":
        latest = latest_checkpoint(checkpoint_dir)
        if latest is None:
            raise RuntimeError(f"No cpt_step_*.pt files found in {checkpoint_dir}")
        checkpoint = str(latest)
    elif checkpoint != "base":
        checkpoint_path = resolve_path(checkpoint, REPO_ROOT)
        if not checkpoint_path.exists():
            raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = str(checkpoint_path)

    print(f"exporting eval model from checkpoint={checkpoint}")
    return export_checkpoint_model(config, checkpoint, args.export_device)


def build_command(
    args: argparse.Namespace,
    config: Dict[str, Any],
    eval_model: str,
    metric_output_path: Path,
) -> Tuple[List[str], Path]:
    eval_cfg = config.get("evaluation", {}).get("bigcode_harness", {})
    configured_command = list(eval_cfg.get("command", []))

    tasks = args.tasks or get_command_arg(configured_command, "--tasks", "humaneval,mbpp")
    precision = args.precision or get_command_arg(configured_command, "--precision", "bf16")
    max_length = args.max_length_generation or get_command_arg(configured_command, "--max_length_generation", "1024")
    temperature = args.temperature or get_command_arg(configured_command, "--temperature", "0.2")
    n_samples = args.n_samples or get_command_arg(configured_command, "--n_samples", "1")
    batch_size = args.batch_size or get_command_arg(configured_command, "--batch_size", "1")

    command = [
        args.accelerate_bin,
        "launch",
        "--num_processes",
        str(args.num_processes),
        "--main_process_port",
        str(args.main_process_port),
        "main.py",
        "--model",
        eval_model,
        "--trust_remote_code",
        "--precision",
        str(precision),
        "--tasks",
        str(tasks),
        "--max_length_generation",
        str(max_length),
        "--temperature",
        str(temperature),
        "--n_samples",
        str(n_samples),
        "--batch_size",
        str(batch_size),
        "--metric_output_path",
        str(metric_output_path),
    ]
    if args.save_generations:
        command.append("--save_generations")
        if args.save_generations_path:
            command.extend(["--save_generations_path", str(resolve_path(args.save_generations_path, Path.cwd()))])
    if args.generation_only:
        command.append("--generation_only")
    if args.load_generations_path:
        command.extend(["--load_generations_path", str(resolve_path(args.load_generations_path, Path.cwd()))])
    if args.check_references:
        command.append("--check_references")
    if args.limit:
        command.extend(["--limit", str(args.limit)])
    if not args.no_allow_code_execution:
        command.append("--allow_code_execution")

    harness_dir_value = args.harness_dir or eval_cfg.get("cwd") or "../bigcode-evaluation-harness"
    harness_dir = resolve_path(str(harness_dir_value), REPO_ROOT)
    return command, harness_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a LightLM checkpoint/export with bigcode-evaluation-harness."
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--model",
        default=None,
        help="Existing HF model directory or Hub model id. If set, checkpoint export is skipped.",
    )
    parser.add_argument(
        "--checkpoint",
        default="auto",
        help="'auto', 'latest', 'base', or a local cpt_step_*.pt path. Ignored when --model is set.",
    )
    parser.add_argument("--export-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--harness-dir", default=None)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument(
        "--max-length-generation",
        default="1024",
        help="Total prompt+completion length for the harness. 1024 avoids zero-room MBPP prompts.",
    )
    parser.add_argument("--temperature", default=None)
    parser.add_argument("--n-samples", default=None)
    parser.add_argument(
        "--batch-size",
        default="1",
        help="Eval batch size. Defaults to 1 because hf_lightlm.py currently ignores attention_mask.",
    )
    parser.add_argument("--num-processes", default=1, type=int)
    parser.add_argument("--main-process-port", default=0)
    parser.add_argument("--accelerate-bin", default="accelerate")
    parser.add_argument("--metric-output-path", default=None)
    parser.add_argument("--stdout-log-file", default=None)
    parser.add_argument("--save-generations", action="store_true")
    parser.add_argument("--save-generations-path", default=None)
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--load-generations-path", default=None)
    parser.add_argument("--check-references", action="store_true")
    parser.add_argument("--limit", default=None, help="Optional harness limit for quick smoke tests.")
    parser.add_argument("--no-allow-code-execution", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = resolve_path(args.config, Path.cwd())
    config = read_json(config_path)
    output_dir = resolve_path(config["paths"]["output_dir"], REPO_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_model = resolve_eval_model(args, config)

    eval_cfg = config.get("evaluation", {}).get("bigcode_harness", {})
    configured_command = list(eval_cfg.get("command", []))
    args.tasks = args.tasks or get_command_arg(configured_command, "--tasks", "humaneval,mbpp")
    task_name = safe_name(args.tasks)
    metric_output_path = (
        resolve_path(args.metric_output_path, Path.cwd())
        if args.metric_output_path
        else output_dir / f"eval_results_manual_{task_name}.json"
    )
    stdout_log_file = (
        resolve_path(args.stdout_log_file, Path.cwd())
        if args.stdout_log_file
        else output_dir / "logs" / f"eval_stdout_manual_{task_name}.log"
    )
    metric_output_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_log_file.parent.mkdir(parents=True, exist_ok=True)

    command, harness_dir = build_command(args, config, eval_model, metric_output_path)
    if not (harness_dir / "main.py").exists():
        raise RuntimeError(
            f"bigcode-evaluation-harness main.py not found in {harness_dir}. "
            "Clone it there or pass --harness-dir."
        )

    print(f"eval_model: {eval_model}")
    print(f"harness_dir: {harness_dir}")
    print(f"metric_output_path: {metric_output_path}")
    print(f"stdout_log_file: {stdout_log_file}")
    print("command:")
    print("  " + " ".join(command))

    if args.dry_run:
        return 0

    with stdout_log_file.open("wb") as f:
        result = subprocess.run(
            command,
            cwd=str(harness_dir),
            env=clean_distributed_env(),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )

    print(f"returncode: {result.returncode}")
    print_results(metric_output_path)
    if result.returncode != 0:
        print(f"eval failed; inspect log: {stdout_log_file}")
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
