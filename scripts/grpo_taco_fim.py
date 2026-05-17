import argparse
import inspect
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from accelerate import PartialState
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
import torch
import trl
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


REPO_ROOT = Path(__file__).resolve().parents[1]


def disable_python_int_digit_limit() -> None:
    setter = getattr(sys, "set_int_max_str_digits", None)
    if setter is None:
        return
    try:
        setter(0)
    except ValueError:
        pass


disable_python_int_digit_limit()


RUNNER_CODE = r'''
import ast
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

try:
    import resource
except ImportError:
    resource = None


def disable_python_int_digit_limit():
    setter = getattr(sys, "set_int_max_str_digits", None)
    if setter is None:
        return
    try:
        setter(0)
    except ValueError:
        pass


disable_python_int_digit_limit()


def apply_limits(memory_limit_mb):
    if resource is None:
        return
    if memory_limit_mb and memory_limit_mb > 0:
        limit = int(memory_limit_mb) * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except (OSError, ValueError):
            pass


def normalize_text(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    return " ".join(value.strip().split())


def parse_scalar(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return text
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            pass
    return value


def close_float(a, b):
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
    except Exception:
        return False


def compare_values(actual, expected):
    expected = parse_scalar(expected)
    if actual == expected:
        return True
    if isinstance(actual, float) or isinstance(expected, float):
        return close_float(actual, expected)
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            return False
        return all(compare_values(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            return False
        return all(compare_values(actual[key], expected[key]) for key in actual)
    return normalize_text(actual) == normalize_text(expected)


def compare_stdout(stdout, expected):
    return normalize_text(stdout) == normalize_text(expected)


def get_callable(namespace, fn_name):
    fn = namespace.get(fn_name)
    if callable(fn):
        return fn
    solution_cls = namespace.get("Solution")
    if solution_cls is not None:
        try:
            solution = solution_cls()
            fn = getattr(solution, fn_name, None)
            if callable(fn):
                return fn
        except Exception:
            return None
    return None


def run_function_tests(candidate_path, tests):
    fn_name = tests.get("fn_name")
    inputs = tests.get("inputs") or []
    outputs = tests.get("outputs") or []
    total = min(len(inputs), len(outputs))
    if not fn_name or total == 0:
        return {"passed": 0, "total": total, "error": "missing fn_name or tests"}

    namespace = {"__name__": "__candidate__"}
    try:
        source = candidate_path.read_text(encoding="utf-8")
        stdin = sys.stdin
        stdout = sys.stdout
        stderr = sys.stderr
        sys.stdin = io.StringIO("")
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            exec(compile(source, str(candidate_path), "exec"), namespace)
        finally:
            sys.stdin = stdin
            sys.stdout = stdout
            sys.stderr = stderr
    except BaseException:
        return {"passed": 0, "total": total, "error": traceback.format_exc(limit=4)}

    fn = get_callable(namespace, fn_name)
    if fn is None:
        return {"passed": 0, "total": total, "error": f"callable {fn_name!r} not found"}

    passed = 0
    for raw_args, expected in zip(inputs[:total], outputs[:total]):
        try:
            args = raw_args if isinstance(raw_args, (list, tuple)) else [raw_args]
            actual = fn(*args)
            passed += int(compare_values(actual, expected))
        except BaseException:
            pass
    return {"passed": passed, "total": total, "error": ""}


def run_stdio_tests(candidate_path, tests, deadline):
    inputs = tests.get("inputs") or []
    outputs = tests.get("outputs") or []
    total = min(len(inputs), len(outputs))
    passed = 0
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0"}
    for stdin_value, expected in zip(inputs[:total], outputs[:total]):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(candidate_path)],
                input=str(stdin_value),
                text=True,
                cwd=str(candidate_path.parent),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=remaining,
            )
            if proc.returncode == 0 and compare_stdout(proc.stdout, expected):
                passed += 1
        except subprocess.TimeoutExpired:
            break
        except BaseException:
            pass
    return {"passed": passed, "total": total, "error": ""}


def main():
    candidate_path = Path(sys.argv[1])
    tests_path = Path(sys.argv[2])
    timeout_seconds = float(sys.argv[3])
    memory_limit_mb = int(sys.argv[4])
    apply_limits(memory_limit_mb)

    tests = json.loads(tests_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + timeout_seconds
    if tests.get("fn_name"):
        result = run_function_tests(candidate_path, tests)
    else:
        result = run_stdio_tests(candidate_path, tests, deadline)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FIM GRPO training on TACO with subprocess unit-test rewards."
    )
    parser.add_argument("--model", required=True, help="HF model id or local exported model directory.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path/id. Defaults to --model.")
    parser.add_argument("--dataset-name", default="BAAI/TACO")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="rl_runs/lightlm-taco-grpo-fim")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-prompt-length", type=int, default=1536)
    parser.add_argument("--max-completion-length", type=int, default=512)

    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument(
        "--steps-per-generation",
        type=int,
        default=1,
        help=(
            "How many local train batches to generate at once. TRL defaults this to "
            "gradient_accumulation_steps, which can OOM during generation for long code prompts."
        ),
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1338)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--report-to", default="none")

    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.01)
    parser.add_argument("--num-iterations", type=int, default=2)
    parser.add_argument("--loss-type", choices=("grpo", "bnpo", "dr_grpo"), default="grpo")
    parser.add_argument("--ref-update-steps", type=int, default=50)
    parser.add_argument("--ref-mixup-alpha", type=float, default=1.0)
    parser.add_argument("--scale-rewards", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--fim-prefix", default="<fim_prefix>")
    parser.add_argument("--fim-suffix", default="<fim_suffix>")
    parser.add_argument("--fim-middle", default="<fim_middle>")
    parser.add_argument("--reward-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--memory-limit-mb", type=int, default=2048)
    return parser.parse_args()


def parse_input_output(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except ValueError:
            return None
    elif isinstance(value, dict):
        payload = value
    else:
        return None
    inputs = payload.get("inputs")
    outputs = payload.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return None
    total = min(len(inputs), len(outputs))
    if total <= 0:
        return None
    return {
        "inputs": inputs[:total],
        "outputs": outputs[:total],
        "fn_name": payload.get("fn_name") or "",
    }


def comment_problem(question: str, max_chars: int = 6000) -> str:
    question = (question or "").strip()
    if len(question) > max_chars:
        question = question[:max_chars].rstrip()
    lines = question.splitlines() or [""]
    return "\n".join("# " + line.rstrip() for line in lines).rstrip() + "\n"


def strip_trailing_stub(starter_code: str) -> str:
    code = (starter_code or "").rstrip()
    if not code:
        return ""
    lines = code.splitlines()
    for idx in range(len(lines) - 1, -1, -1):
        stripped = lines[idx].strip()
        if not stripped:
            continue
        if stripped in {"pass", "..."}:
            indent = re.match(r"^\s*", lines[idx]).group(0)
            return "\n".join(lines[:idx] + [indent])
        return code
    return code


def build_fim_parts(example: Dict[str, Any]) -> Tuple[str, str]:
    problem = comment_problem(example.get("question") or "")
    starter = strip_trailing_stub(example.get("starter_code") or "")
    if starter:
        prefix = problem + "\n" + starter
        if not prefix.endswith("\n"):
            prefix += "\n"
    else:
        prefix = problem + "\n"
    suffix = ""
    return prefix, suffix


def base_split_name(split: str) -> str:
    return split.split("[", 1)[0].strip() or split


def load_taco_arrow_dataset(args: argparse.Namespace) -> Dataset:
    split_name = base_split_name(args.split)
    prefix = f"{split_name}/"
    files = list_repo_files(
        repo_id=args.dataset_name,
        repo_type="dataset",
        revision=args.dataset_revision,
    )
    arrow_files = sorted(
        filename
        for filename in files
        if filename.startswith(prefix) and filename.endswith(".arrow")
    )
    if not arrow_files:
        raise RuntimeError(
            f"No Arrow files found for split {split_name!r} in dataset {args.dataset_name!r}."
        )
    local_files = [
        hf_hub_download(
            repo_id=args.dataset_name,
            filename=filename,
            repo_type="dataset",
            revision=args.dataset_revision,
        )
        for filename in arrow_files
    ]
    return load_dataset("arrow", data_files={split_name: local_files}, split=args.split)


def load_taco_dataset(args: argparse.Namespace) -> Dataset:
    try:
        return load_dataset(
            args.dataset_name,
            split=args.split,
            revision=args.dataset_revision,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "Dataset scripts are no longer supported" not in message and "TACO.py" not in message:
            raise
        print(
            "datasets cannot execute BAAI/TACO.py in this environment; "
            "loading TACO Arrow shards directly from the Hub.",
            flush=True,
        )
        return load_taco_arrow_dataset(args)


def prepare_taco_dataset(args: argparse.Namespace) -> Dataset:
    dataset = load_taco_dataset(args)
    if args.max_train_samples and args.max_train_samples > 0:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))

    def convert(example: Dict[str, Any]) -> Dict[str, Any]:
        tests = parse_input_output(example.get("input_output"))
        prefix, suffix = build_fim_parts(example)
        prompt = f"{args.fim_prefix}{prefix}{args.fim_suffix}{suffix}{args.fim_middle}"
        return {
            "prompt": prompt,
            "completion_prefix": prefix,
            "completion_suffix": suffix,
            "tests_json": json.dumps(tests or {}, ensure_ascii=False),
            "has_tests": tests is not None,
            "task_name": example.get("name") or "",
        }

    mapped = dataset.map(
        convert,
        remove_columns=dataset.column_names,
        num_proc=max(1, args.num_proc),
        desc="Building FIM TACO prompts",
    )
    mapped = mapped.filter(lambda row: bool(row["has_tests"]), desc="Keeping examples with tests")
    mapped = mapped.remove_columns(["has_tests"])
    if len(mapped) == 0:
        raise RuntimeError("No TACO examples with usable input_output tests were found.")
    return mapped


def kill_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        return
    except OSError:
        process.kill()


def execute_candidate(
    code: str,
    tests: Dict[str, Any],
    timeout_seconds: float,
    memory_limit_mb: int,
) -> float:
    total = min(len(tests.get("inputs") or []), len(tests.get("outputs") or []))
    if total <= 0:
        return 0.0
    with tempfile.TemporaryDirectory(prefix="taco_reward_") as tmp:
        tmp_path = Path(tmp)
        candidate_path = tmp_path / "candidate.py"
        tests_path = tmp_path / "tests.json"
        runner_path = tmp_path / "runner.py"
        candidate_path.write_text(code, encoding="utf-8")
        tests_path.write_text(json.dumps(tests, ensure_ascii=False), encoding="utf-8")
        runner_path.write_text(RUNNER_CODE, encoding="utf-8")
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONHASHSEED": "0",
        }
        popen_kwargs: Dict[str, Any] = {}
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        command = [
            sys.executable,
            "-I",
            str(runner_path),
            str(candidate_path),
            str(tests_path),
            str(float(timeout_seconds)),
            str(int(memory_limit_mb)),
        ]
        process = subprocess.Popen(
            command,
            cwd=str(tmp_path),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **popen_kwargs,
        )
        try:
            stdout, _stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            kill_process(process)
            process.communicate()
            return 0.0
        if process.returncode != 0:
            return 0.0
    try:
        result = json.loads(stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return 0.0
    passed = int(result.get("passed") or 0)
    measured_total = int(result.get("total") or total)
    if measured_total <= 0:
        return 0.0
    return max(0.0, min(1.0, passed / measured_total))


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(completion)


def align_reward_column(values: Any, target_len: int, default: Any) -> List[Any]:
    if values is None:
        return [default] * target_len
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return [values] * target_len
    values = list(values)
    if len(values) == target_len:
        return values
    if not values:
        return [default] * target_len
    if target_len % len(values) == 0:
        repeats = target_len // len(values)
        return [value for value in values for _ in range(repeats)]
    return (values + [values[-1]] * target_len)[:target_len]


class TacoUnitTestReward:
    def __init__(self, timeout_seconds: float, memory_limit_mb: int):
        self.__name__ = "taco_unit_test_reward"
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb

    def __call__(self, completions: Sequence[Any], **kwargs: Any) -> List[float]:
        prefixes = align_reward_column(kwargs.get("completion_prefix"), len(completions), "")
        suffixes = align_reward_column(kwargs.get("completion_suffix"), len(completions), "")
        tests_json = align_reward_column(kwargs.get("tests_json"), len(completions), "{}")
        rewards: List[float] = []
        for completion, prefix, suffix, test_payload in zip(completions, prefixes, suffixes, tests_json):
            try:
                tests = json.loads(test_payload) if isinstance(test_payload, str) else dict(test_payload)
            except (TypeError, json.JSONDecodeError):
                rewards.append(0.0)
                continue
            code = str(prefix) + completion_text(completion) + str(suffix)
            reward = execute_candidate(
                code=code,
                tests=tests,
                timeout_seconds=self.timeout_seconds,
                memory_limit_mb=self.memory_limit_mb,
            )
            rewards.append(float(reward))
        return rewards


def validate_batch_geometry(args: argparse.Namespace) -> None:
    effective = args.per_device_train_batch_size * max(1, args.steps_per_generation)
    if effective % args.group_size != 0:
        raise ValueError(
            "per_device_train_batch_size * steps_per_generation must be divisible "
            f"by group_size. Got {effective} and group_size={args.group_size}."
        )


def report_to_value(value: str) -> Any:
    if value.lower() in {"none", "null", "false", "off", ""}:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def set_process_cuda_device() -> None:
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None or not torch.cuda.is_available():
        return
    try:
        torch.cuda.set_device(int(local_rank))
    except (TypeError, ValueError, RuntimeError):
        pass


def supported_init_kwargs(cls: Any) -> Optional[set]:
    signature = inspect.signature(cls)
    params = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return None
    return set(params)


def trainer_supports_ref_model() -> bool:
    supported = supported_init_kwargs(GRPOTrainer.__init__)
    return supported is None or "ref_model" in supported


def patch_trl_causal_lm_loader(default_kwargs: Dict[str, Any]) -> None:
    """Make TRL's internal ref-model loader handle custom CausalLM code."""

    def create_causal_lm_from_path(
        model_id: str,
        architecture: Any = None,
        **kwargs: Any,
    ) -> Any:
        load_kwargs = dict(default_kwargs)
        load_kwargs.update(kwargs)
        dtype = load_kwargs.get("dtype")
        if isinstance(dtype, str) and dtype in {"bfloat16", "float16", "float32"}:
            load_kwargs["dtype"] = getattr(torch, dtype)
        load_kwargs.setdefault("trust_remote_code", True)
        model_cls = architecture or AutoModelForCausalLM
        model = model_cls.from_pretrained(model_id, **load_kwargs)
        patch_lightlm_transformers_compat(model)
        return model

    for module_name in ("trl.trainer.grpo_trainer", "trl.trainer.utils"):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "create_model_from_path"):
            setattr(module, "create_model_from_path", create_causal_lm_from_path)


def patch_lightlm_transformers_compat(model: Any) -> None:
    config = getattr(model, "config", None)
    if config is not None and getattr(config, "model_type", None) == "lightlm":
        aliases = {
            "num_hidden_layers": "num_layers",
            "hidden_size": "num_dims",
            "num_attention_heads": "num_heads",
            "num_key_value_heads": "num_kv_heads",
            "intermediate_size": "ffn_hidden_dims",
            "max_position_embeddings": "context_len",
        }
        for target, source in aliases.items():
            if not hasattr(config, target) and hasattr(config, source):
                setattr(config, target, getattr(config, source))

    # LightLM's exported HF wrapper returns legacy tuple KV caches. Newer
    # Transformers otherwise tries to create DynamicCache objects for generate().
    if config is not None and getattr(config, "model_type", None) == "lightlm":
        model._supports_cache_class = False
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            generation_config.cache_implementation = None


def build_model_load_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
    }
    if args.bf16:
        kwargs["dtype"] = torch.bfloat16
    return kwargs


def filter_supported_kwargs(
    cls: Any,
    kwargs: Dict[str, Any],
    required: Sequence[str],
    state: PartialState,
    owner_name: str,
) -> Dict[str, Any]:
    supported = supported_init_kwargs(cls)
    if supported is None:
        return kwargs
    missing_required = [name for name in required if name not in supported]
    if missing_required:
        version = getattr(trl, "__version__", "unknown")
        raise RuntimeError(
            f"Installed TRL {version} has an incompatible {owner_name}; "
            f"missing required argument(s): {', '.join(missing_required)}. "
            "Upgrade with: pip install -U 'trl>=0.17.0'"
        )
    unsupported = sorted(name for name in kwargs if name not in supported)
    if unsupported and state.is_main_process:
        version = getattr(trl, "__version__", "unknown")
        print(
            f"Installed TRL {version} does not support optional {owner_name} "
            f"argument(s), ignoring: {', '.join(unsupported)}",
            flush=True,
        )
    return {name: value for name, value in kwargs.items() if name in supported}


def build_grpo_config(
    args: argparse.Namespace,
    state: PartialState,
) -> GRPOConfig:
    kwargs = {
        "output_dir": args.output_dir,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "steps_per_generation": args.steps_per_generation,
        "gradient_checkpointing": args.gradient_checkpointing,
        "bf16": args.bf16,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "report_to": report_to_value(args.report_to),
        "seed": args.seed,
        "remove_unused_columns": False,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.group_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": None if args.top_k <= 0 else args.top_k,
        "beta": args.kl_beta,
        "epsilon": args.clip_epsilon,
        "num_iterations": args.num_iterations,
        "scale_rewards": args.scale_rewards,
        "loss_type": args.loss_type,
        "sync_ref_model": True,
        "ref_model_sync_steps": args.ref_update_steps,
        "ref_model_mixup_alpha": args.ref_mixup_alpha,
        "log_completions": True,
    }
    required = (
        "max_completion_length",
        "num_generations",
        "temperature",
        "beta",
        "epsilon",
        "sync_ref_model",
        "ref_model_sync_steps",
    )
    return GRPOConfig(
        **filter_supported_kwargs(GRPOConfig, kwargs, required, state, "GRPOConfig")
    )


def build_grpo_trainer(
    model: Any,
    ref_model: Optional[Any],
    reward_func: TacoUnitTestReward,
    training_args: GRPOConfig,
    train_dataset: Dataset,
    tokenizer: Any,
) -> GRPOTrainer:
    kwargs: Dict[str, Any] = {
        "model": model,
        "reward_funcs": reward_func,
        "args": training_args,
        "train_dataset": train_dataset,
    }
    supported = supported_init_kwargs(GRPOTrainer.__init__)
    if ref_model is not None and (supported is None or "ref_model" in supported):
        kwargs["ref_model"] = ref_model
    if supported is None or "processing_class" in supported:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in supported:
        kwargs["tokenizer"] = tokenizer
    return GRPOTrainer(**kwargs)


def main() -> int:
    set_process_cuda_device()
    args = parse_args()
    validate_batch_geometry(args)
    distributed_state = PartialState()
    with distributed_state.main_process_first():
        train_dataset = prepare_taco_dataset(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_load_kwargs = build_model_load_kwargs(args)
    patch_trl_causal_lm_loader(model_load_kwargs)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_load_kwargs,
    )
    patch_lightlm_transformers_compat(model)
    ref_model: Optional[Any] = None
    if trainer_supports_ref_model():
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            **model_load_kwargs,
        )
        patch_lightlm_transformers_compat(ref_model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad_(False)
    training_args = build_grpo_config(args, distributed_state)

    trainer = build_grpo_trainer(
        model=model,
        ref_model=ref_model,
        reward_func=TacoUnitTestReward(
            timeout_seconds=args.reward_timeout_seconds,
            memory_limit_mb=args.memory_limit_mb,
        ),
        training_args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
