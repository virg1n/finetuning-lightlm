import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def sha256(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as exc:
        return f"unavailable ({exc})"


def git_commit(path: Path) -> str:
    if not path.exists():
        return "missing"
    command = [
        "git",
        "-c",
        f"safe.directory={path.resolve()}",
        "-C",
        str(path),
        "rev-parse",
        "HEAD",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return f"unavailable ({result.stderr.strip() or result.stdout.strip()})"
    return result.stdout.strip()


def print_hashes(model_dir: Path, names: Iterable[str]) -> None:
    print("model_files:")
    for name in names:
        value = sha256(model_dir / name)
        if value:
            print(f"  {name}: {value}")


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt:
        return args.prompt

    from datasets import load_dataset

    dataset = load_dataset("openai_humaneval", split="test")
    return dataset[int(args.task_index)]["prompt"].strip()


def filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < values[..., [-1]], -float("inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        remove = sorted_probs.cumsum(dim=-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        logits = torch.full_like(logits, -float("inf")).scatter(-1, sorted_indices, sorted_logits)
    return logits


def print_top_tokens(tokenizer, logits: torch.Tensor, top_k: int, temperature: float, gen_top_k: int, top_p: float) -> None:
    print("raw_next_token_top:")
    values, indices = torch.topk(logits.float(), top_k)
    for rank, (value, idx) in enumerate(zip(values.tolist(), indices.tolist()), 1):
        print(f"  {rank:02d} id={idx:<6} logit={value: .6f} text={tokenizer.decode([idx])!r}")

    filtered = filter_logits(logits.float() / max(temperature, 1e-8), gen_top_k, top_p)
    probs = F.softmax(filtered, dim=-1)
    values, indices = torch.topk(probs, min(top_k, probs.size(-1)))
    print("sampling_distribution_top:")
    for rank, (value, idx) in enumerate(zip(values.tolist(), indices.tolist()), 1):
        print(f"  {rank:02d} id={idx:<6} prob={value: .8f} text={tokenizer.decode([idx])!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug direct HF generation on the first HumanEval prompt.")
    parser.add_argument("--model", default="cpt_runs/lightlm-python-cpt/eval_model_base_fresh")
    parser.add_argument("--harness-dir", default="../bigcode-evaluation-harness")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length-generation", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--do-sample", type=parse_bool, default=True)
    parser.add_argument("--top-tokens", type=int, default=20)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model)
    model_name = str(model_dir.resolve()) if model_dir.exists() else args.model
    harness_dir = Path(args.harness_dir)

    print("environment:")
    print(f"  python: {sys.version.split()[0]}")
    print(f"  torch: {torch.__version__}")
    print(f"  transformers: {package_version('transformers')}")
    print(f"  datasets: {package_version('datasets')}")
    print(f"  accelerate: {package_version('accelerate')}")
    print(f"  cuda_available: {torch.cuda.is_available()}")
    print(f"  cuda_version: {torch.version.cuda}")
    print(f"  harness_commit: {git_commit(harness_dir)}")

    if model_dir.exists():
        print_hashes(
            model_dir,
            ("model.safetensors", "pytorch_model.bin", "hf_lightlm.py", "model.py", "config.json", "tokenizer.json"),
        )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    prompt = load_prompt(args)
    print(f"prompt_chars: {len(prompt)}")
    print(f"prompt_repr: {prompt!r}")

    torch_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        local_files_only=args.local_files_only,
    ).to(args.device)
    model.eval()

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(args.device)
    print(f"prompt_tokens: {input_ids.size(1)}")

    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].detach().cpu()
    print_top_tokens(tokenizer, logits, args.top_tokens, args.temperature, args.top_k, args.top_p)

    gen_kwargs = {
        "max_length": args.max_length_generation,
        "do_sample": args.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        gen_kwargs.update(
            {
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
            }
        )

    with torch.no_grad():
        output_ids = model.generate(input_ids=input_ids, **gen_kwargs)

    completion = tokenizer.decode(output_ids[0, input_ids.size(1):], skip_special_tokens=False)
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    print("completion:")
    print(completion)
    print("full_generation_json:")
    print(json.dumps(full_text, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
