import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cont_pretrain import build_tokenizer_and_model, get_dtype  # noqa: E402


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(checkpoint_dir.glob("cpt_step_*.pt"))
    return checkpoints[-1] if checkpoints else None


def best_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    path = checkpoint_dir / "cpt_best.pt"
    return path if path.exists() else None


def checkpoint_config_hints(current_config: str, current_checkpoint_dir: Path) -> List[str]:
    hints: List[str] = []
    for config_path in sorted(REPO_ROOT.glob("config*.json")):
        try:
            candidate = read_json(config_path)
            candidate_dir = resolve_path(candidate["paths"]["checkpoint_dir"], REPO_ROOT)
        except (KeyError, OSError, json.JSONDecodeError):
            continue
        if candidate_dir == current_checkpoint_dir:
            continue
        latest = latest_checkpoint(candidate_dir)
        best = best_checkpoint(candidate_dir)
        if latest is None and best is None:
            continue
        checkpoint_name = latest.name if latest is not None else best.name
        hints.append(
            f"{config_path.name}: checkpoint_dir={candidate_dir} newest={checkpoint_name}"
        )
    if hints:
        return [
            f"Current config is {current_config!r}; other configs with checkpoints:",
            *hints,
        ]
    return [f"Current config is {current_config!r}."]


def no_checkpoint_error(config_name: str, checkpoint_dir: Path) -> RuntimeError:
    lines = [
        f"No cpt_step_*.pt files found in {checkpoint_dir}",
        *checkpoint_config_hints(config_name, checkpoint_dir),
        "Pass the matching config, for example --config config_ffn.json, "
        "or pass an explicit checkpoint path.",
    ]
    best = best_checkpoint(checkpoint_dir)
    if best is not None:
        lines.append(f"This directory has {best.name}; use --checkpoint best to prompt it.")
    return RuntimeError("\n".join(lines))


def resolve_checkpoint(checkpoint: str, config: Dict[str, Any], config_name: str) -> str:
    if checkpoint == "base":
        return ""
    checkpoint_dir = resolve_path(config["paths"]["checkpoint_dir"], REPO_ROOT)
    if checkpoint == "latest":
        latest = latest_checkpoint(checkpoint_dir)
        if latest is None:
            raise no_checkpoint_error(config_name, checkpoint_dir)
        return str(latest)
    if checkpoint == "best":
        best = best_checkpoint(checkpoint_dir)
        if best is None:
            raise RuntimeError(f"No cpt_best.pt file found in {checkpoint_dir}")
        return str(best)
    checkpoint_path = resolve_path(checkpoint, REPO_ROOT)
    if not checkpoint_path.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")
    return str(checkpoint_path)


def filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        remove = sorted_probs.cumsum(dim=-1) > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        logits = torch.full_like(logits, -float("inf")).scatter(1, sorted_indices, sorted_logits)

    return logits


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int],
    temperature: float,
    top_k: int,
    top_p: float,
    use_cache: bool,
) -> torch.Tensor:
    model.eval()
    past_key_values = None

    for _ in range(max_new_tokens):
        if use_cache:
            model_input = input_ids if past_key_values is None else input_ids[:, -1:]
            logits, _, _, past_key_values = model(
                model_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
        else:
            logits, _, _ = model(input_ids)

        next_logits = logits[:, -1, :]
        if temperature <= 0:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            next_logits = filter_logits(next_logits / temperature, top_k=top_k, top_p=top_p)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat((input_ids, next_token), dim=1)
        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            break

    return input_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate text from a configured LightLM checkpoint.")
    parser.add_argument("prompt", nargs="?", default=None, help="Prompt text. If omitted, stdin is used.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--checkpoint",
        default="base",
        help="'base', 'latest', 'best', or a local checkpoint path.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--full-text", action="store_true", help="Print prompt plus completion.")
    args = parser.parse_args()

    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    if not prompt:
        raise SystemExit("Prompt is empty.")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config_path = resolve_path(args.config, Path.cwd())
    config = copy.deepcopy(read_json(config_path))
    config["model"]["resume_checkpoint"] = resolve_checkpoint(args.checkpoint, config, str(args.config))
    config["model"]["use_cache"] = True

    device = torch.device(args.device)
    tokenizer, model, _ = build_tokenizer_and_model(config, device)
    dtype = args.dtype or ("bf16" if device.type == "cuda" else "fp32")
    if dtype != "fp32":
        model.to(dtype=get_dtype(dtype))

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    context_len = int(config["model"]["context_len"])
    if input_ids.size(1) > context_len:
        input_ids = input_ids[:, -context_len:]

    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        use_cache=not args.no_cache,
    )

    if args.full_text:
        print(tokenizer.decode(output_ids[0], skip_special_tokens=False))
    else:
        completion_ids = output_ids[0, input_ids.size(1) :]
        print(tokenizer.decode(completion_ids, skip_special_tokens=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
