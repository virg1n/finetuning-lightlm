import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cont_pretrain import build_tokenizer_and_model, get_dtype  # noqa: E402


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
        help="'base', 'latest', or a local cpt_step_*.pt path.",
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

    config = copy.deepcopy(read_json((Path.cwd() / args.config).resolve()))
    config["model"]["resume_checkpoint"] = "" if args.checkpoint == "base" else args.checkpoint
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
