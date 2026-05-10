import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cont_pretrain import build_tokenizer_and_model, get_dtype  # noqa: E402


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def top_tokens(tokenizer: Any, logits: torch.Tensor, k: int) -> str:
    values, indices = torch.topk(logits.float(), k)
    parts = []
    for value, idx in zip(values.tolist(), indices.tolist()):
        text = tokenizer.decode([idx]).replace("\n", "\\n")
        parts.append(f"{idx}:{value:.4f}:{text!r}")
    return ", ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare native LightLM logits with an exported HF LightLM model.")
    parser.add_argument("prompt", nargs="?", default=None, help="Prompt text. If omitted, stdin is used.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--hf-model", default="cpt_runs/lightlm-python-cpt/eval_model_base")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default=None)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    if not prompt:
        raise SystemExit("Prompt is empty.")

    config = copy.deepcopy(read_json((Path.cwd() / args.config).resolve()))
    config["model"]["resume_checkpoint"] = ""
    config["model"]["use_cache"] = False

    device = torch.device(args.device)
    dtype = args.dtype or ("bf16" if device.type == "cuda" else "fp32")
    torch_dtype = get_dtype(dtype) if dtype != "fp32" else torch.float32

    tokenizer, native_model, _ = build_tokenizer_and_model(config, device)
    if dtype != "fp32":
        native_model.to(dtype=torch_dtype)
    native_model.eval()

    hf_path = Path(args.hf_model)
    hf_model_name = str(hf_path.resolve()) if hf_path.exists() else args.hf_model
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).to(device)
    hf_model.eval()

    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    with torch.no_grad():
        native_logits, _, _ = native_model(input_ids)
        hf_logits = hf_model(input_ids=input_ids, use_cache=False).logits

    native_last = native_logits[:, -1, :].float().cpu()
    hf_last = hf_logits[:, -1, :].float().cpu()
    diff = (native_last - hf_last).abs()

    print(f"input_tokens: {input_ids.size(1)}")
    print(f"native_shape: {tuple(native_logits.shape)}")
    print(f"hf_shape: {tuple(hf_logits.shape)}")
    print(f"max_abs_diff_last: {diff.max().item():.8f}")
    print(f"mean_abs_diff_last: {diff.mean().item():.8f}")
    print(f"native_top{args.top_k}: {top_tokens(tokenizer, native_last[0], args.top_k)}")
    print(f"hf_top{args.top_k}: {top_tokens(tokenizer, hf_last[0], args.top_k)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
