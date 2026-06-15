import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict

import torch
from safetensors.torch import save_file as save_safetensors
from transformers import AutoTokenizer

from model import ModelConfig, Transformer


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def load_torch_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    state = checkpoint.get("model") or checkpoint.get("model_state_dict") or checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"Expected checkpoint state dict, got {type(state).__name__}")

    normalized = {}
    prefixes = ("_orig_mod.", "module.", "lightlm.")
    for key, value in state.items():
        new_key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        normalized[new_key] = value
    return normalized


def build_tokenizer(config: Dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    special_tokens = config.get("additional_special_tokens", [])
    if special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    pad_token = config.get("pad_token")
    if pad_token:
        tokenizer.pad_token = pad_token
    return tokenizer


def build_model_config(config: Dict[str, Any], vocab_size: int, use_cache: bool) -> ModelConfig:
    model_cfg = dict(config["model"])
    model_cfg["vocab_size"] = vocab_size
    model_cfg["use_cache"] = use_cache
    return ModelConfig(**model_cfg)


def hf_config_payload(config: Dict[str, Any], tokenizer, use_cache: bool) -> Dict[str, Any]:
    model_cfg = dict(config["model"])
    return {
        "model_type": "lightlm",
        "architectures": ["LightLMForCausalLM"],
        "auto_map": {
            "AutoConfig": "hf_lightlm.LightLMConfig",
            "AutoModelForCausalLM": "hf_lightlm.LightLMForCausalLM",
        },
        "vocab_size": len(tokenizer),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "num_dims": model_cfg["num_dims"],
        "num_heads": model_cfg["num_heads"],
        "num_kv_heads": model_cfg["num_kv_heads"],
        "num_layers": model_cfg["num_layers"],
        "hidden_size": model_cfg["num_dims"],
        "num_hidden_layers": model_cfg["num_layers"],
        "num_attention_heads": model_cfg["num_heads"],
        "num_key_value_heads": model_cfg["num_kv_heads"],
        "intermediate_size": model_cfg["ffn_hidden_dims"],
        "max_position_embeddings": model_cfg["context_len"],
        "ffn_hidden_dims": model_cfg["ffn_hidden_dims"],
        "rmsnorm_eps": model_cfg["rmsnorm_eps"],
        "rope_theta": model_cfg["rope_theta"],
        "context_len": model_cfg["context_len"],
        "use_cache": use_cache,
        "use_flash": model_cfg["use_flash"],
        "use_moe": model_cfg["use_moe"],
        "moe_num_experts": model_cfg["moe_num_experts"],
        "moe_active_experts": model_cfg["moe_active_experts"],
        "moe_eps": model_cfg["moe_eps"],
        "moe_aux_loss_coef": model_cfg["moe_aux_loss_coef"],
        "moe_shared_experts": model_cfg["moe_shared_experts"],
        "use_lossfreebalance": model_cfg["use_lossfreebalance"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a native LightLM CPT checkpoint as a HF model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/eval_model_step_00037918")
    parser.add_argument("--config", default="lightlm_ffn_code_eval_config.json")
    parser.add_argument("--use-cache", type=parse_bool, default=True)
    parser.add_argument("--strict", type=parse_bool, default=True)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    config = read_json((script_dir / args.config).resolve())

    tokenizer = build_tokenizer(config)
    model = Transformer(build_model_config(config, len(tokenizer), args.use_cache))
    checkpoint = load_torch_checkpoint(checkpoint_path)
    state = normalize_state_dict(checkpoint)
    load_result = model.load_state_dict(state, strict=args.strict)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)

    hf_state = {
        f"lightlm.{name}": tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    save_safetensors(hf_state, str(output_dir / "model.safetensors"))
    write_json(output_dir / "config.json", hf_config_payload(config, tokenizer, args.use_cache))
    write_json(
        output_dir / "generation_config.json",
        {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "use_cache": args.use_cache,
        },
    )
    shutil.copy2(script_dir / "hf_lightlm.py", output_dir / "hf_lightlm.py")
    shutil.copy2(script_dir / "model.py", output_dir / "model.py")

    print(f"checkpoint: {checkpoint_path}")
    print(f"exported_model: {output_dir}")
    print(f"loaded_missing_keys: {load_result.missing_keys}")
    print(f"loaded_unexpected_keys: {load_result.unexpected_keys}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
