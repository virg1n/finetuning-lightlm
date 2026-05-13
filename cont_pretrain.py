import argparse
import gzip
import hashlib
import html
import json
import math
import os
import queue
import random
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from urllib.parse import unquote
from urllib.request import urlopen

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset, load_dataset_builder
from datatrove.pipeline.tokens.tokenizer import TokenizedFile
from datatrove.utils.dataset import DatatroveFolderDataset
from huggingface_hub import hf_hub_download, list_repo_files
try:
    from huggingface_hub import get_token as get_hf_hub_token
except ImportError:
    get_hf_hub_token = None
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as checkpoint_fn
from transformers import AutoTokenizer

from model import ModelConfig, Transformer

try:
    from safetensors.torch import load_file as load_safetensors
    from safetensors.torch import save_file as save_safetensors
except ImportError:
    load_safetensors = None
    save_safetensors = None


MODES = {"standard": 0, "fim_psm": 1, "fim_spm": 2}
MODE_NAMES = {v: k for k, v in MODES.items()}

_DOCSTRING_AFTER_DEF = re.compile(
    r'^(?P<indent>[ \t]*)(?:async[ \t]+)?def[ \t]+\w+[ \t]*\([\s\S]*?\)[ \t]*'
    r'(?:->[\s\S]*?)?:[ \t]*\n'
    r'(?P=indent)[ \t]+(?P<q>"""|\'\'\')[\s\S]*?(?P=q)',
    re.MULTILINE,
)
CODE_CATEGORIES = {"python_code", "other_code"}

STOPWORDS = {
    "the",
    "and",
    "that",
    "with",
    "for",
    "this",
    "from",
    "are",
    "you",
    "not",
    "have",
    "can",
    "will",
    "use",
    "using",
    "when",
    "where",
    "what",
    "why",
    "how",
}

KNOWN_GATED_HF_DATASETS = {
    "bigcode/starcoderdata",
    "bigcode/the-stack-dedup",
    "bigcode/the-stack-v2-train-smol-ids",
}

_DIST_RUN_ID = "single"
_RANK0_STAGE_COUNTER = 0
_JSONL_MAX_BYTES = int(os.environ.get("LIGHTLM_JSONL_MAX_BYTES", str(1024 * 1024)))

if os.name == "nt":
    import msvcrt
else:
    import fcntl


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def configure_logging(config: Dict[str, Any]) -> None:
    global _JSONL_MAX_BYTES
    _JSONL_MAX_BYTES = int(config.get("logging", {}).get("jsonl_max_bytes", _JSONL_MAX_BYTES))


def truncate_file_tail(path: Path, max_bytes: int) -> None:
    if max_bytes <= 0 or not path.exists() or path.stat().st_size <= max_bytes:
        return
    with path.open("rb") as f:
        f.seek(-max_bytes, os.SEEK_END)
        data = f.read()
    newline = data.find(b"\n")
    if newline != -1:
        data = data[newline + 1 :]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
    tmp.replace(path)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    truncate_file_tail(path, _JSONL_MAX_BYTES)


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        f.seek(0)
        if os.name == "nt":
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        else:
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)


def manifest_lock_path(config: Dict[str, Any]) -> Path:
    manifest_path = Path(config["paths"]["manifest_path"])
    return manifest_path.with_suffix(manifest_path.suffix + ".lock")


def shard_build_lock_path(config: Dict[str, Any]) -> Path:
    return Path(config["paths"]["output_dir"]) / ".shard_build.lock"


@contextmanager
def locked_manifest(config: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    manifest_path = Path(config["paths"]["manifest_path"])
    with file_lock(manifest_lock_path(config)):
        manifest = load_manifest(manifest_path)
        yield manifest
        manifest["updated_at"] = utc_now()
        write_json(manifest_path, manifest)


def log_rank0(message: str) -> None:
    if rank0():
        print(message, flush=True)


def source_requires_hf_access(source: Dict[str, Any]) -> bool:
    return bool(source.get("requires_hf_access")) or source.get("dataset") in KNOWN_GATED_HF_DATASETS


def get_hf_token(config: Dict[str, Any], source: Optional[Dict[str, Any]] = None) -> Optional[str]:
    if source and source.get("hf_token"):
        return source["hf_token"]
    token = (
        config["data"].get("hf_token")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if token:
        return token
    if get_hf_hub_token is not None:
        return get_hf_hub_token()
    return None


def has_aws_credentials() -> bool:
    return any(
        os.environ.get(name)
        for name in (
            "AWS_ACCESS_KEY_ID",
            "AWS_PROFILE",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        )
    )


def hf_dataset_access_kwargs(source: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    config_name = source.get("config_name") or source.get("name")
    if config_name:
        kwargs["name"] = config_name
    if source.get("data_dir"):
        kwargs["data_dir"] = source["data_dir"]
    if source.get("data_files"):
        kwargs["data_files"] = source["data_files"]
    if source.get("trust_remote_code"):
        kwargs["trust_remote_code"] = True
    hf_token = get_hf_token(config, source)
    if hf_token:
        kwargs["token"] = hf_token
    return kwargs


def verify_hf_source_access(source: Dict[str, Any], config: Dict[str, Any]) -> None:
    try:
        load_dataset_builder(source["dataset"], **hf_dataset_access_kwargs(source, config))
    except Exception as exc:
        if source_requires_hf_access(source):
            hint = (
                "Accept the dataset terms on Hugging Face for this account, run `huggingface-cli login` again "
                "or `export HF_TOKEN=...`, or set this source to enabled=false in config.json."
            )
        else:
            hint = (
                "Check the dataset name, config_name, data_dir, data_files, and split in config.json, "
                "or set this source to enabled=false."
            )
        raise RuntimeError(
            f"Source {source['id']} ({source['dataset']}) could not be opened. {hint}"
        ) from exc


def validate_source_access(config: Dict[str, Any], verify_hf_access: bool = False) -> None:
    for source in enabled_sources(config):
        if source_requires_hf_access(source) and not get_hf_token(config, source):
            raise RuntimeError(
                f"Source {source['id']} ({source['dataset']}) requires a Hugging Face token and accepted dataset terms. "
                "Run `huggingface-cli login` or `export HF_TOKEN=...`, accept the dataset terms on HF, "
                "or set this source to enabled=false in config.json."
            )
        if verify_hf_access:
            verify_hf_source_access(source, config)
        if source.get("requires_swh_s3_access") and source.get("swh_unsigned") is False and not has_aws_credentials():
            raise RuntimeError(
                f"Source {source['id']} is configured for signed SWH S3 access but no AWS credentials were found. "
                "Set AWS credentials, set swh_unsigned=true, set swh_transport=https, or disable this source."
            )


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank0() -> bool:
    return not is_dist() or dist.get_rank() == 0


def barrier() -> None:
    if is_dist():
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def setup_distributed(config: Dict[str, Any]) -> Tuple[torch.device, int, int, int]:
    use_dist = config["training"].get("distributed", True) and "RANK" in os.environ
    if use_dist:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, rank, local_rank, world_size


def cleanup_distributed() -> None:
    if is_dist():
        dist.destroy_process_group()


def suppress_known_compile_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"\s*Online softmax is disabled on the fly since Inductor decides to\s*split the reduction\..*",
        category=UserWarning,
        module=r"torch\._inductor\.lowering",
    )


def initialize_dist_run_id() -> None:
    global _DIST_RUN_ID
    if not is_dist():
        _DIST_RUN_ID = "single"
        return
    value = f"{int(time.time())}_{os.getpid()}" if rank0() else None
    values = [value]
    device = None
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    dist.broadcast_object_list(values, src=0, device=device)
    _DIST_RUN_ID = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(values[0]))


def rank0_stage_status_path(config: Dict[str, Any], name: str, counter: int) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")[:80] or "stage"
    return (
        Path(config["paths"]["output_dir"])
        / ".dist_stages"
        / _DIST_RUN_ID
        / f"{counter:05d}_{safe_name}.json"
    )


def run_rank0_stage(config: Dict[str, Any], name: str, fn):
    global _RANK0_STAGE_COUNTER
    counter = _RANK0_STAGE_COUNTER
    _RANK0_STAGE_COUNTER += 1
    if not is_dist():
        return fn()

    status_path = rank0_stage_status_path(config, name, counter)
    if rank0():
        status_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(status_path, {"status": "running", "stage": name, "updated_at": utc_now()})
        try:
            result = fn()
        except Exception as exc:
            write_json(
                status_path,
                {
                    "status": "error",
                    "stage": name,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "updated_at": utc_now(),
                },
            )
            raise
        write_json(status_path, {"status": "ok", "stage": name, "updated_at": utc_now()})
        barrier()
        return result

    poll_seconds = float(config["training"].get("rank0_stage_poll_seconds", 5.0))
    timeout_seconds = float(config["training"].get("rank0_stage_timeout_seconds", 0.0))
    started = time.monotonic()
    while True:
        if status_path.exists():
            try:
                status = read_json(status_path)
            except json.JSONDecodeError:
                status = {"status": "running"}
            if status.get("status") == "ok":
                barrier()
                return None
            if status.get("status") == "error":
                raise RuntimeError(
                    f"Rank 0 failed during {name}: "
                    f"{status.get('error_type', 'Error')}: {status.get('message', '')}"
                )
        if timeout_seconds > 0 and time.monotonic() - started > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for rank 0 stage {name}.")
        time.sleep(max(poll_seconds, 0.1))


def set_seed(seed: int, rank: int) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_dtype(name: str) -> torch.dtype:
    aliases = {"bf16": "bfloat16", "fp16": "float16"}
    return getattr(torch, aliases.get(name, name))


def infer_token_size(vocab_size: int) -> int:
    return 2 if vocab_size < 65535 else 4


def stable_int_hash(value: str) -> int:
    digest = hashlib.blake2s(value.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def source_epoch_seed(config: Dict[str, Any], source_id: str, epoch: int) -> int:
    return (int(config["seed"]) + stable_int_hash(source_id) + int(epoch) * 1_000_003) % (2**32)


def resolve_shard_target_tokens(config: Dict[str, Any], token_size: int) -> int:
    explicit = config["tokens"].get("shard_target_tokens")
    if explicit:
        return int(explicit)
    budget_bytes = float(config["tokens"]["disk_budget_gb"]) * (1024**3)
    safety = float(config["tokens"].get("disk_safety_fraction", 0.9))
    source_bytes_per_token = float(config["tokens"].get("avg_source_bytes_per_token", 4.0))
    index_overhead = 0.05
    return int((budget_bytes * safety) / (source_bytes_per_token + token_size + index_overhead))


def model_config_from_json(config: Dict[str, Any], vocab_size: int) -> ModelConfig:
    m = config["model"]
    return ModelConfig(
        vocab_size=vocab_size,
        num_dims=int(m["num_dims"]),
        num_heads=int(m["num_heads"]),
        num_kv_heads=int(m["num_kv_heads"]),
        num_layers=int(m["num_layers"]),
        ffn_hidden_dims=int(m["ffn_hidden_dims"]),
        rmsnorm_eps=float(m["rmsnorm_eps"]),
        rope_theta=float(m["rope_theta"]),
        context_len=int(m["context_len"]),
        use_cache=bool(m["use_cache"]),
        use_flash=bool(m["use_flash"]),
        use_moe=bool(m["use_moe"]),
        moe_num_experts=int(m["moe_num_experts"]),
        moe_active_experts=int(m["moe_active_experts"]),
        moe_eps=float(m["moe_eps"]),
        moe_aux_loss_coef=float(m["moe_aux_loss_coef"]),
        moe_shared_experts=int(m["moe_shared_experts"]),
        use_lossfreebalance=bool(m["use_lossfreebalance"]),
    )


def strip_state_prefixes(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state.items():
        for prefix in ("module.", "_orig_mod.", "lightlm."):
            if k.startswith(prefix):
                k = k[len(prefix) :]
        out[k] = v
    return out


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(checkpoint_dir.glob("cpt_step_*.pt"))
    return checkpoints[-1] if checkpoints else None


def split_hf_location(path_or_repo: str, subfolder: str = "") -> Tuple[str, str, Optional[str]]:
    marker = "huggingface.co/"
    if marker not in path_or_repo:
        return path_or_repo, subfolder, None

    tail = unquote(path_or_repo.split(marker, 1)[1].strip("/"))
    parts = tail.split("/")
    if len(parts) < 2:
        return path_or_repo, subfolder, None

    repo_id = "/".join(parts[:2])
    if len(parts) >= 4 and parts[2] in {"tree", "blob"}:
        path_inside_repo = "/".join(parts[4:])
        if parts[2] == "tree":
            return repo_id, path_inside_repo, None
        repo_path = PurePosixPath(path_inside_repo)
        return repo_id, str(repo_path.parent), repo_path.name
    return repo_id, subfolder, None


def checkpoint_candidates(repo_id: str, subfolder: str = "") -> List[str]:
    suffixes = (".safetensors", ".bin", ".pt", ".pth")
    files = list_repo_files(repo_id=repo_id, repo_type="model")
    prefix = subfolder.strip("/")
    if prefix:
        files = [file for file in files if file.startswith(prefix + "/")]
    candidates = [file for file in files if file.endswith(suffixes)]

    preferred_names = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.pt",
        "checkpoint.pt",
    )
    return sorted(
        candidates,
        key=lambda file: (
            Path(file).name not in preferred_names,
            not file.endswith(".safetensors"),
            file,
        ),
    )


def load_checkpoint_file(path_or_repo: str, subfolder: str = "") -> Dict[str, Any]:
    path_or_repo, subfolder, exact_filename = split_hf_location(path_or_repo, subfolder)
    path = Path(path_or_repo)
    if path.exists() and subfolder:
        path = path / subfolder
    if path.exists():
        if path.is_dir():
            for name in ("model.safetensors", "pytorch_model.bin"):
                candidate = path / name
                if candidate.exists():
                    path = candidate
                    break
        if path.suffix == ".safetensors":
            if load_safetensors is None:
                raise RuntimeError("Install safetensors to load .safetensors checkpoints.")
            return {"model": load_safetensors(str(path))}
        return torch.load(path, map_location="cpu", weights_only=False)

    filenames = []
    if exact_filename:
        filenames.append("/".join(part for part in [subfolder.strip("/"), exact_filename] if part))
    else:
        try:
            filenames.extend(checkpoint_candidates(path_or_repo, subfolder))
        except Exception:
            filenames.extend(
                "/".join(part for part in [subfolder.strip("/"), filename] if part)
                for filename in ("model.safetensors", "pytorch_model.bin", "model.pt", "checkpoint.pt")
            )

    for filename in filenames:
        try:
            downloaded = hf_hub_download(repo_id=path_or_repo, filename=filename)
            if filename.endswith(".safetensors"):
                if load_safetensors is None:
                    raise RuntimeError("Install safetensors to load .safetensors checkpoints.")
                return {"model": load_safetensors(downloaded)}
            return torch.load(downloaded, map_location="cpu", weights_only=False)
        except Exception:
            continue
    location = f"{path_or_repo}/{subfolder}" if subfolder else path_or_repo
    raise FileNotFoundError(
        f"Could not find a checkpoint in {location!r}. "
        f"Tried: {filenames or 'no checkpoint-like files found'}"
    )


def load_state_into_model(model: Transformer, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    raw_state = checkpoint.get("model", checkpoint)
    state = strip_state_prefixes(raw_state)
    model_state = model.state_dict()
    compatible = {}
    partial = {}

    for name, tensor in state.items():
        if name not in model_state or not torch.is_tensor(tensor):
            continue
        target = model_state[name]
        if tuple(tensor.shape) == tuple(target.shape):
            compatible[name] = tensor
        elif (
            name in {"tokens_embedding.weight", "ll_head.weight"}
            and tensor.ndim == 2
            and target.ndim == 2
            and tensor.shape[1] == target.shape[1]
        ):
            rows = min(tensor.shape[0], target.shape[0])
            target[:rows].copy_(tensor[:rows])
            partial[name] = {"loaded_rows": rows, "target_rows": target.shape[0]}

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {
        "compatible_tensors": len(compatible),
        "partial_tensors": partial,
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


def init_special_embeddings(
    model: Transformer,
    tokenizer: AutoTokenizer,
    old_vocab_size: int,
    special_cfg: Dict[str, Any],
) -> None:
    seed_words = special_cfg.get("init_seed_words", {})
    weight = model.tokens_embedding.weight.data
    base_mean = weight[:old_vocab_size].mean(dim=0)

    for token in special_cfg["additional_special_tokens"]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < old_vocab_size:
            continue
        related_ids: List[int] = []
        for word in seed_words.get(token, []):
            related_ids.extend(
                i for i in tokenizer.encode(word, add_special_tokens=False) if 0 <= i < old_vocab_size
            )
        if related_ids:
            weight[token_id].copy_(weight[related_ids].mean(dim=0))
        else:
            weight[token_id].copy_(base_mean)


def build_tokenizer(config: Dict[str, Any]) -> Tuple[Any, int]:
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["tokenizer_name"], use_fast=True)
    old_vocab_size = len(tokenizer)
    special_tokens = config["special_tokens"]["additional_special_tokens"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    tokenizer.pad_token = config["special_tokens"]["fim_pad"]
    return tokenizer, old_vocab_size


def build_tokenizer_and_model(config: Dict[str, Any], device: torch.device) -> Tuple[Any, Transformer, int]:
    tokenizer, old_vocab_size = build_tokenizer(config)
    model_cfg = model_config_from_json(config, len(tokenizer))
    model = Transformer(model_cfg)

    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    resume_setting = config["model"].get("resume_checkpoint", "latest")
    resume_path: Optional[Path] = None
    if resume_setting == "latest":
        resume_path = find_latest_checkpoint(checkpoint_dir)
    elif resume_setting:
        candidate = Path(resume_setting)
        if candidate.exists():
            resume_path = candidate

    if resume_path:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        info = load_state_into_model(model, checkpoint)
        if rank0():
            print(f"Resumed CPT checkpoint: {resume_path}")
            print(f"Loaded tensors: {info['compatible_tensors']}, partial: {info['partial_tensors']}")
    else:
        base_checkpoint = load_checkpoint_file(
            config["model"]["base_checkpoint"],
            config["model"].get("base_checkpoint_subfolder", ""),
        )
        info = load_state_into_model(model, base_checkpoint)
        init_special_embeddings(model, tokenizer, old_vocab_size, config["special_tokens"])
        if rank0():
            print(f"Loaded base checkpoint: {config['model']['base_checkpoint']}")
            print(f"Loaded tensors: {info['compatible_tensors']}, partial: {info['partial_tensors']}")

    model.to(device)
    return tokenizer, model, old_vocab_size


def default_manifest() -> Dict[str, Any]:
    return {
        "version": 1,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "tokens_prepared": 0,
        "tokens_trained": 0,
        "global_step": 0,
        "next_eval_tokens": 0,
        "initial_eval_done": False,
        "source_offsets": {},
        "source_epochs": {},
        "shards": [],
    }


def load_manifest(path: Path) -> Dict[str, Any]:
    if path.exists():
        return read_json(path)
    return default_manifest()


def enabled_sources(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [s for s in config["data"]["sources"] if s.get("enabled", True)]


def source_targets(config: Dict[str, Any], shard_tokens: int) -> Dict[str, int]:
    mix = config["data"]["mix"]
    sources = enabled_sources(config)
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for src in sources:
        by_category[src["category"]].append(src)

    targets: Dict[str, int] = {}
    for category, category_fraction in mix.items():
        category_sources = by_category.get(category, [])
        if not category_sources:
            continue
        total_weight = sum(float(s.get("weight", 1.0)) for s in category_sources)
        category_tokens = int(shard_tokens * float(category_fraction))
        for src in category_sources:
            targets[src["id"]] = int(category_tokens * float(src.get("weight", 1.0)) / total_weight)
    return targets


def should_reset_zero_token_offset(manifest: Dict[str, Any], source: Dict[str, Any]) -> bool:
    if not source.get("reset_offset_if_zero_tokens", False):
        return False
    source_id = source["id"]
    stats = [
        shard.get("source_stats", {}).get(source_id)
        for shard in manifest.get("shards", [])
        if shard.get("source_stats", {}).get(source_id) is not None
    ]
    if not stats:
        return False
    tokens = sum(int(stat.get("actual_tokens", 0)) for stat in stats)
    examples = sum(int(stat.get("examples_consumed", 0)) for stat in stats)
    return tokens == 0 and examples > 0


def load_hf_stream(source: Dict[str, Any], config: Dict[str, Any], seed: int, skip: int):
    config_name = source.get("config_name") or source.get("name")
    config_text = f" config={config_name}" if config_name else ""
    log_rank0(
        f"Opening source {source['id']} "
        f"dataset={source['dataset']}{config_text} split={source.get('split', 'train')} skip={skip:,}"
    )
    kwargs = {
        "split": source.get("split", "train"),
        "streaming": True,
    }
    kwargs.update(hf_dataset_access_kwargs(source, config))

    try:
        dataset = load_dataset(source["dataset"], **kwargs)
    except Exception as exc:
        if source_requires_hf_access(source) or type(exc).__name__ == "DatasetNotFoundError":
            raise RuntimeError(
                f"Could not open source {source['id']} ({source['dataset']}). "
                "If this is a gated dataset, accept the dataset terms on Hugging Face and "
                "provide a token with `huggingface-cli login` or `export HF_TOKEN=...`; "
                "otherwise disable this source in config.json."
            ) from exc
        raise RuntimeError(
            f"Could not open source {source['id']} ({source['dataset']}). "
            "Check dataset config_name, data_dir, data_files, and split in config.json, "
            "or disable this source."
        ) from exc
    shuffle_buffer = int(source.get("shuffle_buffer", config["data"].get("stream_shuffle_buffer", 0)))
    if shuffle_buffer > 0:
        log_rank0(f"Shuffling source {source['id']} with buffer_size={shuffle_buffer:,}")
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)
    if skip > 0:
        dataset = dataset.skip(skip)
    return dataset


def stream_retry_settings(config: Dict[str, Any], source: Dict[str, Any]) -> Tuple[int, float, float]:
    data_config = config.get("data", {})
    max_retries = int(source.get("stream_max_retries", data_config.get("stream_max_retries", 20)))
    base_sleep = float(source.get("stream_retry_initial_seconds", data_config.get("stream_retry_initial_seconds", 5.0)))
    max_sleep = float(source.get("stream_retry_max_seconds", data_config.get("stream_retry_max_seconds", 120.0)))
    return max_retries, base_sleep, max_sleep


def retry_sleep_seconds(base_sleep: float, max_sleep: float, retry_count: int) -> float:
    return min(max_sleep, base_sleep * (2 ** max(retry_count - 1, 0)))


class SWHContentClient:
    def __init__(self, transport: str = "https", unsigned: bool = True, timeout: int = 60):
        self.transport = transport
        self.timeout = timeout
        self.smart_open = None
        self.client = None

        if transport == "s3":
            import boto3
            from smart_open import open as smart_open

            self.smart_open = smart_open
            if unsigned:
                from botocore import UNSIGNED
                from botocore.config import Config

                self.client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
            else:
                self.client = boto3.Session().client("s3")

    def read_text(self, blob_id: str, encoding: str = "utf-8") -> str:
        if self.transport == "https":
            url = f"https://softwareheritage.s3.amazonaws.com/content/{blob_id}"
            with urlopen(url, timeout=self.timeout) as response:
                raw = response.read()
            return gzip.decompress(raw).decode(encoding or "utf-8", errors="replace")

        url = f"s3://softwareheritage/content/{blob_id}"
        with self.smart_open(url, "rb", compression=".gz", transport_params={"client": self.client}) as f:
            return f.read().decode(encoding or "utf-8", errors="replace")


def normalize_language(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[\s_-]+", "", str(value).strip().lower())


def language_matches(doc: Dict[str, Any], source: Dict[str, Any]) -> bool:
    languages = {normalize_language(x) for x in source.get("languages", [])}
    if not languages:
        return True
    candidates = [
        doc.get("language"),
        doc.get("lang"),
        doc.get("programming_language"),
        source.get("language"),
    ]
    path = str(doc.get("path") or "")
    extension_map = {
        ".py": "python",
        ".ipynb": "jupyternotebook",
        ".js": "javascript",
        ".jsx": "javascript",
        ".c": "c",
        ".h": "c",
        ".rs": "rust",
        ".go": "go",
    }
    for ext, lang in extension_map.items():
        if path.lower().endswith(ext):
            candidates.append(lang)
    return any(normalize_language(c) in languages for c in candidates if c)


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n\n".join(flatten_text(v) for v in value if v is not None)
    if isinstance(value, dict):
        return "\n\n".join(flatten_text(v) for v in value.values() if v is not None)
    return str(value)


def extract_text(row: Dict[str, Any], text_field: Any) -> str:
    if isinstance(text_field, list):
        return "\n\n".join(flatten_text(row.get(field)) for field in text_field if row.get(field) is not None)
    return flatten_text(row.get(text_field))


def extract_python_notebook(text: str) -> str:
    try:
        notebook = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(notebook, dict):
        return ""

    metadata = notebook.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    language_info = metadata.get("language_info", {})
    if not isinstance(language_info, dict):
        language_info = {}
    kernelspec = metadata.get("kernelspec", {})
    if not isinstance(kernelspec, dict):
        kernelspec = {}
    language = normalize_language(
        language_info.get("name") or kernelspec.get("language") or kernelspec.get("name")
    )
    if language and "python" not in language:
        return ""

    cells = []
    raw_cells = notebook.get("cells", [])
    if not isinstance(raw_cells, list):
        return ""
    for cell in raw_cells:
        if not isinstance(cell, dict):
            continue
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if source is None:
            continue
        if isinstance(source, list):
            source = "".join(flatten_text(part) for part in source if part is not None)
        elif not isinstance(source, str):
            source = flatten_text(source)
        if source.strip():
            cells.append(source)
    return "\n\n".join(cells)


def strip_html(text: str) -> str:
    text = re.sub(r"<pre><code>(.*?)</code></pre>", r"\n\1\n", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<code>(.*?)</code>", r"`\1`", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_stackexchange_preference_text(row: Dict[str, Any]) -> str:
    question = row.get("question", "")
    if isinstance(question, dict):
        question = question.get("text") or question.get("body") or flatten_text(question)
    parts = [strip_html(flatten_text(question))]

    answers = row.get("answers") or []
    if isinstance(answers, dict):
        answers = answers.get("answers") or answers.get("text") or []
    if isinstance(answers, str):
        answers = [answers]

    answer_texts = []
    for answer in answers:
        if isinstance(answer, dict):
            text = answer.get("text") or answer.get("body") or answer.get("content") or ""
        else:
            text = flatten_text(answer)
        text = strip_html(text)
        if text:
            answer_texts.append(text)

    if answer_texts:
        parts.append("\n\n".join(answer_texts))
    return "\n\n".join(part for part in parts if part)


def extract_github_issue_text(row: Dict[str, Any]) -> str:
    content = flatten_text(row.get("content")).strip()
    if content:
        return content

    parts = []
    for key in ("title", "body", "description"):
        text = strip_html(flatten_text(row.get(key))).strip()
        if text:
            parts.append(text)

    events = row.get("events")
    comments = row.get("comments")
    for value in (events, comments):
        if not value:
            continue
        if isinstance(value, str):
            text = strip_html(value).strip()
            if text:
                parts.append(text)
            continue
        if isinstance(value, dict):
            value = value.get("events") or value.get("comments") or value.get("items") or [value]
        if not isinstance(value, list):
            value = [value]
        for item in value:
            if isinstance(item, dict):
                text = (
                    item.get("text")
                    or item.get("body")
                    or item.get("comment")
                    or item.get("content")
                    or item.get("description")
                    or ""
                )
            else:
                text = flatten_text(item)
            text = strip_html(flatten_text(text)).strip()
            if text:
                parts.append(text)

    return "\n\n".join(parts)


def row_to_documents(
    row: Dict[str, Any],
    source: Dict[str, Any],
    swh_client: Optional[SWHContentClient],
) -> Iterator[Dict[str, Any]]:
    if source.get("content_loader") == "stack_v2_swh":
        if swh_client is None:
            raise RuntimeError("The Stack v2 source requires smart_open[s3] and boto3.")
        for file_info in row.get("files", []):
            doc = {
                "category": source["category"],
                "path": file_info.get("path"),
                "language": file_info.get("language"),
                "source_id": source["id"],
            }
            if not language_matches(doc, source):
                continue
            if file_info.get("is_vendor") or file_info.get("is_generated"):
                continue
            try:
                text = swh_client.read_text(file_info["blob_id"], file_info.get("src_encoding") or "utf-8")
            except Exception as exc:
                if rank0():
                    print(f"Skipping SWH blob {file_info.get('blob_id')} after error: {exc}")
                continue
            doc["text"] = text
            yield doc
        return

    path_field = source.get("path_field")
    processor = source.get("content_processor")
    text = extract_text(row, source.get("text_field", "text"))
    if processor == "python_notebook_code_cells":
        text = extract_python_notebook(text)
    elif processor == "stackexchange_preference_qa":
        text = extract_stackexchange_preference_text(row)
    elif processor == "github_issue_conversation":
        text = extract_github_issue_text(row)
    elif not text and source.get("id") == "github_issues":
        text = extract_github_issue_text(row)
    doc = {
        "text": text,
        "category": source["category"],
        "path": row.get(path_field) if path_field else None,
        "language": row.get("language") or row.get("lang"),
        "source_id": source["id"],
    }
    yield doc


def looks_english(text: str, config: Dict[str, Any]) -> bool:
    english_cfg = config["data"].get("english_filter", {})
    if not english_cfg.get("enabled", True):
        return True
    sample = text[:5000]
    if not sample.strip():
        return False
    ascii_ratio = sum(ord(ch) < 128 for ch in sample) / max(len(sample), 1)
    if ascii_ratio < float(english_cfg.get("min_ascii_ratio", 0.85)):
        return False
    words = re.findall(r"[A-Za-z]{2,}", sample.lower())
    hits = sum(1 for word in words[:400] if word in STOPWORDS)
    return hits >= int(english_cfg.get("min_stopword_hits", 2))


def keep_document(doc: Dict[str, Any], source: Dict[str, Any], config: Dict[str, Any]) -> bool:
    text = doc.get("text") or ""
    if len(text) < int(config["data"].get("min_doc_chars", 32)):
        return False
    category = source["category"]
    if category in CODE_CATEGORIES and not language_matches(doc, source):
        return False
    if source.get("english_only") or category in {"technical_english", "english_replay"}:
        return looks_english(text, config)
    return True


def fim_probability(tokens_seen: int, config: Dict[str, Any]) -> float:
    fim_cfg = config["fim"]
    if not fim_cfg.get("enabled", True):
        return 0.0
    ramp_tokens = max(int(fim_cfg["ramp_tokens"]), 1)
    progress = min(tokens_seen / ramp_tokens, 1.0)
    start = float(fim_cfg.get("start_probability", 0.0))
    max_probability = float(fim_cfg["max_probability"])
    return start + (max_probability - start) * math.sin((math.pi / 2.0) * progress)


def encode_document_tokens(
    doc: Dict[str, Any],
    tokenizer: Any,
    rng: random.Random,
    config: Dict[str, Any],
) -> List[int]:
    text = doc["text"]
    path = doc.get("path")
    tokens: List[int] = []
    if path and rng.random() < float(config["fim"].get("prepend_filename_probability", 0.0)):
        filename_id = tokenizer.convert_tokens_to_ids(config["special_tokens"]["filename"])
        tokens.append(filename_id)
        tokens.extend(tokenizer.encode(str(path) + "\n", add_special_tokens=False))
    tokens.extend(tokenizer.encode(text, add_special_tokens=False))
    return tokens


def tokenize_with_fim(
    doc: Dict[str, Any],
    tokens_seen: int,
    rng: random.Random,
    config: Dict[str, Any],
    tokenizer: Any,
) -> Tuple[List[int], str]:
    tokens = encode_document_tokens(doc, tokenizer, rng, config)
    category = doc["category"]
    if category not in set(config["fim"].get("apply_to_categories", [])):
        return tokens, "standard"
    if len(tokens) < int(config["fim"].get("min_tokens", 8)):
        return tokens, "standard"
    if rng.random() >= fim_probability(tokens_seen, config):
        return tokens, "standard"

    fim_prefix_id = tokenizer.convert_tokens_to_ids(config["special_tokens"]["fim_prefix"])
    fim_middle_id = tokenizer.convert_tokens_to_ids(config["special_tokens"]["fim_middle"])
    fim_suffix_id = tokenizer.convert_tokens_to_ids(config["special_tokens"]["fim_suffix"])

    # HumanEval-style completion: place the cut right after a `def ...: """docstring"""`.
    # Trains the model to fill the function body given signature + docstring.
    he_frac = float(config["fim"].get("humaneval_completion_fraction", 0.0))
    if he_frac > 0.0 and category == "python_code" and rng.random() < he_frac:
        text = doc["text"]
        matches = list(_DOCSTRING_AFTER_DEF.finditer(text))
        if matches:
            m = rng.choice(matches)
            cut = m.end()
            prefix_text, middle_text = text[:cut], text[cut:]
            if middle_text.strip():
                head: List[int] = []
                path = doc.get("path")
                if path and rng.random() < float(config["fim"].get("prepend_filename_probability", 0.0)):
                    filename_id = tokenizer.convert_tokens_to_ids(config["special_tokens"]["filename"])
                    head.append(filename_id)
                    head.extend(tokenizer.encode(str(path) + "\n", add_special_tokens=False))
                prefix_tokens = head + tokenizer.encode(prefix_text, add_special_tokens=False)
                middle_tokens = tokenizer.encode(middle_text, add_special_tokens=False)
                if len(prefix_tokens) >= 4 and len(middle_tokens) >= 4:
                    return (
                        [fim_prefix_id] + prefix_tokens + [fim_suffix_id]
                        + [fim_middle_id] + middle_tokens,
                        "fim_psm",
                    )
        # fall through if no docstring found or chunks too short

    # End-of-file completion: empty suffix, PSM order. Matches the prompt
    # shape used at inference when the cursor is at end-of-buffer.
    end_frac = float(config["fim"].get("end_completion_fraction", 0.0))
    if end_frac > 0.0 and rng.random() < end_frac:
        min_chunk = max(1, int(config["fim"].get("min_tokens", 8)) // 2)
        a = rng.randint(min_chunk, len(tokens) - min_chunk)
        prefix, middle = tokens[:a], tokens[a:]
        return [fim_prefix_id] + prefix + [fim_suffix_id] + [fim_middle_id] + middle, "fim_psm"

    a, b = sorted(rng.sample(range(1, len(tokens)), 2))
    prefix, middle, suffix = tokens[:a], tokens[a:b], tokens[b:]

    if rng.random() < float(config["fim"].get("psm_fraction", 0.5)):
        return [fim_prefix_id] + prefix + [fim_suffix_id] + suffix + [fim_middle_id] + middle, "fim_psm"
    return [fim_suffix_id] + suffix + [fim_prefix_id] + prefix + [fim_middle_id] + middle, "fim_spm"


class DSShardWriter:
    def __init__(
        self,
        shard_dir: Path,
        tokenizer_name: str,
        token_size: int,
        max_tokens_per_file: int,
    ):
        self.shard_dir = shard_dir
        self.tokenizer_name = tokenizer_name
        self.token_size = token_size
        self.max_tokens_per_file = max_tokens_per_file
        self.files: Dict[str, TokenizedFile] = {}
        self.file_tokens: Dict[str, int] = defaultdict(int)
        self.parts: Dict[str, int] = defaultdict(int)
        self.mode_tokens: Dict[str, int] = defaultdict(int)
        self.mode_docs: Dict[str, int] = defaultdict(int)

    def _open(self, mode: str) -> TokenizedFile:
        filename = f"{mode}_{self.parts[mode]:05d}.ds"
        return TokenizedFile(
            str(self.shard_dir),
            filename,
            save_index=True,
            save_loss_metadata=False,
            tokenizer_name_or_path=self.tokenizer_name,
            save_final_metadata=True,
            token_size=self.token_size,
        )

    def write(self, mode: str, tokens: List[int]) -> None:
        if not tokens:
            return
        current = self.files.get(mode)
        if current is None:
            current = self._open(mode)
            self.files[mode] = current
        if self.file_tokens[mode] and self.file_tokens[mode] + len(tokens) > self.max_tokens_per_file:
            current.close()
            self.parts[mode] += 1
            current = self._open(mode)
            self.files[mode] = current
            self.file_tokens[mode] = 0
        current.write(tokens, loss_values=None)
        self.file_tokens[mode] += len(tokens)
        self.mode_tokens[mode] += len(tokens)
        self.mode_docs[mode] += 1

    def close(self) -> Dict[str, Any]:
        for writer in self.files.values():
            writer.close()
        return {
            "mode_tokens": dict(self.mode_tokens),
            "mode_docs": dict(self.mode_docs),
            "files": sum(self.parts.values()) + len(self.files),
        }


def _consume_source(
    source: Dict[str, Any],
    target: int,
    base_skip: int,
    base_epoch: int,
    shard_id: int,
    shard_start_tokens: int,
    config: Dict[str, Any],
    tokenizer: Any,
    writer: DSShardWriter,
    writer_lock: threading.Lock,
    token_counter: List[int],
    counter_lock: threading.Lock,
    progress_times: Dict[str, float],
    progress_lock: threading.Lock,
) -> Dict[str, Any]:
    # Per-source streamer used by the threaded shard builder. Writer access
    # and the shared FIM-position counter are the only cross-thread state;
    # both are guarded by locks. Each source seeds its own RNG so tokenize
    # decisions are independent of thread scheduling. progress_times[source_id]
    # is updated on each successful row read so a watchdog can spot stalls.
    source_id = source["id"]
    rng = random.Random(
        int(config["seed"]) + int(shard_id) * 1_000_003 + stable_int_hash(source_id)
    )

    swh_client = None
    if source.get("content_loader") == "stack_v2_swh":
        swh_transport = source.get("swh_transport", "https")
        log_rank0(
            f"Source {source_id} uses SWH content fetches via {swh_transport}; "
            "first rows can be slow."
        )
        swh_client = SWHContentClient(
            transport=swh_transport,
            unsigned=bool(source.get("swh_unsigned", True)),
            timeout=int(source.get("swh_timeout_seconds", 60)),
        )

    source_tokens = 0
    source_docs = 0
    source_examples = 0
    stream_retries = 0
    consecutive_stream_errors = 0
    max_stream_retries, retry_initial_seconds, retry_max_seconds = stream_retry_settings(config, source)
    max_empty_examples = int(
        source.get("max_empty_examples", config["data"].get("max_empty_examples_per_source", 500_000))
    )
    progress_interval = max(1, int(config["tokens"].get("progress_log_tokens", 1_000_000)))
    cycle_on_exhaustion = bool(source.get("cycle_on_exhaustion", config["data"].get("cycle_on_exhaustion", True)))
    max_epochs_per_shard = max(
        1,
        int(source.get("max_epochs_per_shard", config["data"].get("max_source_epochs_per_shard", 16))),
    )
    current_epoch = int(base_epoch)
    current_skip = max(0, int(base_skip))
    final_epoch = current_epoch
    final_offset = current_skip
    epochs_completed = 0

    def mark_alive() -> None:
        with progress_lock:
            progress_times[source_id] = time.time()

    def handle_stream_error(exc: Exception, stream_epoch: int, stream_skip: int) -> None:
        nonlocal consecutive_stream_errors, stream_retries
        consecutive_stream_errors += 1
        stream_retries += 1
        if consecutive_stream_errors > max_stream_retries:
            raise RuntimeError(
                f"Source {source_id} stream failed after {max_stream_retries} retries "
                f"at epoch {stream_epoch:,} offset {stream_skip:,}."
            ) from exc
        sleep_seconds = retry_sleep_seconds(
            retry_initial_seconds,
            retry_max_seconds,
            consecutive_stream_errors,
        )
        log_rank0(
            f"Source {source_id} stream error after {source_examples:,} examples: "
            f"{type(exc).__name__}: {exc}. Reopening at skip={stream_skip:,} "
            f"epoch={stream_epoch:,} in {sleep_seconds:.1f}s "
            f"(retry {consecutive_stream_errors}/{max_stream_retries})."
        )
        time.sleep(sleep_seconds)
        mark_alive()

    log_rank0(f"Source {source_id} target={target:,} tokens")
    mark_alive()

    while source_tokens < target:
        stream_epoch = current_epoch
        stream_skip = current_skip
        stream_seed = source_epoch_seed(config, source_id, stream_epoch)
        epoch_examples = 0
        stream_exhausted = False
        stream_failed = False
        try:
            stream = load_hf_stream(source, config, stream_seed, stream_skip)
        except Exception as exc:
            handle_stream_error(exc, stream_epoch, stream_skip)
            continue
        stream_iter = iter(stream)
        mark_alive()
        while source_tokens < target:
            try:
                row = next(stream_iter)
            except StopIteration:
                stream_exhausted = True
                break
            except Exception as exc:
                current_skip = stream_skip + epoch_examples
                final_epoch = stream_epoch
                final_offset = current_skip
                handle_stream_error(exc, stream_epoch, current_skip)
                stream_failed = True
                break
            consecutive_stream_errors = 0
            source_examples += 1
            epoch_examples += 1
            final_epoch = stream_epoch
            final_offset = stream_skip + epoch_examples
            mark_alive()
            if source_docs == 0 and source_examples >= max_empty_examples:
                keys = ", ".join(sorted(str(key) for key in row.keys()))
                raise RuntimeError(
                    f"Source {source_id} produced zero accepted documents after "
                    f"{source_examples:,} streamed examples. Check text_field/content_processor "
                    f"for this dataset. Last row keys: {keys}"
                )
            for doc in row_to_documents(row, source, swh_client):
                if not keep_document(doc, source, config):
                    continue
                with counter_lock:
                    fim_position = shard_start_tokens + token_counter[0]
                tokens, mode = tokenize_with_fim(
                    doc,
                    fim_position,
                    rng,
                    config,
                    tokenizer,
                )
                tokens.append(tokenizer.eos_token_id)
                token_count = len(tokens)
                with writer_lock:
                    writer.write(mode, tokens)
                with counter_lock:
                    token_counter[0] += token_count
                    shard_total_snapshot = token_counter[0]
                prev_tokens = source_tokens
                source_tokens += token_count
                source_docs += 1
                if (
                    source_docs == 1
                    or source_tokens // progress_interval != prev_tokens // progress_interval
                ):
                    log_rank0(
                        f"Source {source_id} progress: "
                        f"{source_tokens:,}/{target:,} tokens, docs={source_docs:,}, "
                        f"shard_total={shard_total_snapshot:,}"
                    )
                if source_tokens >= target:
                    break

        if source_tokens >= target:
            break
        if stream_failed:
            continue
        if not stream_exhausted:
            break

        epochs_completed += 1
        next_epoch = stream_epoch + 1
        final_epoch = next_epoch
        final_offset = 0
        log_rank0(
            f"Source {source_id} exhausted epoch {stream_epoch:,} after "
            f"{epoch_examples:,} streamed examples; "
            f"tokens={source_tokens:,}/{target:,}."
        )
        if not cycle_on_exhaustion:
            break
        if epochs_completed >= max_epochs_per_shard:
            log_rank0(
                f"Source {source_id} reached max_epochs_per_shard={max_epochs_per_shard}; "
                f"leaving source under target for this shard."
            )
            break
        current_epoch = next_epoch
        current_skip = 0
        mark_alive()

    summary = (
        f"Finished source {source_id}: "
        f"{source_tokens:,}/{target:,} tokens from {source_docs:,} docs "
        f"({source_examples:,} streamed examples, retries={stream_retries}, "
        f"epoch={final_epoch:,}, offset={final_offset:,})"
    )
    return {
        "source_id": source_id,
        "final_offset": final_offset,
        "final_epoch": final_epoch,
        "stats": {
            "target_tokens": target,
            "actual_tokens": source_tokens,
            "documents": source_docs,
            "examples_consumed": source_examples,
            "epoch": final_epoch,
            "offset": final_offset,
            "epochs_completed": epochs_completed,
            "stream_retries": stream_retries,
        },
        "summary": summary,
    }


def build_shard(
    config: Dict[str, Any],
    manifest: Dict[str, Any],
    shard_id: int,
    tokenizer: Any,
    token_size: int,
) -> Dict[str, Any]:
    shard_root = Path(config["paths"]["shard_root"])
    shard_dir = shard_root / f"shard_{shard_id:05d}"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    remaining = int(config["tokens"]["target_total_tokens"]) - int(manifest.get("tokens_prepared", 0))
    shard_target = min(resolve_shard_target_tokens(config, token_size), remaining)
    targets = source_targets(config, shard_target)

    writer = DSShardWriter(
        shard_dir,
        config["model"]["tokenizer_name"],
        token_size,
        int(config["tokens"].get("max_tokens_per_ds_file", 250_000_000)),
    )
    source_offsets = dict(manifest.get("source_offsets", {}))
    source_epochs = dict(manifest.get("source_epochs", {}))
    source_stats: Dict[str, Any] = {}
    shard_start_tokens = int(manifest.get("tokens_prepared", 0))
    for source in enabled_sources(config):
        source_id = source["id"]
        if int(source_offsets.get(source_id, 0)) > 0 and should_reset_zero_token_offset(manifest, source):
            log_rank0(
                f"Resetting source offset for {source_id}: previous shards consumed examples "
                "but produced zero tokens."
            )
            source_offsets[source_id] = 0
            source_epochs[source_id] = int(source_epochs.get(source_id, 0)) + 1

    active_sources = [s for s in enabled_sources(config) if int(targets.get(s["id"], 0)) > 0]
    configured_workers = int(config["training"].get("max_parallel_sources", 4))
    max_workers = max(1, min(configured_workers, len(active_sources))) if active_sources else 1

    stream_timeout = float(config["training"].get("stream_read_timeout_seconds", 120))
    shard_build_timeout = float(config["training"].get("shard_build_timeout_seconds", 0))
    shard_build_timeout_text = (
        f"{shard_build_timeout}s" if shard_build_timeout > 0 else "disabled"
    )
    stall_warn_seconds = float(config["training"].get("source_stall_warn_seconds", 300))

    log_rank0(
        f"Building shard {shard_id} at {shard_dir} "
        f"target={shard_target:,} tokens start={shard_start_tokens:,} "
        f"sources={len(active_sources)} parallel={max_workers} "
        f"stream_timeout={stream_timeout}s build_timeout={shard_build_timeout_text}"
    )

    writer_lock = threading.Lock()
    counter_lock = threading.Lock()
    token_counter = [0]

    # Apply a default socket read timeout so a hung HF stream raises instead
    # of blocking forever; the existing per-source retry logic then reopens
    # the stream. Save/restore around this call so we don't disturb the
    # surrounding process when build_shard is invoked from the trainer as a
    # fallback (the dedicated producer subprocess sets it once globally).
    prev_socket_timeout = socket.getdefaulttimeout()
    if stream_timeout > 0:
        socket.setdefaulttimeout(stream_timeout)

    progress_times: Dict[str, float] = {s["id"]: time.time() for s in active_sources}
    progress_lock = threading.Lock()
    watchdog_stop = threading.Event()

    def watchdog() -> None:
        while not watchdog_stop.wait(30):
            now = time.time()
            with progress_lock:
                progress_items = list(progress_times.items())
            for sid, last_t in progress_items:
                stall = now - last_t
                if stall > stall_warn_seconds:
                    log_rank0(
                        f"Watchdog: source {sid} no progress for {stall:.0f}s "
                        f"(socket_timeout={socket.getdefaulttimeout()}s, "
                        f"build_elapsed={now - build_started:.0f}s)."
                    )

    build_started = time.time()
    watchdog_thread = threading.Thread(target=watchdog, daemon=True, name="src-watchdog")
    watchdog_thread.start()

    try:
        result_queue: "queue.Queue[Tuple[str, bool, Any]]" = queue.Queue()
        task_queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        for source in active_sources:
            task_queue.put(source)
        for _ in range(max_workers):
            task_queue.put(None)

        def source_worker() -> None:
            while True:
                source = task_queue.get()
                try:
                    if source is None:
                        return
                    source_id = source["id"]
                    try:
                        result = _consume_source(
                            source,
                            int(targets.get(source_id, 0)),
                            int(source_offsets.get(source_id, 0)),
                            int(source_epochs.get(source_id, 0)),
                            shard_id,
                            shard_start_tokens,
                            config,
                            tokenizer,
                            writer,
                            writer_lock,
                            token_counter,
                            counter_lock,
                            progress_times,
                            progress_lock,
                        )
                    except BaseException as exc:
                        result_queue.put((source_id, False, exc))
                    else:
                        result_queue.put((source_id, True, result))
                finally:
                    task_queue.task_done()

        source_threads = [
            threading.Thread(target=source_worker, daemon=True, name=f"src-{idx}")
            for idx in range(max_workers)
        ]
        for thread in source_threads:
            thread.start()

        deadline = build_started + shard_build_timeout if shard_build_timeout > 0 else None
        remaining_sources = {source["id"] for source in active_sources}
        while remaining_sources:
            if deadline is None:
                result_timeout = 30.0
            else:
                time_left = deadline - time.time()
                if time_left <= 0:
                    stuck_ids = sorted(remaining_sources)
                    raise RuntimeError(
                        f"Shard {shard_id} build timed out after "
                        f"{shard_build_timeout:.0f}s; stuck sources: {stuck_ids}"
                    )
                result_timeout = min(time_left, 30.0)
            try:
                source_id, ok, payload = result_queue.get(timeout=result_timeout)
            except queue.Empty:
                continue
            if source_id not in remaining_sources:
                continue
            remaining_sources.discard(source_id)
            with progress_lock:
                progress_times.pop(source_id, None)
            if not ok:
                raise RuntimeError(
                    f"Source {source_id} failed while building shard {shard_id}."
                ) from payload
            result = payload
            source_offsets[result["source_id"]] = result["final_offset"]
            source_epochs[result["source_id"]] = result["final_epoch"]
            source_stats[result["source_id"]] = result["stats"]
            log_rank0(result["summary"])

        for thread in source_threads:
            thread.join(timeout=1)
    finally:
        watchdog_stop.set()
        watchdog_thread.join(timeout=5)
        if stream_timeout > 0:
            socket.setdefaulttimeout(prev_socket_timeout)

    tokens_written_total = token_counter[0]
    writer_stats = writer.close()
    if tokens_written_total <= 0:
        raise RuntimeError(f"Shard {shard_id} produced zero tokens; refusing to add an empty shard.")
    log_rank0(
        f"Shard {shard_id} built: {tokens_written_total:,} tokens, "
        f"files={writer_stats['files']}, mode_tokens={writer_stats['mode_tokens']}"
    )
    shard_stats = {
        "id": shard_id,
        "path": str(shard_dir),
        "status": "built",
        "created_at": utc_now(),
        "target_tokens": shard_target,
        "actual_tokens": tokens_written_total,
        "tokens_start": shard_start_tokens,
        "tokens_end": shard_start_tokens + tokens_written_total,
        "source_stats": source_stats,
        **writer_stats,
    }
    write_json(shard_dir / "shard_stats.json", shard_stats)

    with locked_manifest(config) as current_manifest:
        if int(current_manifest.get("tokens_prepared", 0)) != shard_start_tokens:
            raise RuntimeError(
                "Manifest tokens_prepared changed while building shard "
                f"{shard_id}: expected {shard_start_tokens:,}, "
                f"found {int(current_manifest.get('tokens_prepared', 0)):,}."
            )
        if len(current_manifest.get("shards", [])) != shard_id:
            raise RuntimeError(
                "Manifest shard list changed while building shard "
                f"{shard_id}: expected length {shard_id}, "
                f"found {len(current_manifest.get('shards', []))}."
            )
        current_manifest["tokens_prepared"] = shard_start_tokens + tokens_written_total
        current_manifest["source_offsets"] = source_offsets
        current_manifest["source_epochs"] = source_epochs
        current_manifest["shards"].append(shard_stats)
    return shard_stats


def mode_from_path(path: str) -> int:
    name = Path(path).name
    if name.startswith("fim_psm_"):
        return MODES["fim_psm"]
    if name.startswith("fim_spm_"):
        return MODES["fim_spm"]
    return MODES["standard"]


class PackedShardBatcher:
    def __init__(
        self,
        shard_dir: Path,
        config: Dict[str, Any],
        token_size: int,
        rank: int,
        world_size: int,
        seed: int,
        start_microbatch: int = 0,
    ):
        self.config = config
        self.batch_size = int(config["training"]["per_device_batch_size"])
        self.dataset = DatatroveFolderDataset(
            str(shard_dir),
            int(config["training"]["max_seq_len"]),
            filename_pattern="*.ds",
            token_size=token_size,
            recursive=True,
            shuffle=True,
            seed=seed + rank,
            return_positions=bool(config["training"].get("isolate_documents", True)),
        )
        full_len = len(self.dataset)
        per_rank = full_len // world_size
        self.start = rank * per_rank
        self.end = self.start + per_rank
        self.current = self.start + start_microbatch * self.batch_size
        self.microbatches = max((self.end - self.start) // self.batch_size, 0)

    def current_mode(self) -> int:
        current_file = self.dataset.files[self.dataset.current_file]
        for attr in ("read_path", "file_path", "path"):
            file_path = getattr(current_file, attr, None)
            if file_path:
                return mode_from_path(str(file_path))
        return MODES["standard"]

    def next_batch(
        self,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
        if self.current + self.batch_size > self.end:
            return None
        rows = []
        positions = []
        modes = []
        for idx in range(self.current, self.current + self.batch_size):
            item = self.dataset[idx]
            rows.append(item["input_ids"])
            if "positions" in item:
                positions.append(item["positions"])
            modes.append(self.current_mode())
        self.current += self.batch_size
        batch = torch.stack(rows)
        if positions:
            full_positions = torch.stack(positions)
            pos_batch = full_positions[:, :-1]
            loss_mask = full_positions[:, 1:].ne(0)
        else:
            pos_batch = None
            loss_mask = None
        return batch[:, :-1], batch[:, 1:], torch.tensor(modes, dtype=torch.long), pos_batch, loss_mask


def lr_for_step(step: int, total_steps: int, config: Dict[str, Any]) -> float:
    tcfg = config["training"]
    peak = float(tcfg["learning_rate"])
    min_lr = float(tcfg["min_learning_rate"])
    start_lr = float(tcfg.get("warmup_start_learning_rate", min_lr))
    warmup_steps = max(1, int(total_steps * float(tcfg["warmup_ratio"])))
    if step < warmup_steps:
        return start_lr + (peak - start_lr) * ((step + 1) / warmup_steps)
    progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
    return min_lr + 0.5 * (peak - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def compute_loss_by_mode(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    modes: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    loss_chunk_tokens: int = 0,
) -> Tuple[torch.Tensor, Dict[str, Tuple[float, int]]]:
    if loss_chunk_tokens > 0 and loss_chunk_tokens < y.shape[1]:
        hidden, _, aux_loss = model(x, targets=None, positions=positions, return_hidden=True)
        head = getattr(model, "module", model).ll_head
        batch_size, seq_len = y.shape
        sample_loss_sums = torch.zeros(batch_size, device=y.device, dtype=torch.float32)

        for start in range(0, seq_len, loss_chunk_tokens):
            end = min(start + loss_chunk_tokens, seq_len)
            hidden_chunk = hidden[:, start:end, :]
            target_chunk = y[:, start:end]
            mask_chunk = loss_mask[:, start:end] if loss_mask is not None else None
            chunk_len = end - start

            def chunk_loss_sums(
                chunk_hidden: torch.Tensor,
                target_chunk: torch.Tensor = target_chunk,
                mask_chunk: Optional[torch.Tensor] = mask_chunk,
                chunk_len: int = chunk_len,
            ) -> torch.Tensor:
                logits = head(chunk_hidden)
                vocab = logits.shape[-1]
                losses = F.cross_entropy(
                    logits.reshape(-1, vocab),
                    target_chunk.reshape(-1),
                    reduction="none",
                ).view(batch_size, chunk_len)
                if mask_chunk is not None:
                    losses = losses * mask_chunk.to(losses.dtype)
                return losses.float().sum(dim=1)

            sample_loss_sums = sample_loss_sums + checkpoint_fn(
                chunk_loss_sums,
                hidden_chunk,
                use_reentrant=False,
            )

        if loss_mask is not None:
            valid_tokens = loss_mask.sum(dim=1).clamp_min(1).to(sample_loss_sums.dtype)
        else:
            valid_tokens = torch.full(
                (batch_size,),
                seq_len,
                device=y.device,
                dtype=sample_loss_sums.dtype,
            )
        sample_loss = sample_loss_sums / valid_tokens
        loss = sample_loss.mean()
        if torch.is_tensor(aux_loss):
            loss = loss + aux_loss
        metrics: Dict[str, Tuple[float, int]] = {}
        for mode_id, mode_name in MODE_NAMES.items():
            mask = modes == mode_id
            count = int(mask.sum().item())
            if count:
                metrics[mode_name] = (float(sample_loss[mask].sum().detach().item()), count)
            else:
                metrics[mode_name] = (0.0, 0)
        return loss, metrics

    logits, _, aux_loss = model(x, targets=None, positions=positions)
    vocab = logits.shape[-1]
    token_loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1), reduction="none").view(y.shape)
    if loss_mask is not None:
        token_loss = token_loss * loss_mask.to(token_loss.dtype)
        valid_tokens = loss_mask.sum(dim=1).clamp_min(1).to(token_loss.dtype)
        sample_loss = token_loss.sum(dim=1) / valid_tokens
    else:
        sample_loss = token_loss.mean(dim=1)
    loss = sample_loss.mean()
    if torch.is_tensor(aux_loss):
        loss = loss + aux_loss
    metrics: Dict[str, Tuple[float, int]] = {}
    for mode_id, mode_name in MODE_NAMES.items():
        mask = modes == mode_id
        count = int(mask.sum().item())
        if count:
            metrics[mode_name] = (float(sample_loss[mask].sum().detach().item()), count)
        else:
            metrics[mode_name] = (0.0, 0)
    return loss, metrics


def reduce_mode_metrics(metrics: Dict[str, Tuple[float, int]], device: torch.device) -> Dict[str, float]:
    values: List[float] = []
    for mode in MODES:
        loss_sum, count = metrics.get(mode, (0.0, 0))
        values.extend([loss_sum, float(count)])
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    if is_dist():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    out = {}
    for i, mode in enumerate(MODES):
        loss_sum = tensor[2 * i].item()
        count = tensor[2 * i + 1].item()
        out[f"loss_{mode}"] = loss_sum / max(count, 1.0)
        out[f"count_{mode}"] = int(count)
    return out


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    tokens_trained: int,
    shard_id: int,
    shard_microbatch: int,
    best_loss: float,
    loss_ema: Optional[float],
) -> Dict[str, Any]:
    raw_model = model.module if isinstance(model, DDP) else model
    return {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "tokens_trained": tokens_trained,
        "shard_id": shard_id,
        "shard_microbatch": shard_microbatch,
        "best_loss": float(best_loss),
        "loss_ema": (float(loss_ema) if loss_ema is not None else None),
        "saved_at": utc_now(),
    }


def save_checkpoint(
    config: Dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    tokens_trained: int,
    shard_id: int,
    shard_microbatch: int,
    best_loss: float = float("inf"),
    loss_ema: Optional[float] = None,
) -> Path:
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"cpt_step_{global_step:08d}.pt"
    torch.save(
        _checkpoint_payload(
            model, optimizer, global_step, tokens_trained,
            shard_id, shard_microbatch, best_loss, loss_ema,
        ),
        path,
    )
    # Reserve one slot for the best checkpoint, so max_checkpoints_to_keep counts (recent + best).
    max_total = int(config["training"].get("max_checkpoints_to_keep", 0))
    prune_checkpoints(checkpoint_dir, max(max_total - 1, 1) if max_total > 0 else 0)
    return path


def save_best_checkpoint(
    config: Dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    tokens_trained: int,
    shard_id: int,
    shard_microbatch: int,
    best_loss: float,
    loss_ema: Optional[float],
) -> Path:
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "cpt_best.pt"
    torch.save(
        _checkpoint_payload(
            model, optimizer, global_step, tokens_trained,
            shard_id, shard_microbatch, best_loss, loss_ema,
        ),
        path,
    )
    return path


def prune_checkpoints(checkpoint_dir: Path, max_to_keep: int) -> None:
    if max_to_keep <= 0:
        return
    checkpoints = sorted(checkpoint_dir.glob("cpt_step_*.pt"))
    for checkpoint in checkpoints[:-max_to_keep]:
        checkpoint.unlink()


def load_resume_state(config: Dict[str, Any]) -> Dict[str, Any]:
    resume_setting = config["model"].get("resume_checkpoint", "latest")
    checkpoint_path = None
    if resume_setting == "latest":
        checkpoint_path = find_latest_checkpoint(Path(config["paths"]["checkpoint_dir"]))
    elif resume_setting:
        candidate = Path(resume_setting)
        if candidate.exists():
            checkpoint_path = candidate
    if checkpoint_path is None:
        return {}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["checkpoint_path"] = str(checkpoint_path)
    return checkpoint


def export_eval_model(
    config: Dict[str, Any],
    tokenizer: Any,
    model: torch.nn.Module,
    export_dir: Optional[Path] = None,
) -> str:
    if export_dir is None:
        export_dir = Path(config["paths"]["output_dir"]) / "eval_model"
    else:
        export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(export_dir)

    raw_model = model.module if isinstance(model, DDP) else model
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod
    state = {
        f"lightlm.{name}": tensor.detach().cpu().clone()
        for name, tensor in raw_model.state_dict().items()
    }
    if save_safetensors is not None:
        save_safetensors(state, str(export_dir / "model.safetensors"))
    else:
        torch.save(state, export_dir / "pytorch_model.bin")

    model_cfg = config["model"]
    hf_config = {
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
        "ffn_hidden_dims": model_cfg["ffn_hidden_dims"],
        "rmsnorm_eps": model_cfg["rmsnorm_eps"],
        "rope_theta": model_cfg["rope_theta"],
        "context_len": model_cfg["context_len"],
        "use_cache": True,
        "use_flash": model_cfg["use_flash"],
        "use_moe": model_cfg["use_moe"],
        "moe_num_experts": model_cfg["moe_num_experts"],
        "moe_active_experts": model_cfg["moe_active_experts"],
        "moe_eps": model_cfg["moe_eps"],
        "moe_aux_loss_coef": model_cfg["moe_aux_loss_coef"],
        "moe_shared_experts": model_cfg["moe_shared_experts"],
        "use_lossfreebalance": model_cfg["use_lossfreebalance"],
    }
    write_json(export_dir / "config.json", hf_config)
    write_json(
        export_dir / "generation_config.json",
        {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "use_cache": True,
        },
    )
    shutil.copy2(Path(__file__).with_name("hf_lightlm.py"), export_dir / "hf_lightlm.py")
    shutil.copy2(Path(__file__).with_name("model.py"), export_dir / "model.py")
    return str(export_dir.resolve())


def eval_format_values(
    output_dir: str,
    global_step: int,
    tokens_trained: int,
    checkpoint: str,
    eval_model: str,
    metric_output_path: str = "",
    stdout_log_file: str = "",
) -> Dict[str, Any]:
    return {
        "output_dir": output_dir,
        "step": global_step,
        "tokens": tokens_trained,
        "checkpoint": checkpoint,
        "eval_model": eval_model,
        "metric_output_path": metric_output_path,
        "stdout_log_file": stdout_log_file,
    }


def clean_subprocess_distributed_env(env: Dict[str, str]) -> Dict[str, str]:
    cleaned = env.copy()
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
        cleaned.pop(key, None)
    return cleaned


def run_subprocess_with_bounded_output(
    command: List[str],
    cwd: Optional[str],
    env: Dict[str, str],
    stdout_path: Optional[Path],
    mode: str,
    max_bytes: int,
) -> int:
    if mode == "full" and stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("wb") as f:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, check=False)
        truncate_file_tail(stdout_path, max_bytes)
        return result.returncode

    if mode == "discard":
        with open(os.devnull, "wb") as f:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, check=False)
        return result.returncode

    tail = bytearray()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert process.stdout is not None
    for chunk in iter(lambda: process.stdout.read(64 * 1024), b""):
        tail.extend(chunk)
        if max_bytes > 0 and len(tail) > max_bytes:
            del tail[: len(tail) - max_bytes]
    returncode = process.wait()
    if stdout_path is not None and tail:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("wb") as f:
            f.write(tail)
    return returncode


def compact_eval_results(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(payload, dict) and "results" in payload:
        return payload["results"]
    return payload


def run_bigcode_eval(
    config: Dict[str, Any],
    tokenizer: Any,
    model: torch.nn.Module,
    global_step: int,
    tokens_trained: int,
    checkpoint_path: Optional[Path],
    manifest: Dict[str, Any],
    force: bool = False,
) -> bool:
    eval_cfg = config["evaluation"]["bigcode_harness"]
    if not eval_cfg.get("enabled", False):
        return True
    next_eval = int(manifest.get("next_eval_tokens") or eval_cfg.get("interval_tokens", 0))
    if not force and tokens_trained < next_eval:
        return True

    eval_model = eval_cfg.get("eval_model", "{output_dir}")
    output_dir = str(Path(config["paths"]["output_dir"]).resolve())
    eval_model = eval_model.format(output_dir=output_dir, step=global_step, tokens=tokens_trained)
    if eval_model == output_dir:
        eval_model = export_eval_model(config, tokenizer, model)
    checkpoint = str(checkpoint_path.resolve()) if checkpoint_path else ""
    base_format_values = eval_format_values(output_dir, global_step, tokens_trained, checkpoint, eval_model)
    metric_output_path = eval_cfg.get(
        "metric_output_path",
        "{output_dir}/eval_results_step_{step}.json",
    ).format(**base_format_values)
    stdout_log_file = eval_cfg.get(
        "stdout_log_file",
        "{output_dir}/logs/eval_stdout_step_{step}.log",
    ).format(**base_format_values)
    stdout_log_mode = str(eval_cfg.get("stdout_log_mode", "tail")).lower()
    stdout_log_max_bytes = int(eval_cfg.get("stdout_log_max_bytes", 64 * 1024))
    if stdout_log_mode not in {"tail", "discard", "full"}:
        stdout_log_mode = "tail"
    if stdout_log_mode == "discard":
        stdout_log_file = ""
    format_values = eval_format_values(
        output_dir,
        global_step,
        tokens_trained,
        checkpoint,
        eval_model,
        metric_output_path,
        stdout_log_file,
    )

    command = [
        part.format(**format_values)
        for part in eval_cfg.get("command", [])
    ]
    log_path = Path(config["paths"]["eval_log_file"])
    append_jsonl(
        log_path,
        {
            "event": "bigcode_eval_start",
            "step": global_step,
            "tokens_trained": tokens_trained,
            "command": command,
            "metric_output_path": metric_output_path,
            "stdout_log_file": stdout_log_file,
            "time": utc_now(),
        },
    )
    if not command:
        append_jsonl(log_path, {"event": "bigcode_eval_skipped", "reason": "empty command", "time": utc_now()})
        return False

    cwd = eval_cfg.get("cwd") or None
    if cwd and not (Path(cwd) / "main.py").exists():
        append_jsonl(
            log_path,
            {
                "event": "bigcode_eval_skipped",
                "step": global_step,
                "tokens_trained": tokens_trained,
                "reason": f"main.py not found in cwd: {cwd}",
                "time": utc_now(),
            },
        )
        return False
    stdout_path = Path(stdout_log_file) if stdout_log_file else None
    env = clean_subprocess_distributed_env(os.environ)
    returncode = run_subprocess_with_bounded_output(
        command,
        cwd,
        env,
        stdout_path,
        stdout_log_mode,
        stdout_log_max_bytes,
    )
    eval_results = compact_eval_results(Path(metric_output_path))
    append_jsonl(
        log_path,
        {
            "event": "bigcode_eval_end",
            "step": global_step,
            "tokens_trained": tokens_trained,
            "returncode": returncode,
            "result_keys": sorted(eval_results.keys()) if isinstance(eval_results, dict) else None,
            "metric_output_path": metric_output_path,
            "stdout_log_file": stdout_log_file,
            "stdout_log_mode": stdout_log_mode,
            "time": utc_now(),
        },
    )
    if returncode != 0:
        return False
    interval = int(eval_cfg["interval_tokens"])
    if force and next_eval <= tokens_trained:
        next_eval = tokens_trained + interval
    else:
        while next_eval <= tokens_trained:
            next_eval += interval
    manifest["next_eval_tokens"] = next_eval
    return True


def mark_shard_status(config: Dict[str, Any], shard_id: int, status: str, **extra: Any) -> None:
    with locked_manifest(config) as manifest:
        for shard in manifest["shards"]:
            if int(shard["id"]) == int(shard_id):
                shard["status"] = status
                shard.update(extra)
                break


def train_shard(
    config: Dict[str, Any],
    shard: Dict[str, Any],
    tokenizer: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    token_size: int,
    device: torch.device,
    rank: int,
    world_size: int,
    resume_state: Dict[str, Any],
) -> Tuple[int, int, float, Optional[float]]:
    tcfg = config["training"]
    global_tokens_per_step = (
        int(tcfg["per_device_batch_size"])
        * int(tcfg["max_seq_len"])
        * int(tcfg["gradient_accumulation_steps"])
        * world_size
    )
    total_steps = math.ceil(int(config["tokens"]["target_total_tokens"]) / global_tokens_per_step)
    global_step = int(resume_state.get("global_step", 0))
    tokens_trained = int(resume_state.get("tokens_trained", 0))
    best_loss = float(resume_state.get("best_loss", float("inf")))
    loss_ema: Optional[float] = resume_state.get("loss_ema")
    if loss_ema is not None:
        loss_ema = float(loss_ema)
    loss_ema_alpha = float(tcfg.get("best_loss_ema_alpha", 0.1))
    start_microbatch = 0
    if int(resume_state.get("shard_id", -1)) == int(shard["id"]):
        start_microbatch = int(resume_state.get("shard_microbatch", 0))

    batcher = PackedShardBatcher(
        Path(shard["path"]),
        config,
        token_size,
        rank,
        world_size,
        int(config["seed"]) + int(shard["id"]),
        start_microbatch=start_microbatch,
    )
    if batcher.microbatches == 0:
        if rank0():
            print(f"Shard {shard['id']} has no full training batches.")
        return global_step, tokens_trained, best_loss, loss_ema

    dtype = get_dtype(tcfg["dtype"])
    use_amp = device.type == "cuda"
    grad_accum = int(tcfg["gradient_accumulation_steps"])
    loss_chunk_tokens = int(tcfg.get("loss_chunk_tokens", 0))
    checkpoint_interval = int(tcfg["checkpoint_interval_steps"])
    log_interval = int(tcfg["log_interval_steps"])
    train_log = Path(config["paths"]["train_log_file"])
    last_checkpoint: Optional[Path] = None

    model.train()

    start_time = time.perf_counter()
    microbatch = start_microbatch
    while microbatch + grad_accum <= batcher.microbatches:
        step_metrics = {mode: (0.0, 0) for mode in MODES}
        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        lr = lr_for_step(global_step, total_steps, config)
        set_optimizer_lr(optimizer, lr)

        for accum_idx in range(grad_accum):
            batch = batcher.next_batch()
            if batch is None:
                break
            x, y, modes, positions, loss_mask = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            modes = modes.to(device, non_blocking=True)
            positions = positions.to(device, non_blocking=True) if positions is not None else None
            loss_mask = loss_mask.to(device, non_blocking=True) if loss_mask is not None else None
            sync_context = (
                model.no_sync()
                if isinstance(model, DDP) and accum_idx < grad_accum - 1
                else nullcontext()
            )
            amp_context = torch.autocast(device_type=device.type, dtype=dtype) if use_amp else nullcontext()
            with sync_context:
                with amp_context:
                    loss, metrics = compute_loss_by_mode(
                        model,
                        x,
                        y,
                        modes,
                        positions,
                        loss_mask,
                        loss_chunk_tokens=loss_chunk_tokens,
                    )
                    (loss / grad_accum).backward()
            for mode, (loss_sum, count) in metrics.items():
                prev_sum, prev_count = step_metrics[mode]
                step_metrics[mode] = (prev_sum + loss_sum, prev_count + count)
            microbatch += 1

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
        optimizer.step()
        global_step += 1
        tokens_trained += global_tokens_per_step

        reduced = reduce_mode_metrics(step_metrics, device)
        elapsed = time.perf_counter() - step_start
        toks_per_sec = global_tokens_per_step / max(elapsed, 1e-6)

        # Token-weighted combined loss across all modes, smoothed with an EMA for stability.
        loss_sum_all = sum(reduced[f"loss_{m}"] * reduced[f"count_{m}"] for m in MODES)
        count_all = sum(reduced[f"count_{m}"] for m in MODES)
        if count_all > 0:
            combined = loss_sum_all / count_all
            loss_ema = combined if loss_ema is None else (1 - loss_ema_alpha) * loss_ema + loss_ema_alpha * combined

        if rank0() and (global_step % log_interval == 0 or global_step == 1):
            row = {
                "event": "train_step",
                "time": utc_now(),
                "step": global_step,
                "shard_id": shard["id"],
                "shard_microbatch": microbatch,
                "tokens_trained": tokens_trained,
                "lr": lr,
                "grad_norm": float(grad_norm),
                "tokens_per_second": toks_per_sec,
                **reduced,
            }
            append_jsonl(train_log, row)
            print(
                f"step {global_step} shard {shard['id']} "
                f"loss standard={reduced['loss_standard']:.4f} "
                f"psm={reduced['loss_fim_psm']:.4f} spm={reduced['loss_fim_spm']:.4f} "
                f"lr={lr:.2e} tok/s={toks_per_sec:.0f}"
            )

        if global_step % checkpoint_interval == 0:
            def checkpoint_eval_stage() -> None:
                nonlocal last_checkpoint, best_loss
                last_checkpoint = save_checkpoint(
                    config, model, optimizer, global_step, tokens_trained,
                    int(shard["id"]), microbatch, best_loss, loss_ema,
                )
                if loss_ema is not None and loss_ema < best_loss:
                    best_loss = loss_ema
                    save_best_checkpoint(
                        config, model, optimizer, global_step, tokens_trained,
                        int(shard["id"]), microbatch, best_loss, loss_ema,
                    )
                    print(f"  new best loss: {best_loss:.4f} (step {global_step}) -> cpt_best.pt")
                manifest = load_manifest(Path(config["paths"]["manifest_path"]))
                manifest["global_step"] = global_step
                manifest["tokens_trained"] = tokens_trained
                run_bigcode_eval(config, tokenizer, model, global_step, tokens_trained, last_checkpoint, manifest)
                with locked_manifest(config) as current_manifest:
                    current_manifest["global_step"] = global_step
                    current_manifest["tokens_trained"] = tokens_trained
                    current_manifest["next_eval_tokens"] = manifest.get(
                        "next_eval_tokens",
                        current_manifest.get("next_eval_tokens", 0),
                    )

            run_rank0_stage(config, f"checkpoint_eval_step_{global_step}", checkpoint_eval_stage)

    def final_checkpoint_eval_stage() -> None:
        nonlocal last_checkpoint, best_loss
        last_checkpoint = save_checkpoint(
            config, model, optimizer, global_step, tokens_trained,
            int(shard["id"]), microbatch, best_loss, loss_ema,
        )
        if loss_ema is not None and loss_ema < best_loss:
            best_loss = loss_ema
            save_best_checkpoint(
                config, model, optimizer, global_step, tokens_trained,
                int(shard["id"]), microbatch, best_loss, loss_ema,
            )
            print(f"  new best loss: {best_loss:.4f} (step {global_step}) -> cpt_best.pt")
        manifest = load_manifest(Path(config["paths"]["manifest_path"]))
        manifest["global_step"] = global_step
        manifest["tokens_trained"] = tokens_trained
        run_bigcode_eval(config, tokenizer, model, global_step, tokens_trained, last_checkpoint, manifest)
        with locked_manifest(config) as current_manifest:
            current_manifest["global_step"] = global_step
            current_manifest["tokens_trained"] = tokens_trained
            current_manifest["next_eval_tokens"] = manifest.get(
                "next_eval_tokens",
                current_manifest.get("next_eval_tokens", 0),
            )
        append_jsonl(
            train_log,
            {
                "event": "shard_complete",
                "time": utc_now(),
                "shard_id": shard["id"],
                "step": global_step,
                "tokens_trained": tokens_trained,
                "elapsed_seconds": time.perf_counter() - start_time,
            },
        )

    run_rank0_stage(config, f"final_checkpoint_eval_shard_{shard['id']}", final_checkpoint_eval_stage)
    return global_step, tokens_trained, best_loss, loss_ema


def next_shard_to_train(manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for shard in manifest.get("shards", []):
        if shard.get("status") in {"built", "training"}:
            return shard
    return None


def count_prefetch_shards(manifest: Dict[str, Any]) -> int:
    return sum(1 for shard in manifest.get("shards", []) if shard.get("status") in {"built", "training"})


def build_next_shard(config: Dict[str, Any], tokenizer: Any, token_size: int) -> Optional[Dict[str, Any]]:
    with file_lock(shard_build_lock_path(config)):
        manifest = load_manifest(Path(config["paths"]["manifest_path"]))
        if int(manifest.get("tokens_prepared", 0)) >= int(config["tokens"]["target_total_tokens"]):
            return None
        return build_shard(config, manifest, len(manifest.get("shards", [])), tokenizer, token_size)


def shard_producer_log_path(config: Dict[str, Any]) -> Path:
    return Path(config["paths"]["output_dir"]) / "logs" / "shard_producer.log"


def shard_producer_loop(config: Dict[str, Any], tokenizer: Any, token_size: int) -> None:
    prefetch_shards = max(1, int(config["training"].get("prefetch_shards", 2)))
    poll_seconds = max(1.0, float(config["training"].get("shard_producer_poll_seconds", 15.0)))
    stream_timeout = float(config["training"].get("stream_read_timeout_seconds", 120))
    if stream_timeout > 0:
        socket.setdefaulttimeout(stream_timeout)
    log_rank0(
        f"Shard producer started with prefetch_shards={prefetch_shards} "
        f"socket_default_timeout={socket.getdefaulttimeout()}s"
    )

    while True:
        manifest = load_manifest(Path(config["paths"]["manifest_path"]))
        if int(manifest.get("tokens_prepared", 0)) >= int(config["tokens"]["target_total_tokens"]):
            log_rank0("Shard producer finished: target tokens are prepared.")
            return
        if count_prefetch_shards(manifest) >= prefetch_shards:
            time.sleep(poll_seconds)
            continue
        build_next_shard(config, tokenizer, token_size)


def start_shard_producer(config: Dict[str, Any], config_path: str) -> subprocess.Popen:
    log_enabled = bool(config["training"].get("shard_producer_log_enabled", False))
    log_path = shard_producer_log_path(config) if log_enabled else None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    env = clean_subprocess_distributed_env(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONUNBUFFERED"] = "1"
    command = [sys.executable, str(Path(__file__).resolve()), "--config", config_path, "--shard-producer"]
    log_file = log_path.open("ab", buffering=0) if log_path is not None else open(os.devnull, "ab", buffering=0)
    process = subprocess.Popen(
        command,
        cwd=str(Path.cwd()),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    process._lightlm_log_file = log_file  # type: ignore[attr-defined]
    log_rank0(f"Started shard producer pid={process.pid}; log={log_path or 'discarded'}")
    return process


def stop_shard_producer(process: Optional[subprocess.Popen]) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
    log_file = getattr(process, "_lightlm_log_file", None)
    if log_file is not None:
        log_file.close()


def wait_for_produced_shard(
    config: Dict[str, Any],
    producer: Optional[subprocess.Popen],
) -> Optional[Dict[str, Any]]:
    poll_seconds = max(1.0, float(config["training"].get("shard_producer_poll_seconds", 15.0)))
    while True:
        manifest = load_manifest(Path(config["paths"]["manifest_path"]))
        shard = next_shard_to_train(manifest)
        if shard is not None:
            return shard
        if int(manifest.get("tokens_prepared", 0)) >= int(config["tokens"]["target_total_tokens"]):
            return None
        if producer is None:
            return None
        if producer is not None:
            returncode = producer.poll()
            if returncode is not None:
                append_jsonl(
                    Path(config["paths"]["train_log_file"]),
                    {
                        "event": "shard_producer_exited",
                        "time": utc_now(),
                        "returncode": returncode,
                    },
                )
                if rank0():
                    print(
                        f"Shard producer exited with code {returncode}; "
                        "building next shard synchronously.",
                        flush=True,
                    )
                return None
        time.sleep(poll_seconds)


def safe_delete_shard(shard_path: Path, shard_root: Path) -> None:
    root = shard_root.resolve()
    target = shard_path.resolve()
    if root not in target.parents and target != root:
        raise RuntimeError(f"Refusing to delete outside shard root: {target}")
    if target.exists():
        shutil.rmtree(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue pretrain LightLM on code shards.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--shard-producer", action="store_true")
    args = parser.parse_args()

    config = read_json(Path(args.config))
    configure_logging(config)
    if args.shard_producer:
        set_seed(int(config["seed"]), 0)
        tokenizer, _old_vocab_size = build_tokenizer(config)
        token_size = infer_token_size(len(tokenizer))
        shard_producer_loop(config, tokenizer, token_size)
        return

    if config["training"].get("compile", False):
        suppress_known_compile_warnings()
    device, rank, _local_rank, world_size = setup_distributed(config)
    initialize_dist_run_id()
    set_seed(int(config["seed"]), rank)
    torch.set_float32_matmul_precision("high")
    run_rank0_stage(
        config,
        "validate_source_access",
        lambda: validate_source_access(config, verify_hf_access=True),
    )

    tokenizer, model, _old_vocab_size = build_tokenizer_and_model(config, device)
    token_size = infer_token_size(len(tokenizer))

    if config["training"].get("compile", False) and device.type == "cuda":
        model = torch.compile(model)
    if is_dist():
        model = DDP(model, device_ids=[device.index])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        betas=tuple(float(x) for x in config["training"]["betas"]),
        weight_decay=float(config["training"]["weight_decay"]),
        fused=(device.type == "cuda"),
    )
    resume_state = load_resume_state(config)
    if resume_state.get("optimizer"):
        optimizer.load_state_dict(resume_state["optimizer"])

    def startup_rank0_stage() -> None:
        output_dir = Path(config["paths"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(output_dir / "tokenizer")
        expected_gpus = int(config["training"].get("expected_gpus", world_size))
        effective = (
            int(config["training"]["per_device_batch_size"])
            * int(config["training"]["max_seq_len"])
            * int(config["training"]["gradient_accumulation_steps"])
            * world_size
        )
        print(f"Device: {device}")
        print(f"World size: {world_size} (config expected {expected_gpus})")
        print(f"Tokens per optimizer step: {effective:,}")
        print(f"Token size for .ds files: {token_size} bytes")

        manifest_path = Path(config["paths"]["manifest_path"])
        manifest = load_manifest(manifest_path)
        if (
            config["evaluation"]["bigcode_harness"].get("enabled", False)
            and not manifest.get("initial_eval_done", False)
        ):
            eval_ok = run_bigcode_eval(
                config,
                tokenizer,
                model,
                int(manifest.get("global_step", 0)),
                int(manifest.get("tokens_trained", 0)),
                None,
                manifest,
                force=True,
            )
            if eval_ok:
                manifest["initial_eval_done"] = True
                print("Initial BigCode eval completed.")
            else:
                manifest["initial_eval_done"] = False
                print("Initial BigCode eval failed or was skipped; see eval log.")
            with locked_manifest(config) as current_manifest:
                current_manifest["initial_eval_done"] = manifest["initial_eval_done"]
                current_manifest["next_eval_tokens"] = manifest.get(
                    "next_eval_tokens",
                    current_manifest.get("next_eval_tokens", 0),
                )

    run_rank0_stage(config, "startup", startup_rank0_stage)

    parallel_shard_building = (
        bool(config["training"].get("parallel_shard_building", True))
        and not args.prepare_only
        and rank0()
    )
    shard_producer: Optional[subprocess.Popen] = None

    def start_shard_producer_stage() -> None:
        nonlocal shard_producer
        if parallel_shard_building:
            shard_producer = start_shard_producer(config, args.config)

    run_rank0_stage(config, "start_shard_producer", start_shard_producer_stage)

    try:
        while True:
            def prepare_or_mark_shard_stage() -> None:
                nonlocal shard_producer
                manifest = load_manifest(Path(config["paths"]["manifest_path"]))
                shard = next_shard_to_train(manifest)
                if shard is None and int(manifest["tokens_prepared"]) < int(config["tokens"]["target_total_tokens"]):
                    if parallel_shard_building:
                        shard = wait_for_produced_shard(config, shard_producer)
                        if shard is None:
                            stop_shard_producer(shard_producer)
                            shard_producer = None
                            shard = build_next_shard(config, tokenizer, token_size)
                            refreshed = load_manifest(Path(config["paths"]["manifest_path"]))
                            if int(refreshed.get("tokens_prepared", 0)) < int(config["tokens"]["target_total_tokens"]):
                                shard_producer = start_shard_producer(config, args.config)
                    else:
                        shard = build_next_shard(config, tokenizer, token_size)
                if shard is not None and not args.prepare_only:
                    mark_shard_status(config, int(shard["id"]), "training")

            run_rank0_stage(config, "prepare_or_mark_shard", prepare_or_mark_shard_stage)

            manifest = load_manifest(Path(config["paths"]["manifest_path"]))
            shard = next_shard_to_train(manifest)
            if shard is None:
                break
            if args.prepare_only:
                break

            global_step, tokens_trained, best_loss, loss_ema = train_shard(
                config,
                shard,
                tokenizer,
                model,
                optimizer,
                token_size,
                device,
                rank,
                world_size,
                resume_state,
            )
            resume_state = {
                "global_step": global_step,
                "tokens_trained": tokens_trained,
                "best_loss": best_loss,
                "loss_ema": loss_ema,
            }
            barrier()

            def finish_shard_stage() -> None:
                mark_shard_status(config, int(shard["id"]), "trained", trained_at=utc_now())
                if config["training"].get("delete_shard_after_train", True):
                    safe_delete_shard(Path(shard["path"]), Path(config["paths"]["shard_root"]))
                    mark_shard_status(config, int(shard["id"]), "deleted", deleted_at=utc_now())

            run_rank0_stage(config, f"finish_shard_{int(shard['id'])}", finish_shard_stage)
    finally:
        if rank0():
            stop_shard_producer(shard_producer)

    cleanup_distributed()


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
