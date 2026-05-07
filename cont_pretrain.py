import argparse
import gzip
import html
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from urllib.parse import unquote
from urllib.request import urlopen

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from datatrove.pipeline.tokens.tokenizer import TokenizedFile
from datatrove.utils.dataset import DatatroveFolderDataset
from huggingface_hub import hf_hub_download, list_repo_files
from torch.nn.parallel import DistributedDataParallel as DDP
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


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def log_rank0(message: str) -> None:
    if rank0():
        print(message, flush=True)


def get_hf_token(config: Dict[str, Any], source: Optional[Dict[str, Any]] = None) -> Optional[str]:
    if source and source.get("hf_token"):
        return source["hf_token"]
    return (
        config["data"].get("hf_token")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


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


def validate_source_access(config: Dict[str, Any]) -> None:
    for source in enabled_sources(config):
        if source.get("requires_hf_access") and not get_hf_token(config, source):
            raise RuntimeError(
                f"Source {source['id']} requires a Hugging Face token and accepted dataset terms. "
                "Run `huggingface-cli login` or `export HF_TOKEN=...`, accept the dataset terms on HF, "
                "or set this source to enabled=false in config.json."
            )
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
        dist.barrier()


def setup_distributed(config: Dict[str, Any]) -> Tuple[torch.device, int, int, int]:
    use_dist = config["training"].get("distributed", True) and "RANK" in os.environ
    if use_dist:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
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
        for prefix in ("module.", "_orig_mod."):
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


def build_tokenizer_and_model(config: Dict[str, Any], device: torch.device) -> Tuple[Any, Transformer, int]:
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["tokenizer_name"], use_fast=True)
    old_vocab_size = len(tokenizer)
    special_tokens = config["special_tokens"]["additional_special_tokens"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    tokenizer.pad_token = config["special_tokens"]["fim_pad"]

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


def load_hf_stream(source: Dict[str, Any], config: Dict[str, Any], seed: int, skip: int):
    log_rank0(
        f"Opening source {source['id']} "
        f"dataset={source['dataset']} split={source.get('split', 'train')} skip={skip:,}"
    )
    kwargs = {
        "split": source.get("split", "train"),
        "streaming": True,
    }
    if source.get("data_dir"):
        kwargs["data_dir"] = source["data_dir"]
    if source.get("data_files"):
        kwargs["data_files"] = source["data_files"]
    if source.get("trust_remote_code"):
        kwargs["trust_remote_code"] = True
    hf_token = get_hf_token(config, source)
    if hf_token:
        kwargs["token"] = hf_token

    dataset = load_dataset(source["dataset"], **kwargs)
    shuffle_buffer = int(source.get("shuffle_buffer", config["data"].get("stream_shuffle_buffer", 0)))
    if shuffle_buffer > 0:
        log_rank0(f"Shuffling source {source['id']} with buffer_size={shuffle_buffer:,}")
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)
    if skip > 0:
        dataset = dataset.skip(skip)
    return dataset


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

    metadata = notebook.get("metadata", {})
    language_info = metadata.get("language_info", {})
    kernelspec = metadata.get("kernelspec", {})
    language = normalize_language(
        language_info.get("name") or kernelspec.get("language") or kernelspec.get("name")
    )
    if language and "python" not in language:
        return ""

    cells = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
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

    a, b = sorted(rng.sample(range(1, len(tokens)), 2))
    prefix, middle, suffix = tokens[:a], tokens[a:b], tokens[b:]
    fim_prefix_id = tokenizer.convert_tokens_to_ids(config["special_tokens"]["fim_prefix"])
    fim_middle_id = tokenizer.convert_tokens_to_ids(config["special_tokens"]["fim_middle"])
    fim_suffix_id = tokenizer.convert_tokens_to_ids(config["special_tokens"]["fim_suffix"])

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
    rng = random.Random(int(config["seed"]) + shard_id)
    source_offsets = dict(manifest.get("source_offsets", {}))
    source_stats: Dict[str, Any] = {}
    tokens_written_total = 0
    shard_start_tokens = int(manifest.get("tokens_prepared", 0))
    log_rank0(
        f"Building shard {shard_id} at {shard_dir} "
        f"target={shard_target:,} tokens start={shard_start_tokens:,}"
    )

    for source in enabled_sources(config):
        target = int(targets.get(source["id"], 0))
        if target <= 0:
            continue
        skip = int(source_offsets.get(source["id"], 0))
        log_rank0(f"Source {source['id']} target={target:,} tokens")
        stream = load_hf_stream(source, config, int(config["seed"]) + shard_id, skip)
        swh_client = None
        if source.get("content_loader") == "stack_v2_swh":
            swh_transport = source.get("swh_transport", "https")
            log_rank0(
                f"Source {source['id']} uses SWH content fetches via {swh_transport}; "
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
        for row in stream:
            source_examples += 1
            for doc in row_to_documents(row, source, swh_client):
                if not keep_document(doc, source, config):
                    continue
                tokens, mode = tokenize_with_fim(
                    doc,
                    shard_start_tokens + tokens_written_total,
                    rng,
                    config,
                    tokenizer,
                )
                tokens.append(tokenizer.eos_token_id)
                writer.write(mode, tokens)
                source_tokens += len(tokens)
                source_docs += 1
                tokens_written_total += len(tokens)
                progress_interval = int(config["tokens"].get("progress_log_tokens", 1_000_000))
                if (
                    source_docs == 1
                    or source_tokens // progress_interval != (source_tokens - len(tokens)) // progress_interval
                ):
                    log_rank0(
                        f"Source {source['id']} progress: "
                        f"{source_tokens:,}/{target:,} tokens, docs={source_docs:,}, "
                        f"shard_total={tokens_written_total:,}"
                    )
                if source_tokens >= target:
                    break
            if source_tokens >= target:
                break

        source_offsets[source["id"]] = skip + source_examples
        source_stats[source["id"]] = {
            "target_tokens": target,
            "actual_tokens": source_tokens,
            "documents": source_docs,
            "examples_consumed": source_examples,
            "offset": source_offsets[source["id"]],
        }
        log_rank0(
            f"Finished source {source['id']}: "
            f"{source_tokens:,}/{target:,} tokens from {source_docs:,} docs "
            f"({source_examples:,} streamed examples)"
        )

    writer_stats = writer.close()
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

    manifest["tokens_prepared"] = shard_start_tokens + tokens_written_total
    manifest["source_offsets"] = source_offsets
    manifest["updated_at"] = utc_now()
    manifest["shards"].append(shard_stats)
    write_json(Path(config["paths"]["manifest_path"]), manifest)
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
            folder_path=str(shard_dir),
            filename_pattern=os.path.join(str(shard_dir), "**", "*.ds"),
            seq_len=int(config["training"]["max_seq_len"]),
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
            modes.append(mode_from_path(self.dataset.current_file_path))
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
) -> Tuple[torch.Tensor, Dict[str, Tuple[float, int]]]:
    logits, _, _ = model(x, targets=None, positions=positions)
    vocab = logits.shape[-1]
    token_loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1), reduction="none").view(y.shape)
    if loss_mask is not None:
        token_loss = token_loss * loss_mask.to(token_loss.dtype)
        valid_tokens = loss_mask.sum(dim=1).clamp_min(1).to(token_loss.dtype)
        sample_loss = token_loss.sum(dim=1) / valid_tokens
    else:
        sample_loss = token_loss.mean(dim=1)
    loss = sample_loss.mean()
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


def save_checkpoint(
    config: Dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    tokens_trained: int,
    shard_id: int,
    shard_microbatch: int,
) -> Path:
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if isinstance(model, DDP) else model
    path = checkpoint_dir / f"cpt_step_{global_step:08d}.pt"
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "tokens_trained": tokens_trained,
            "shard_id": shard_id,
            "shard_microbatch": shard_microbatch,
            "saved_at": utc_now(),
        },
        path,
    )
    prune_checkpoints(checkpoint_dir, int(config["training"].get("max_checkpoints_to_keep", 0)))
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


def export_eval_model(config: Dict[str, Any], tokenizer: Any, model: torch.nn.Module) -> str:
    export_dir = Path(config["paths"]["output_dir"]) / "eval_model"
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
        "use_cache": False,
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
    shutil.copy2(Path(__file__).with_name("hf_lightlm.py"), export_dir / "hf_lightlm.py")
    shutil.copy2(Path(__file__).with_name("model.py"), export_dir / "model.py")
    return str(export_dir)


def run_bigcode_eval(
    config: Dict[str, Any],
    tokenizer: Any,
    model: torch.nn.Module,
    global_step: int,
    tokens_trained: int,
    checkpoint_path: Optional[Path],
    manifest: Dict[str, Any],
    force: bool = False,
) -> None:
    eval_cfg = config["evaluation"]["bigcode_harness"]
    if not eval_cfg.get("enabled", False):
        return
    next_eval = int(manifest.get("next_eval_tokens") or eval_cfg.get("interval_tokens", 0))
    if not force and tokens_trained < next_eval:
        return

    eval_model = eval_cfg.get("eval_model", "{output_dir}")
    output_dir = config["paths"]["output_dir"]
    eval_model = eval_model.format(output_dir=output_dir, step=global_step, tokens=tokens_trained)
    if eval_model == output_dir:
        eval_model = export_eval_model(config, tokenizer, model)

    command = [
        part.format(
            output_dir=output_dir,
            step=global_step,
            tokens=tokens_trained,
            checkpoint=str(checkpoint_path or ""),
            eval_model=eval_model,
        )
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
            "time": utc_now(),
        },
    )
    if not command:
        append_jsonl(log_path, {"event": "bigcode_eval_skipped", "reason": "empty command", "time": utc_now()})
        return

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
        return
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "bigcode_eval_stdout_begin", "step": global_step}) + "\n")
        result = subprocess.run(command, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, text=True, check=False)
        f.write(json.dumps({"event": "bigcode_eval_stdout_end", "step": global_step}) + "\n")
    append_jsonl(
        log_path,
        {
            "event": "bigcode_eval_end",
            "step": global_step,
            "tokens_trained": tokens_trained,
            "returncode": result.returncode,
            "time": utc_now(),
        },
    )
    interval = int(eval_cfg["interval_tokens"])
    if force and next_eval <= tokens_trained:
        next_eval = tokens_trained + interval
    else:
        while next_eval <= tokens_trained:
            next_eval += interval
    manifest["next_eval_tokens"] = next_eval


def mark_shard_status(config: Dict[str, Any], shard_id: int, status: str, **extra: Any) -> None:
    manifest_path = Path(config["paths"]["manifest_path"])
    manifest = load_manifest(manifest_path)
    for shard in manifest["shards"]:
        if int(shard["id"]) == int(shard_id):
            shard["status"] = status
            shard.update(extra)
            break
    manifest["updated_at"] = utc_now()
    write_json(manifest_path, manifest)


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
) -> Tuple[int, int]:
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
        return global_step, tokens_trained

    dtype = get_dtype(tcfg["dtype"])
    use_amp = device.type == "cuda"
    grad_accum = int(tcfg["gradient_accumulation_steps"])
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
                    loss, metrics = compute_loss_by_mode(model, x, y, modes, positions, loss_mask)
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

        if rank0() and (global_step % checkpoint_interval == 0):
            last_checkpoint = save_checkpoint(
                config, model, optimizer, global_step, tokens_trained, int(shard["id"]), microbatch
            )
            manifest = load_manifest(Path(config["paths"]["manifest_path"]))
            manifest["global_step"] = global_step
            manifest["tokens_trained"] = tokens_trained
            run_bigcode_eval(config, tokenizer, model, global_step, tokens_trained, last_checkpoint, manifest)
            write_json(Path(config["paths"]["manifest_path"]), manifest)

    if rank0():
        last_checkpoint = save_checkpoint(
            config, model, optimizer, global_step, tokens_trained, int(shard["id"]), microbatch
        )
        manifest = load_manifest(Path(config["paths"]["manifest_path"]))
        manifest["global_step"] = global_step
        manifest["tokens_trained"] = tokens_trained
        run_bigcode_eval(config, tokenizer, model, global_step, tokens_trained, last_checkpoint, manifest)
        write_json(Path(config["paths"]["manifest_path"]), manifest)
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
    return global_step, tokens_trained


def next_shard_to_train(manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for shard in manifest.get("shards", []):
        if shard.get("status") in {"built", "training"}:
            return shard
    return None


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
    args = parser.parse_args()

    config = read_json(Path(args.config))
    device, rank, _local_rank, world_size = setup_distributed(config)
    set_seed(int(config["seed"]), rank)
    torch.set_float32_matmul_precision("high")
    validate_source_access(config)

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

    if rank0():
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
            run_bigcode_eval(
                config,
                tokenizer,
                model,
                int(manifest.get("global_step", 0)),
                int(manifest.get("tokens_trained", 0)),
                None,
                manifest,
                force=True,
            )
            manifest["initial_eval_done"] = True
            manifest["updated_at"] = utc_now()
            write_json(manifest_path, manifest)
            print("Initial BigCode eval completed or logged as skipped.")
    barrier()

    while True:
        if rank0():
            manifest_path = Path(config["paths"]["manifest_path"])
            manifest = load_manifest(manifest_path)
            shard = next_shard_to_train(manifest)
            if shard is None and int(manifest["tokens_prepared"]) < int(config["tokens"]["target_total_tokens"]):
                shard = build_shard(config, manifest, len(manifest["shards"]), tokenizer, token_size)
            if shard is not None and not args.prepare_only:
                mark_shard_status(config, int(shard["id"]), "training")
        barrier()

        manifest = load_manifest(Path(config["paths"]["manifest_path"]))
        shard = next_shard_to_train(manifest)
        if shard is None:
            break
        if args.prepare_only:
            break

        train_shard(
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
        resume_state = {}
        barrier()

        if rank0():
            manifest = load_manifest(Path(config["paths"]["manifest_path"]))
            mark_shard_status(config, int(shard["id"]), "trained", trained_at=utc_now())
            if config["training"].get("delete_shard_after_train", True):
                safe_delete_shard(Path(shard["path"]), Path(config["paths"]["shard_root"]))
                mark_shard_status(config, int(shard["id"]), "deleted", deleted_at=utc_now())
        barrier()

    cleanup_distributed()


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
