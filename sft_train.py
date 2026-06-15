import argparse
import copy
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP

from cont_pretrain import (
    _DOCSTRING_AFTER_DEF,
    BEST_SCORE_EPS,
    append_jsonl,
    barrier,
    build_tokenizer_and_model,
    cleanup_distributed,
    clean_subprocess_distributed_env,
    configure_logging,
    export_eval_model,
    finite_score,
    get_dtype,
    init_special_embeddings,
    is_dist,
    load_resume_state,
    load_state_into_model,
    lr_for_step,
    optimizer_state_is_compatible,
    rank0,
    read_json,
    reduce_mode_metrics,
    run_bigcode_eval,
    run_rank0_stage,
    score_better,
    score_equal,
    set_optimizer_lr,
    set_seed,
    setup_distributed,
    suppress_known_compile_warnings,
    utc_now,
    write_json,
)


SFT_MODES = {
    "instruction": 0,
    "raw_completion": 1,
    "docstring_completion": 2,
}
SFT_MODE_NAMES = {v: k for k, v in SFT_MODES.items()}
IGNORE_INDEX = -100


def source_access_kwargs(source: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
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
    token = (
        source.get("hf_token")
        or config.get("data", {}).get("hf_token")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if token:
        kwargs["token"] = token
    return kwargs


def load_source_dataset(source: Dict[str, Any], config: Dict[str, Any]):
    dataset_name = source.get("dataset")
    split = source.get("split", "train")
    streaming = bool(source.get("streaming", config.get("sft", {}).get("streaming", True)))
    kwargs = source_access_kwargs(source, config)
    if dataset_name == "json" or (not dataset_name and source.get("data_files")):
        return load_dataset("json", split=split, streaming=streaming, **kwargs)
    if not dataset_name:
        raise ValueError(f"SFT source {source.get('id')} needs dataset or data_files.")
    return load_dataset(dataset_name, split=split, streaming=streaming, **kwargs)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def first_text(row: Dict[str, Any], fields: Iterable[str]) -> str:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def markdown_code_unwrap(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def normalize_role(role: Any) -> str:
    value = clean_text(role).lower()
    if value in {"human", "user", "prompter", "instruction"}:
        return "user"
    if value in {"gpt", "assistant", "bot", "model", "response"}:
        return "assistant"
    if value == "system":
        return "system"
    return value or "user"


def extract_message_content(message: Dict[str, Any]) -> str:
    for key in ("content", "value", "text", "message"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_messages(row: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    for field in ("messages", "conversations", "conversation", "data"):
        value = row.get(field)
        if not isinstance(value, list):
            continue
        messages = []
        for item in value:
            if not isinstance(item, dict):
                continue
            role = normalize_role(item.get("role", item.get("from", item.get("speaker"))))
            content = extract_message_content(item)
            if content:
                messages.append({"role": role, "content": content})
        if any(message["role"] == "assistant" for message in messages):
            return messages
    return None


def best_ultrafeedback_response(row: Dict[str, Any]) -> str:
    responses = row.get("responses")
    if not isinstance(responses, list) or not responses:
        return ""
    annotations = row.get("annotations")
    ratings_by_model: Dict[str, float] = {}
    if isinstance(annotations, list):
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            model = clean_text(annotation.get("model"))
            try:
                rating = float(annotation.get("rating"))
            except (TypeError, ValueError):
                continue
            if model:
                ratings_by_model[model] = rating

    best_response = ""
    best_rating = float("-inf")
    for index, response in enumerate(responses):
        if isinstance(response, dict):
            text = clean_text(response.get("response") or response.get("content") or response.get("text"))
            model = clean_text(response.get("model"))
        else:
            text = clean_text(response)
            model = ""
        rating = ratings_by_model.get(model, float(len(responses) - index) * 1e-6)
        if text and rating > best_rating:
            best_response = text
            best_rating = rating
    return best_response


def extract_prompt_completion(row: Dict[str, Any], source: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    prompt_field = source.get("prompt_field")
    completion_field = source.get("completion_field")
    if prompt_field and completion_field:
        prompt = clean_text(row.get(prompt_field))
        completion = clean_text(row.get(completion_field))
        if prompt and completion:
            return prompt, completion

    prompt = first_text(
        row,
        (
            "instruction",
            "input",
            "problem",
            "question",
            "prompt",
            "query",
            "task",
        ),
    )
    extra_input = first_text(row, ("context", "additional_context"))
    if prompt and extra_input and extra_input not in prompt:
        prompt = f"{prompt}\n\n{extra_input}"

    completion = first_text(
        row,
        (
            "output",
            "response",
            "solution",
            "answer",
            "completion",
            "accepted_solution",
        ),
    )
    if not completion:
        completion = best_ultrafeedback_response(row)
    if prompt and completion:
        return prompt, completion
    return None


def encode_piece(tokenizer: Any, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def encode_instruction_messages(tokenizer: Any, messages: List[Dict[str, str]]) -> Optional[Tuple[List[int], List[int]]]:
    ids: List[int] = []
    labels: List[int] = []
    for message in messages:
        role = message["role"]
        if role not in {"system", "user", "assistant"}:
            role = "user"
        header = f"<|im_start|>{role}\n"
        footer = "<|im_end|>\n"
        header_ids = encode_piece(tokenizer, header)
        content_ids = encode_piece(tokenizer, message["content"])
        footer_ids = encode_piece(tokenizer, footer)
        ids.extend(header_ids)
        labels.extend([IGNORE_INDEX] * len(header_ids))
        ids.extend(content_ids)
        if role == "assistant":
            labels.extend(content_ids)
        else:
            labels.extend([IGNORE_INDEX] * len(content_ids))
        ids.extend(footer_ids)
        labels.extend(footer_ids if role == "assistant" else [IGNORE_INDEX] * len(footer_ids))
    if not any(label != IGNORE_INDEX for label in labels):
        return None
    return ids, labels


def encode_prompt_completion(tokenizer: Any, prompt: str, completion: str) -> Optional[Tuple[List[int], List[int]]]:
    prompt_text = f"<|im_start|>user\n{prompt.strip()}\n<|im_end|>\n<|im_start|>assistant\n"
    completion_text = completion.strip() + "\n<|im_end|>\n"
    prompt_ids = encode_piece(tokenizer, prompt_text)
    completion_ids = encode_piece(tokenizer, completion_text)
    if not prompt_ids or not completion_ids:
        return None
    return prompt_ids + completion_ids, [IGNORE_INDEX] * len(prompt_ids) + completion_ids


def format_raw_completion(tokenizer: Any, text: str) -> Optional[Tuple[List[int], List[int]]]:
    ids = encode_piece(tokenizer, text.strip())
    if len(ids) < 8:
        return None
    return ids, ids.copy()


def format_docstring_completion(tokenizer: Any, text: str, rng: random.Random) -> Optional[Tuple[List[int], List[int]]]:
    matches = list(_DOCSTRING_AFTER_DEF.finditer(text))
    if not matches:
        return None
    match = rng.choice(matches)
    split_at = match.end()
    prompt = text[:split_at].rstrip() + "\n"
    completion = text[split_at:].strip()
    if len(completion) < 16:
        return None
    prompt_ids = encode_piece(tokenizer, prompt)
    completion_ids = encode_piece(tokenizer, completion)
    if not prompt_ids or len(completion_ids) < 4:
        return None
    return prompt_ids + completion_ids, [IGNORE_INDEX] * len(prompt_ids) + completion_ids


def truncate_example(
    ids: List[int],
    labels: List[int],
    max_tokens: int,
    mode: str,
    rng: random.Random,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
    if len(ids) != len(labels):
        raise ValueError("ids and labels length mismatch")
    max_len = max_tokens + 1
    if len(ids) > max_len:
        if mode == "raw_completion":
            start = rng.randint(0, len(ids) - max_len)
        else:
            label_positions = [i for i, label in enumerate(labels) if label != IGNORE_INDEX]
            if not label_positions:
                return None
            if mode == "docstring_completion":
                start = 0
            else:
                label_pos = rng.choice(label_positions)
                start = max(0, min(label_pos - max_len // 2, len(ids) - max_len))
        ids = ids[start : start + max_len]
        labels = labels[start : start + max_len]
    if len(ids) < 2 or not any(label != IGNORE_INDEX for label in labels[1:]):
        return None
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        SFT_MODES[mode],
    )


def format_sft_example(
    row: Dict[str, Any],
    source: Dict[str, Any],
    tokenizer: Any,
    max_tokens: int,
    rng: random.Random,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
    group = source.get("group", "instruction")
    mode = str(source.get("format", group))

    if group == "raw_completion":
        text_field = source.get("text_field")
        fields = (text_field,) if isinstance(text_field, str) else ("content", "text", "code", "solution")
        text = first_text(row, fields)
        if not text:
            pair = extract_prompt_completion(row, source)
            if pair:
                text = "\n\n".join(pair)
        if not text:
            return None
        if rng.random() < float(source.get("docstring_probability", 0.0)):
            encoded = format_docstring_completion(tokenizer, text, rng)
            if encoded is not None:
                return truncate_example(*encoded, max_tokens, "docstring_completion", rng)
        encoded = format_raw_completion(tokenizer, text)
        if encoded is None:
            return None
        return truncate_example(*encoded, max_tokens, "raw_completion", rng)

    messages = extract_messages(row)
    if messages is not None:
        encoded = encode_instruction_messages(tokenizer, messages)
    else:
        pair = extract_prompt_completion(row, source)
        encoded = encode_prompt_completion(tokenizer, *pair) if pair else None
    if encoded is None:
        return None
    return truncate_example(*encoded, max_tokens, "instruction", rng)


class SFTSourceIterator:
    def __init__(self, source: Dict[str, Any], config: Dict[str, Any], tokenizer: Any, rank: int):
        self.source = source
        self.config = config
        self.tokenizer = tokenizer
        self.rank = rank
        self.epoch = 0
        self.rng = random.Random(int(config["seed"]) + rank + stable_source_seed(source["id"]))
        self.iterator: Optional[Iterator[Dict[str, Any]]] = None
        self.reset()

    def reset(self) -> None:
        dataset = load_source_dataset(self.source, self.config)
        if hasattr(dataset, "shuffle"):
            buffer_size = int(self.config.get("sft", {}).get("stream_shuffle_buffer", 10_000))
            dataset = dataset.shuffle(seed=int(self.config["seed"]) + self.rank + self.epoch, buffer_size=buffer_size)
        self.iterator = iter(dataset)
        self.epoch += 1

    def next(self, max_tokens: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        assert self.iterator is not None
        max_bad = int(self.source.get("max_bad_rows", 10_000))
        bad_rows = 0
        while True:
            try:
                row = next(self.iterator)
            except StopIteration:
                if not bool(self.config.get("sft", {}).get("cycle_on_exhaustion", True)):
                    raise
                self.reset()
                continue
            example = format_sft_example(row, self.source, self.tokenizer, max_tokens, self.rng)
            if example is not None:
                return example
            bad_rows += 1
            if bad_rows >= max_bad:
                raise RuntimeError(
                    f"SFT source {self.source['id']} produced {bad_rows} unusable rows. "
                    "Check its format/prompt/completion/text fields."
                )


def stable_source_seed(value: str) -> int:
    total = 0
    for char in value:
        total = (total * 131 + ord(char)) % (2**31 - 1)
    return total


class SFTMixture:
    def __init__(self, config: Dict[str, Any], tokenizer: Any, rank: int):
        self.config = config
        self.tokenizer = tokenizer
        self.rank = rank
        self.max_tokens = int(config["training"]["max_seq_len"])
        self.rng = random.Random(int(config["seed"]) + rank * 9973)
        sources = [source for source in config["sft"]["sources"] if source.get("enabled", True)]
        if not sources:
            raise RuntimeError("No enabled SFT sources.")
        self.group_mix = dict(config["sft"].get("group_mix", {"instruction": 0.7, "raw_completion": 0.3}))
        self.sources_by_group: Dict[str, List[Tuple[SFTSourceIterator, float]]] = {}
        for source in sources:
            group = str(source.get("group", "instruction"))
            self.sources_by_group.setdefault(group, []).append(
                (SFTSourceIterator(source, config, tokenizer, rank), float(source.get("weight", 1.0)))
            )
        for group in list(self.group_mix):
            if group not in self.sources_by_group:
                self.group_mix.pop(group)
        if not self.group_mix:
            raise RuntimeError("SFT group_mix has no enabled sources.")

    def choose_group(self) -> str:
        groups = list(self.group_mix)
        weights = [max(float(self.group_mix[group]), 0.0) for group in groups]
        return self.rng.choices(groups, weights=weights, k=1)[0]

    def choose_source(self, group: str) -> SFTSourceIterator:
        items = self.sources_by_group[group]
        sources = [item[0] for item in items]
        weights = [max(item[1], 0.0) for item in items]
        return self.rng.choices(sources, weights=weights, k=1)[0]

    def next_example(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        group = self.choose_group()
        return self.choose_source(group).next(self.max_tokens)


def collate_sft_batch(
    examples: List[Tuple[torch.Tensor, torch.Tensor, int]],
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(int(ids.numel()) for ids, _, _ in examples)
    batch_ids = torch.full((len(examples), max_len), pad_token_id, dtype=torch.long)
    batch_labels = torch.full((len(examples), max_len), IGNORE_INDEX, dtype=torch.long)
    modes = torch.empty(len(examples), dtype=torch.long)
    for idx, (ids, labels, mode) in enumerate(examples):
        length = int(ids.numel())
        batch_ids[idx, :length] = ids
        batch_labels[idx, :length] = labels
        modes[idx] = mode
    x = batch_ids[:, :-1]
    y = batch_ids[:, 1:]
    label_targets = batch_labels[:, 1:]
    loss_mask = label_targets.ne(IGNORE_INDEX)
    y = torch.where(loss_mask, label_targets, y)
    return x, y, loss_mask, modes


def next_sft_batch(
    mixture: SFTMixture,
    batch_size: int,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return collate_sft_batch([mixture.next_example() for _ in range(batch_size)], pad_token_id)


def compute_sft_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Tuple[float, int]]]:
    logits, _, aux_loss = model(x, targets=None)
    vocab = logits.shape[-1]
    token_loss = F.cross_entropy(
        logits.reshape(-1, vocab),
        y.reshape(-1),
        reduction="none",
    ).view(y.shape)
    mask = loss_mask.to(token_loss.dtype)
    sample_den = mask.sum(dim=1).clamp_min(1)
    sample_loss = (token_loss * mask).sum(dim=1) / sample_den
    loss = sample_loss.mean()
    if torch.is_tensor(aux_loss):
        loss = loss + aux_loss
    return loss, {}


def reduce_sft_metrics(
    per_mode: Dict[str, Tuple[float, int]],
    device: torch.device,
) -> Dict[str, float]:
    values: List[float] = []
    for mode in SFT_MODES:
        loss_sum, count = per_mode.get(mode, (0.0, 0))
        values.extend([loss_sum, float(count)])
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    if is_dist():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    out: Dict[str, float] = {}
    for i, mode in enumerate(SFT_MODES):
        loss_sum = tensor[2 * i].item()
        count = tensor[2 * i + 1].item()
        out[f"loss_{mode}"] = loss_sum / max(count, 1.0)
        out[f"count_{mode}"] = int(count)
    return out


def compute_sft_loss_by_mode(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_mask: torch.Tensor,
    modes: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Tuple[float, int]]]:
    logits, _, aux_loss = model(x, targets=None)
    vocab = logits.shape[-1]
    token_loss = F.cross_entropy(
        logits.reshape(-1, vocab),
        y.reshape(-1),
        reduction="none",
    ).view(y.shape)
    mask = loss_mask.to(token_loss.dtype)
    valid = mask.sum(dim=1).clamp_min(1)
    sample_loss = (token_loss * mask).sum(dim=1) / valid
    loss = sample_loss.mean()
    if torch.is_tensor(aux_loss):
        loss = loss + aux_loss
    metrics: Dict[str, Tuple[float, int]] = {}
    for mode_id, mode_name in SFT_MODE_NAMES.items():
        mode_mask = modes == mode_id
        count = int(mode_mask.sum().item())
        if count:
            metrics[mode_name] = (float(sample_loss[mode_mask].sum().detach().item()), count)
        else:
            metrics[mode_name] = (0.0, 0)
    return loss, metrics


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    tokens_trained: int,
    best_loss: float,
    loss_ema: Optional[float],
    best_humaneval_score: Optional[float],
    best_humaneval_step: Optional[int],
    best_mbpp_score: Optional[float],
    best_mbpp_step: Optional[int],
    best_mbpp_humaneval_score: Optional[float],
) -> Dict[str, Any]:
    raw_model = model.module if isinstance(model, DDP) else model
    return {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "sft_step": global_step,
        "tokens_trained": tokens_trained,
        "sft_tokens_trained": tokens_trained,
        "best_loss": best_loss,
        "loss_ema": loss_ema,
        "best_humaneval_score": best_humaneval_score if best_humaneval_score is not None else None,
        "best_humaneval_step": best_humaneval_step,
        "best_mbpp_score": best_mbpp_score if best_mbpp_score is not None else None,
        "best_mbpp_step": best_mbpp_step,
        "best_mbpp_humaneval_score": best_mbpp_humaneval_score,
        "stage": "sft",
        "saved_at": utc_now(),
    }


def save_sft_checkpoint(
    config: Dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    tokens_trained: int,
    best_loss: float,
    loss_ema: Optional[float],
    best_humaneval_score: Optional[float],
    best_humaneval_step: Optional[int],
    best_mbpp_score: Optional[float],
    best_mbpp_step: Optional[int],
    best_mbpp_humaneval_score: Optional[float],
    filename: Optional[str] = None,
) -> Path:
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / (filename or f"sft_step_{global_step:08d}.pt")
    torch.save(
        checkpoint_payload(
            model,
            optimizer,
            global_step,
            tokens_trained,
            best_loss,
            loss_ema,
            best_humaneval_score,
            best_humaneval_step,
            best_mbpp_score,
            best_mbpp_step,
            best_mbpp_humaneval_score,
        ),
        path,
    )
    return path


def prune_sft_checkpoints(config: Dict[str, Any]) -> None:
    max_to_keep = int(config["training"].get("max_checkpoints_to_keep", 0))
    if max_to_keep <= 0:
        return
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoints = sorted(checkpoint_dir.glob("sft_step_*.pt"))
    keep_regular = max(max_to_keep - 3, 1)
    for checkpoint in checkpoints[:-keep_regular]:
        checkpoint.unlink(missing_ok=True)


def update_best_sft_checkpoints(
    config: Dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    tokens_trained: int,
    best_loss: float,
    loss_ema: Optional[float],
    eval_result: Dict[str, Any],
    train_log: Path,
    best_humaneval_score: float,
    best_humaneval_step: Optional[int],
    best_mbpp_score: float,
    best_mbpp_step: Optional[int],
    best_mbpp_humaneval_score: Optional[float],
) -> Tuple[float, Optional[int], float, Optional[int], Optional[float]]:
    if not eval_result.get("ok"):
        return best_humaneval_score, best_humaneval_step, best_mbpp_score, best_mbpp_step, best_mbpp_humaneval_score
    humaneval_score = finite_score(eval_result.get("score"))
    mbpp_score = finite_score(eval_result.get("tie_break_score"))
    if humaneval_score is None:
        return best_humaneval_score, best_humaneval_step, best_mbpp_score, best_mbpp_step, best_mbpp_humaneval_score

    save_mbpp = False
    if score_better(humaneval_score, best_humaneval_score):
        best_humaneval_score = humaneval_score
        best_humaneval_step = global_step
        best_mbpp_score = float("-inf")
        best_mbpp_step = None
        best_mbpp_humaneval_score = humaneval_score
        if mbpp_score is not None:
            best_mbpp_score = mbpp_score
            best_mbpp_step = global_step
            save_mbpp = True
        for filename in ("sft_best.pt", "sft_best_humaneval.pt"):
            save_sft_checkpoint(
                config,
                model,
                optimizer,
                global_step,
                tokens_trained,
                best_loss,
                loss_ema,
                best_humaneval_score,
                best_humaneval_step,
                best_mbpp_score,
                best_mbpp_step,
                best_mbpp_humaneval_score,
                filename=filename,
            )
        append_jsonl(
            train_log,
            {
                "event": "new_best_humaneval",
                "time": utc_now(),
                "step": global_step,
                "tokens_trained": tokens_trained,
                "score": best_humaneval_score,
                "scores": eval_result.get("scores"),
                "metric_output_path": eval_result.get("metric_output_path"),
                "checkpoint": "sft_best.pt",
            },
        )
        print(f"new best HumanEval {best_humaneval_score:.4f} at SFT step {global_step}")
    elif score_equal(humaneval_score, best_humaneval_score) and mbpp_score is not None:
        mbpp_tied_to_current_humaneval = score_equal(best_mbpp_humaneval_score, best_humaneval_score)
        mbpp_baseline = best_mbpp_score if mbpp_tied_to_current_humaneval else float("-inf")
        if score_better(mbpp_score, mbpp_baseline):
            best_mbpp_score = mbpp_score
            best_mbpp_step = global_step
            best_mbpp_humaneval_score = best_humaneval_score
            save_mbpp = True

    if save_mbpp:
        save_sft_checkpoint(
            config,
            model,
            optimizer,
            global_step,
            tokens_trained,
            best_loss,
            loss_ema,
            best_humaneval_score,
            best_humaneval_step,
            best_mbpp_score,
            best_mbpp_step,
            best_mbpp_humaneval_score,
            filename="sft_best_mbpp.pt",
        )
        append_jsonl(
            train_log,
            {
                "event": "new_best_mbpp_at_best_humaneval",
                "time": utc_now(),
                "step": global_step,
                "tokens_trained": tokens_trained,
                "humaneval_score": best_humaneval_score,
                "mbpp_score": best_mbpp_score,
                "scores": eval_result.get("scores"),
                "metric_output_path": eval_result.get("metric_output_path"),
                "checkpoint": "sft_best_mbpp.pt",
            },
        )
        print(
            f"new best MBPP {best_mbpp_score:.4f} among HumanEval={best_humaneval_score:.4f} "
            f"at SFT step {global_step}"
        )
    return best_humaneval_score, best_humaneval_step, best_mbpp_score, best_mbpp_step, best_mbpp_humaneval_score


def sft_manifest(config: Dict[str, Any]) -> Dict[str, Any]:
    path = Path(config["paths"].get("manifest_path", Path(config["paths"]["output_dir"]) / "sft_manifest.json"))
    if path.exists():
        return read_json(path)
    return {
        "version": 1,
        "stage": "sft",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "global_step": 0,
        "tokens_trained": 0,
        "next_eval_step": int(config["evaluation"]["bigcode_harness"].get("interval_steps", 0)),
    }


def write_sft_manifest(config: Dict[str, Any], manifest: Dict[str, Any]) -> None:
    manifest["updated_at"] = utc_now()
    write_json(Path(config["paths"].get("manifest_path", Path(config["paths"]["output_dir"]) / "sft_manifest.json")), manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT LightLM from a CPT checkpoint for code instruction/autocomplete.")
    parser.add_argument("--config", default="config_sft_ffn.json")
    args = parser.parse_args()

    config = read_json(Path(args.config))
    configure_logging(config)
    if config["training"].get("compile", False):
        suppress_known_compile_warnings()
    device, rank, _local_rank, world_size = setup_distributed(config)
    set_seed(int(config["seed"]), rank)
    torch.set_float32_matmul_precision("high")

    tokenizer, model, _ = build_tokenizer_and_model(config, device)
    if bool(config["training"].get("gradient_checkpointing", True)):
        model.gradient_checkpointing = True
    if config["training"].get("compile", False) and device.type == "cuda":
        model = torch.compile(model)
    if is_dist():
        model = DDP(model, device_ids=[device.index])

    tcfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["learning_rate"]),
        betas=tuple(float(x) for x in tcfg["betas"]),
        weight_decay=float(tcfg["weight_decay"]),
        fused=(device.type == "cuda"),
    )
    resume_state = load_resume_state(config)
    if bool(tcfg.get("resume_optimizer", False)) and resume_state.get("optimizer"):
        optimizer_ok, optimizer_reason = optimizer_state_is_compatible(optimizer, resume_state["optimizer"])
        if optimizer_ok:
            optimizer.load_state_dict(resume_state["optimizer"])
        elif rank0():
            print(f"Skipped optimizer resume: {optimizer_reason}")

    global_step = int(resume_state.get("sft_step", 0) or 0)
    tokens_trained = int(resume_state.get("sft_tokens_trained", 0) or 0)
    best_loss = float(resume_state.get("best_loss", resume_state.get("loss_ema", float("inf"))))
    loss_ema = resume_state.get("loss_ema")
    loss_ema = float(loss_ema) if loss_ema is not None else None
    best_humaneval_value = resume_state.get("best_humaneval_score")
    best_humaneval_score = float(best_humaneval_value) if best_humaneval_value is not None else float("-inf")
    best_humaneval_step_value = resume_state.get("best_humaneval_step")
    best_humaneval_step = int(best_humaneval_step_value) if best_humaneval_step_value is not None else None
    best_mbpp_value = resume_state.get("best_mbpp_score")
    best_mbpp_score = float(best_mbpp_value) if best_mbpp_value is not None else float("-inf")
    best_mbpp_step_value = resume_state.get("best_mbpp_step")
    best_mbpp_step = int(best_mbpp_step_value) if best_mbpp_step_value is not None else None
    best_mbpp_humaneval_value = resume_state.get("best_mbpp_humaneval_score")
    best_mbpp_humaneval_score = (
        float(best_mbpp_humaneval_value)
        if best_mbpp_humaneval_value is not None
        else (best_humaneval_score if math.isfinite(best_mbpp_score) else None)
    )

    train_log = Path(config["paths"]["train_log_file"])
    output_dir = Path(config["paths"]["output_dir"])
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    if rank0():
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(output_dir / "tokenizer")
        print(f"SFT device={device} world_size={world_size}")
        print(f"SFT resume checkpoint={resume_state.get('checkpoint_path')}")
        print(f"SFT max_seq_len={tcfg['max_seq_len']} per_device_batch_size={tcfg['per_device_batch_size']}")

    mixture = SFTMixture(config, tokenizer, rank)
    dtype = get_dtype(tcfg["dtype"])
    use_amp = device.type == "cuda"
    grad_accum = int(tcfg["gradient_accumulation_steps"])
    batch_size = int(tcfg["per_device_batch_size"])
    max_steps = int(tcfg["max_steps"])
    checkpoint_interval = int(tcfg["checkpoint_interval_steps"])
    eval_interval = int(config["evaluation"]["bigcode_harness"].get("interval_steps", checkpoint_interval))
    log_interval = int(tcfg["log_interval_steps"])
    loss_ema_alpha = float(tcfg.get("best_loss_ema_alpha", 0.1))
    global_tokens_per_step = batch_size * int(tcfg["max_seq_len"]) * grad_accum * world_size
    manifest = sft_manifest(config)

    def run_eval_stage(name: str, force: bool = False) -> Dict[str, Any]:
        nonlocal best_humaneval_score, best_humaneval_step
        nonlocal best_mbpp_score, best_mbpp_step, best_mbpp_humaneval_score
        nonlocal manifest
        result: Dict[str, Any] = {"ok": True, "skipped": True, "score": None}

        def rank0_eval() -> None:
            nonlocal best_humaneval_score, best_humaneval_step
            nonlocal best_mbpp_score, best_mbpp_step, best_mbpp_humaneval_score
            nonlocal manifest, result
            checkpoint_path = save_sft_checkpoint(
                config,
                model,
                optimizer,
                global_step,
                tokens_trained,
                best_loss,
                loss_ema,
                best_humaneval_score,
                best_humaneval_step,
                best_mbpp_score,
                best_mbpp_step,
                best_mbpp_humaneval_score,
            )
            prune_sft_checkpoints(config)
            manifest["global_step"] = global_step
            manifest["tokens_trained"] = tokens_trained
            result = run_bigcode_eval(
                config,
                tokenizer,
                model,
                global_step,
                tokens_trained,
                checkpoint_path,
                manifest,
                force=force,
            )
            (
                best_humaneval_score,
                best_humaneval_step,
                best_mbpp_score,
                best_mbpp_step,
                best_mbpp_humaneval_score,
            ) = update_best_sft_checkpoints(
                config,
                model,
                optimizer,
                global_step,
                tokens_trained,
                best_loss,
                loss_ema,
                result,
                train_log,
                best_humaneval_score,
                best_humaneval_step,
                best_mbpp_score,
                best_mbpp_step,
                best_mbpp_humaneval_score,
            )
            manifest["best_humaneval_score"] = best_humaneval_score if math.isfinite(best_humaneval_score) else None
            manifest["best_humaneval_step"] = best_humaneval_step
            manifest["best_mbpp_score"] = best_mbpp_score if math.isfinite(best_mbpp_score) else None
            manifest["best_mbpp_step"] = best_mbpp_step
            manifest["best_mbpp_humaneval_score"] = best_mbpp_humaneval_score
            write_sft_manifest(config, manifest)

        run_rank0_stage(config, name, rank0_eval)
        manifest = sft_manifest(config)
        best_humaneval_value = manifest.get("best_humaneval_score")
        best_humaneval_score = float(best_humaneval_value) if best_humaneval_value is not None else best_humaneval_score
        best_humaneval_step = manifest.get("best_humaneval_step", best_humaneval_step)
        best_mbpp_value = manifest.get("best_mbpp_score")
        best_mbpp_score = float(best_mbpp_value) if best_mbpp_value is not None else best_mbpp_score
        best_mbpp_step = manifest.get("best_mbpp_step", best_mbpp_step)
        best_mbpp_humaneval_value = manifest.get("best_mbpp_humaneval_score")
        best_mbpp_humaneval_score = (
            float(best_mbpp_humaneval_value)
            if best_mbpp_humaneval_value is not None
            else best_mbpp_humaneval_score
        )
        return result

    if config["evaluation"]["bigcode_harness"].get("initial_eval_enabled", True) and global_step == 0:
        run_eval_stage("sft_initial_eval", force=True)

    model.train()
    start_time = time.perf_counter()
    while global_step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        step_metrics = {mode: (0.0, 0) for mode in SFT_MODES}
        step_start = time.perf_counter()
        lr = lr_for_step(global_step, max_steps, config)
        set_optimizer_lr(optimizer, lr)
        for accum_idx in range(grad_accum):
            x, y, loss_mask, modes = next_sft_batch(mixture, batch_size, tokenizer.pad_token_id)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss_mask = loss_mask.to(device, non_blocking=True)
            modes = modes.to(device, non_blocking=True)
            sync_context = model.no_sync() if isinstance(model, DDP) and accum_idx < grad_accum - 1 else nullcontext()
            amp_context = torch.autocast(device_type=device.type, dtype=dtype) if use_amp else nullcontext()
            with sync_context:
                with amp_context:
                    loss, metrics = compute_sft_loss_by_mode(model, x, y, loss_mask, modes)
                    (loss / grad_accum).backward()
            for mode, (loss_sum, count) in metrics.items():
                prev_sum, prev_count = step_metrics[mode]
                step_metrics[mode] = (prev_sum + loss_sum, prev_count + count)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
        optimizer.step()
        global_step += 1
        tokens_trained += global_tokens_per_step
        reduced = reduce_sft_metrics(step_metrics, device)
        elapsed = time.perf_counter() - step_start
        toks_per_sec = global_tokens_per_step / max(elapsed, 1e-6)
        count_all = sum(reduced[f"count_{mode}"] for mode in SFT_MODES)
        if count_all > 0:
            combined = sum(reduced[f"loss_{mode}"] * reduced[f"count_{mode}"] for mode in SFT_MODES) / count_all
            loss_ema = combined if loss_ema is None else (1 - loss_ema_alpha) * loss_ema + loss_ema_alpha * combined
            best_loss = min(best_loss, loss_ema)

        if rank0() and (global_step % log_interval == 0 or global_step == 1):
            row = {
                "event": "sft_train_step",
                "time": utc_now(),
                "step": global_step,
                "tokens_trained": tokens_trained,
                "lr": lr,
                "grad_norm": float(grad_norm),
                "tokens_per_second": toks_per_sec,
                "elapsed_seconds": time.perf_counter() - start_time,
                **reduced,
            }
            append_jsonl(train_log, row)
            print(
                f"sft step {global_step} "
                f"loss instruction={reduced['loss_instruction']:.4f} "
                f"raw={reduced['loss_raw_completion']:.4f} "
                f"docstring={reduced['loss_docstring_completion']:.4f} "
                f"lr={lr:.2e} tok/s={toks_per_sec:.0f}"
            )

        if global_step % checkpoint_interval == 0:
            if rank0():
                save_sft_checkpoint(
                    config,
                    model,
                    optimizer,
                    global_step,
                    tokens_trained,
                    best_loss,
                    loss_ema,
                    best_humaneval_score,
                    best_humaneval_step,
                    best_mbpp_score,
                    best_mbpp_step,
                    best_mbpp_humaneval_score,
                )
                prune_sft_checkpoints(config)
            barrier()

        if eval_interval > 0 and global_step % eval_interval == 0:
            run_eval_stage(f"sft_eval_step_{global_step}", force=True)
            model.train()

    if config["evaluation"]["bigcode_harness"].get("final_eval_enabled", True):
        run_eval_stage("sft_final_eval", force=True)
    cleanup_distributed()


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
