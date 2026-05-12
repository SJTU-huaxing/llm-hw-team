"""Shared utilities for datasets, model loading, metrics, and result files."""

from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_MODEL = "EleutherAI/pythia-70m"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def preferred_dtype(device: torch.device, dtype_name: str = "float32") -> torch.dtype:
    if dtype_name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype {dtype_name!r}.")
    return mapping[dtype_name]


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL,
    device: torch.device | None = None,
    dtype_name: str = "float32",
    attn_implementation: str = "eager",
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = resolve_device("auto")
    dtype = preferred_dtype(device, dtype_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        )
    model.to(device)
    model.eval()
    return model, tokenizer


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: Iterable[str]) -> str:
    columns = list(columns)
    if not rows:
        return "(no rows)"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _extract_text(row: dict[str, Any]) -> str:
    if isinstance(row.get("text"), str):
        return row["text"]
    strings = [value for value in row.values() if isinstance(value, str)]
    return max(strings, key=len) if strings else ""


def _fallback_text(dataset_name: str) -> str:
    base = (
        "KV cache compression reduces autoregressive attention memory by keeping "
        "only the most useful tokens. Layer-adaptive budgets can preserve lower "
        "layer context while aggressively shrinking high-layer context. "
    )
    return (f"Fallback sample for {dataset_name}. " + base) * 256


def collect_dataset_text(
    dataset_name: str,
    tokenizer,
    min_tokens: int,
    split: str = "test",
    allow_fallback: bool = False,
) -> tuple[str, str]:
    from datasets import load_dataset

    if dataset_name == "wikitext":
        candidates = [
            ("Salesforce/wikitext", "wikitext-2-raw-v1", split),
            ("wikitext", "wikitext-2-raw-v1", split),
        ]
    elif dataset_name == "pg19":
        candidates = [
            ("deepmind/pg19", None, split),
            ("emozilla/pg19", None, split),
            ("ZengXiangyu/pg19", None, split),
        ]
    else:
        raise ValueError("dataset_name must be 'wikitext' or 'pg19'.")

    errors: list[str] = []
    target_chars = max(4096, min_tokens * 8)
    for name, config, ds_split in candidates:
        try:
            kwargs = {"split": ds_split, "streaming": True}
            dataset = load_dataset(name, config, **kwargs) if config else load_dataset(name, **kwargs)
            pieces: list[str] = []
            char_count = 0
            for row in dataset:
                text = _extract_text(row).strip()
                if len(text) < 40:
                    continue
                pieces.append(text)
                char_count += len(text)
                if char_count >= target_chars:
                    candidate = "\n\n".join(pieces)
                    ids = tokenizer(
                        candidate,
                        add_special_tokens=False,
                        return_tensors="pt",
                        truncation=True,
                        max_length=min_tokens,
                    ).input_ids
                    if ids.shape[1] >= min_tokens:
                        return candidate, f"{name}/{config or 'default'}:{ds_split}"
                    target_chars *= 2
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    if allow_fallback:
        return _fallback_text(dataset_name), f"fallback:{dataset_name}"
    raise RuntimeError(
        f"Could not load enough text for {dataset_name}. Use --allow-fallback for smoke runs.\n"
        + "\n".join(errors)
    )


def tokenize_to_length(tokenizer, text: str, token_count: int) -> torch.Tensor:
    ids = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=token_count,
    ).input_ids
    if ids.shape[1] < token_count:
        repeated = (text + "\n") * (token_count // max(1, ids.shape[1]) + 2)
        ids = tokenizer(
            repeated,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True,
            max_length=token_count,
        ).input_ids
    if ids.shape[1] < token_count:
        raise RuntimeError(f"Only tokenized {ids.shape[1]} tokens, need {token_count}.")
    return ids[:, :token_count]


def conda_env_hint() -> str:
    return os.environ.get("CONDA_DEFAULT_ENV", "(not inside conda)")


def now_s() -> float:
    return time.perf_counter()
