"""Evaluate continuation perplexity after KVPress prefill compression."""

from __future__ import annotations

import argparse
import math
import platform
from pathlib import Path

import torch
from transformers import DynamicCache

try:
    from .presses import (
        estimate_decode_attention_flops,
        estimate_kv_cache_mb,
        get_average_cache_seq_len,
        get_cache_layer_lengths,
        iter_policies,
        kvpress_model_context,
        make_press,
        patch_pythia_for_kvpress,
        press_needs_attentions,
    )
    from .utils import (
        DEFAULT_MODEL,
        collect_dataset_text,
        conda_env_hint,
        ensure_results_dir,
        load_model_and_tokenizer,
        peak_memory_mb,
        reset_peak_memory,
        resolve_device,
        seed_everything,
        synchronize,
        tokenize_to_length,
        write_csv,
        write_json,
    )
except ImportError:
    from presses import (
        estimate_decode_attention_flops,
        estimate_kv_cache_mb,
        get_average_cache_seq_len,
        get_cache_layer_lengths,
        iter_policies,
        kvpress_model_context,
        make_press,
        patch_pythia_for_kvpress,
        press_needs_attentions,
    )
    from utils import (
        DEFAULT_MODEL,
        collect_dataset_text,
        conda_env_hint,
        ensure_results_dir,
        load_model_and_tokenizer,
        peak_memory_mb,
        reset_peak_memory,
        resolve_device,
        seed_everything,
        synchronize,
        tokenize_to_length,
        write_csv,
        write_json,
    )


def _forward_prefill(model, input_ids: torch.Tensor, cache: DynamicCache, press) -> None:
    kwargs = {
        "input_ids": input_ids,
        "past_key_values": cache,
        "use_cache": True,
        "output_attentions": press_needs_attentions(press),
        "return_dict": True,
    }
    if press is None:
        model(**kwargs)
    else:
        with kvpress_model_context(model, press):
            model(**kwargs)


def _model_forward_step(model, input_ids: torch.Tensor, cache, position: int):
    position_ids = torch.tensor([[position]], dtype=torch.long, device=input_ids.device)
    kwargs = {
        "input_ids": input_ids,
        "past_key_values": cache,
        "use_cache": True,
        "position_ids": position_ids,
        "return_dict": True,
    }
    try:
        return model(**kwargs, logits_to_keep=1)
    except TypeError:
        try:
            return model(**kwargs, num_logits_to_keep=1)
        except TypeError:
            return model(**kwargs)


@torch.no_grad()
def evaluate_one(
    *,
    model,
    tokenizer,
    dataset_name: str,
    text: str,
    text_source: str,
    policy: str,
    context_len: int,
    continuation_len: int,
    target_cache_size: int,
    sink_size: int,
    snap_observation_window: int,
    pyramid_window_size: int,
    pyramid_beta: int,
    lazy_threshold: float,
    device: torch.device,
) -> dict:
    token_ids = tokenize_to_length(tokenizer, text, context_len + continuation_len + 1).to(device)
    context_ids = token_ids[:, :context_len]
    press, spec = make_press(
        policy,
        context_len=context_len,
        target_cache_size=target_cache_size,
        sink_size=sink_size,
        snap_observation_window=snap_observation_window,
        pyramid_window_size=pyramid_window_size,
        pyramid_beta=pyramid_beta,
        lazy_threshold=lazy_threshold,
    )

    cache = DynamicCache()
    reset_peak_memory(device)
    synchronize(device)
    prefill_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    prefill_end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    import time

    t0 = time.perf_counter()
    if prefill_start is not None:
        prefill_start.record()
    _forward_prefill(model, context_ids, cache, press)
    if prefill_end is not None:
        prefill_end.record()
    synchronize(device)
    prefill_elapsed_s = time.perf_counter() - t0
    if prefill_start is not None and prefill_end is not None:
        prefill_elapsed_s = prefill_start.elapsed_time(prefill_end) / 1000.0

    prompt_lengths = get_cache_layer_lengths(cache)
    prompt_avg_cache = get_average_cache_seq_len(cache)
    prompt_cache_mb = estimate_kv_cache_mb(cache)

    nll_sum = 0.0
    decode_lengths_by_step: list[list[int]] = []
    decode_t0 = time.perf_counter()
    current = token_ids[:, context_len : context_len + 1]

    for step in range(continuation_len):
        position = context_len + step
        outputs = _model_forward_step(model, current, cache, position)
        target = token_ids[:, context_len + step + 1]
        logits = outputs.logits[:, -1, :]
        nll = torch.nn.functional.cross_entropy(logits.float(), target, reduction="sum")
        nll_sum += float(nll.item())
        decode_lengths_by_step.append(get_cache_layer_lengths(cache))
        current = target.view(1, 1)

    synchronize(device)
    decode_elapsed_s = time.perf_counter() - decode_t0
    mean_nll = nll_sum / continuation_len
    ppl = math.exp(mean_nll) if mean_nll < 80 else float("inf")
    final_lengths = get_cache_layer_lengths(cache)
    n_heads = int(model.config.num_attention_heads)
    head_dim = int(model.config.hidden_size // model.config.num_attention_heads)

    return {
        "dataset": dataset_name,
        "source": text_source,
        "policy": policy,
        "implementation": spec.implementation,
        "model": model.config.name_or_path or DEFAULT_MODEL,
        "context_len": context_len,
        "continuation_len": continuation_len,
        "target_cache_size": target_cache_size,
        "compression_ratio": spec.compression_ratio,
        "sink_size": sink_size,
        "snap_observation_window": snap_observation_window,
        "pyramid_window_size": pyramid_window_size,
        "pyramid_beta": pyramid_beta,
        "lazy_threshold": lazy_threshold,
        "nll": round(mean_nll, 6),
        "ppl": round(ppl, 6),
        "prefill_s": round(prefill_elapsed_s, 6),
        "decode_s": round(decode_elapsed_s, 6),
        "tokens_per_s": round(continuation_len / max(decode_elapsed_s, 1e-9), 3),
        "prompt_avg_cache_len": round(prompt_avg_cache, 3),
        "prompt_cache_lengths": " ".join(str(v) for v in prompt_lengths),
        "final_avg_cache_len": round(get_average_cache_seq_len(cache), 3),
        "final_cache_lengths": " ".join(str(v) for v in final_lengths),
        "kept_fraction_prompt": round(prompt_avg_cache / context_len, 6),
        "kv_cache_mb_prompt": round(prompt_cache_mb, 3),
        "kv_cache_mb_final": round(estimate_kv_cache_mb(cache), 3),
        "peak_cuda_mb": round(peak_memory_mb(device), 3),
        "decode_attention_flops": estimate_decode_attention_flops(decode_lengths_by_step, n_heads, head_dim),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--datasets", nargs="+", default=["wikitext", "pg19"])
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["dense", "streaming", "snapkv", "pyramidkv", "expected", "expected_pyramid", "layer_adaptive_expected_pyramid"],
    )
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--continuation-len", type=int, default=128)
    parser.add_argument("--target-cache-size", type=int, default=132)
    parser.add_argument("--sink-size", type=int, default=4)
    parser.add_argument("--snap-observation-window", type=int, default=32)
    parser.add_argument("--pyramid-window-size", type=int, default=4)
    parser.add_argument("--pyramid-beta", type=int, default=20)
    parser.add_argument("--lazy-threshold", type=float, default=0.75)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    policies = iter_policies(args.policies)
    model, tokenizer = load_model_and_tokenizer(args.model, device=device, dtype_name=args.dtype, attn_implementation="eager")
    patch_pythia_for_kvpress(model)

    rows: list[dict] = []
    metadata = {
        "task": "continuation_ppl",
        "conda_env": conda_env_hint(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "python": platform.python_version(),
        "torch_version": torch.__version__,
    }

    min_tokens = args.context_len + args.continuation_len + 1
    texts = {
        dataset: collect_dataset_text(dataset, tokenizer, min_tokens=min_tokens, allow_fallback=args.allow_fallback)
        for dataset in args.datasets
    }

    for dataset in args.datasets:
        text, source = texts[dataset]
        for policy in policies:
            print(f"[ppl] dataset={dataset} policy={policy}")
            rows.append(
                evaluate_one(
                    model=model,
                    tokenizer=tokenizer,
                    dataset_name=dataset,
                    text=text,
                    text_source=source,
                    policy=policy,
                    context_len=args.context_len,
                    continuation_len=args.continuation_len,
                    target_cache_size=args.target_cache_size,
                    sink_size=args.sink_size,
                    snap_observation_window=args.snap_observation_window,
                    pyramid_window_size=args.pyramid_window_size,
                    pyramid_beta=args.pyramid_beta,
                    lazy_threshold=args.lazy_threshold,
                    device=device,
                )
            )

    out_csv = args.output or (ensure_results_dir() / "ppl_results.csv")
    out_json = out_csv.with_suffix(".json")
    write_csv(out_csv, rows)
    write_json(out_json, {"metadata": metadata, "rows": rows})
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
