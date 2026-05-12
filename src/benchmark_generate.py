"""Benchmark KVPress generation latency and cache/FLOPs proxies."""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path
from types import MethodType

import torch
from transformers import DynamicCache

try:
    from .presses import (
        build_kvpress_pipeline,
        estimate_decode_attention_flops,
        estimate_kv_cache_mb,
        get_average_cache_seq_len,
        get_cache_layer_lengths,
        iter_policies,
        make_press,
        wrap_press_for_pythia,
    )
    from .utils import (
        DEFAULT_MODEL,
        collect_dataset_text,
        conda_env_hint,
        ensure_results_dir,
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
        build_kvpress_pipeline,
        estimate_decode_attention_flops,
        estimate_kv_cache_mb,
        get_average_cache_seq_len,
        get_cache_layer_lengths,
        iter_policies,
        make_press,
        wrap_press_for_pythia,
    )
    from utils import (
        DEFAULT_MODEL,
        collect_dataset_text,
        conda_env_hint,
        ensure_results_dir,
        peak_memory_mb,
        reset_peak_memory,
        resolve_device,
        seed_everything,
        synchronize,
        tokenize_to_length,
        write_csv,
        write_json,
    )


def _call_model(model, **kwargs):
    try:
        return model(**kwargs, logits_to_keep=1)
    except TypeError:
        try:
            return model(**kwargs, num_logits_to_keep=1)
        except TypeError:
            return model(**kwargs)


def install_timing_generate(pipe, timings: dict) -> None:
    def timed_generate_answer(self, question_ids: torch.Tensor, cache, context_length: int, max_new_tokens: int) -> str:
        position_ids = torch.arange(
            context_length, context_length + question_ids.shape[1], device=self.model.device
        ).unsqueeze(0)

        step_lengths: list[list[int]] = []
        step_times: list[float] = []

        synchronize(self.model.device)
        t0 = time.perf_counter()
        outputs = _call_model(
            self.model,
            input_ids=question_ids.to(self.model.device),
            past_key_values=cache,
            position_ids=position_ids,
            return_dict=True,
        )
        synchronize(self.model.device)
        ttft = time.perf_counter() - t0
        step_lengths.append(get_cache_layer_lengths(cache))
        step_times.append(ttft)

        position_ids = position_ids[:, -1:] + 1
        generated_ids = [outputs.logits[0, -1].argmax()]

        should_stop_token_ids = self.model.generation_config.eos_token_id
        if not isinstance(should_stop_token_ids, list):
            should_stop_token_ids = [should_stop_token_ids]

        for i in range(max_new_tokens - 1):
            synchronize(self.model.device)
            step_t0 = time.perf_counter()
            outputs = _call_model(
                self.model,
                input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
                past_key_values=cache,
                position_ids=position_ids + i,
                return_dict=True,
            )
            synchronize(self.model.device)
            step_times.append(time.perf_counter() - step_t0)
            step_lengths.append(get_cache_layer_lengths(cache))

            new_id = outputs.logits[0, -1].argmax()
            generated_ids.append(new_id)
            if new_id.item() in should_stop_token_ids:
                break

        timings.clear()
        timings.update(
            {
                "ttft_s": ttft,
                "step_times_s": step_times,
                "generated_tokens": len(generated_ids),
                "step_lengths": step_lengths,
                "final_lengths": get_cache_layer_lengths(cache),
            }
        )
        return str(self.tokenizer.decode(torch.stack(generated_ids), skip_special_tokens=True))

    pipe.generate_answer = MethodType(timed_generate_answer, pipe)


def run_one(
    *,
    pipe,
    prompt: str,
    dataset: str,
    source: str,
    policy: str,
    context_len: int,
    max_new_tokens: int,
    target_cache_size: int,
    sink_size: int,
    snap_observation_window: int,
    pyramid_window_size: int,
    pyramid_beta: int,
    lazy_threshold: float,
    device: torch.device,
) -> dict:
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
    timings: dict = {}
    install_timing_generate(pipe, timings)

    reset_peak_memory(device)
    synchronize(device)
    t0 = time.perf_counter()
    result = pipe(
        prompt,
        question="",
        press=wrap_press_for_pythia(press),
        cache=cache,
        max_context_length=context_len,
        max_new_tokens=max_new_tokens,
    )
    synchronize(device)
    total_elapsed = time.perf_counter() - t0

    prompt_lengths = get_cache_layer_lengths(cache)
    final_lengths = timings.get("final_lengths", prompt_lengths)
    generated_tokens = int(timings.get("generated_tokens", max_new_tokens))
    step_times = timings.get("step_times_s", [])
    decode_total = float(sum(step_times))
    tpot = float(sum(step_times[1:]) / max(1, generated_tokens - 1)) if step_times else 0.0
    n_heads = int(pipe.model.config.num_attention_heads)
    head_dim = int(pipe.model.config.hidden_size // pipe.model.config.num_attention_heads)
    decode_flops = estimate_decode_attention_flops(timings.get("step_lengths", []), n_heads, head_dim)

    return {
        "dataset": dataset,
        "source": source,
        "policy": policy,
        "implementation": spec.implementation,
        "model": pipe.model.config.name_or_path or DEFAULT_MODEL,
        "context_len": context_len,
        "max_new_tokens": max_new_tokens,
        "generated_tokens": generated_tokens,
        "target_cache_size": target_cache_size,
        "compression_ratio": spec.compression_ratio,
        "sink_size": sink_size,
        "snap_observation_window": snap_observation_window,
        "pyramid_window_size": pyramid_window_size,
        "pyramid_beta": pyramid_beta,
        "lazy_threshold": lazy_threshold,
        "ttft_s": round(float(timings.get("ttft_s", 0.0)), 6),
        "tpot_s": round(tpot, 6),
        "throughput_tok_s": round(generated_tokens / max(decode_total, 1e-9), 3),
        "total_pipeline_s": round(total_elapsed, 6),
        "prompt_avg_cache_len": round(get_average_cache_seq_len(cache), 3),
        "prompt_cache_lengths": " ".join(str(v) for v in prompt_lengths),
        "final_avg_cache_len": round(sum(final_lengths) / len(final_lengths), 3) if final_lengths else 0.0,
        "final_cache_lengths": " ".join(str(v) for v in final_lengths),
        "kept_fraction_prompt": round(get_average_cache_seq_len(cache) / context_len, 6),
        "kv_cache_mb_prompt": round(estimate_kv_cache_mb(cache), 3),
        "peak_cuda_mb": round(peak_memory_mb(device), 3),
        "decode_attention_flops": decode_flops,
        "avg_attention_flops_per_token": int(decode_flops / max(1, generated_tokens)),
        "answer_preview": result["answer"][:80].replace("\n", " "),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default="pg19", choices=["wikitext", "pg19"])
    parser.add_argument(
        "--policies",
        nargs="+",
        default=[
            "dense",
            "expected",
            "expected_soft_pyramid",
            "expected_pyramid",
            "layer_adaptive_expected_pyramid",
            "observed",
            "tova",
            "critical_expected",
            "chunk_expected",
            "hybrid_expected_observed",
            "hybrid_soft_pyramid",
        ],
    )
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=32)
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
    pipe = build_kvpress_pipeline(args.model, device=device, dtype_name=args.dtype)

    text, source = collect_dataset_text(
        args.dataset,
        pipe.tokenizer,
        min_tokens=args.context_len,
        allow_fallback=args.allow_fallback,
    )
    prompt_ids = tokenize_to_length(pipe.tokenizer, text, args.context_len)
    prompt = pipe.tokenizer.decode(prompt_ids[0], skip_special_tokens=False)

    rows: list[dict] = []
    for policy in policies:
        print(f"[bench] dataset={args.dataset} policy={policy}")
        rows.append(
            run_one(
                pipe=pipe,
                prompt=prompt,
                dataset=args.dataset,
                source=source,
                policy=policy,
                context_len=args.context_len,
                max_new_tokens=args.max_new_tokens,
                target_cache_size=args.target_cache_size,
                sink_size=args.sink_size,
                snap_observation_window=args.snap_observation_window,
                pyramid_window_size=args.pyramid_window_size,
                pyramid_beta=args.pyramid_beta,
                lazy_threshold=args.lazy_threshold,
                device=device,
            )
        )

    dense = next((row for row in rows if row["policy"] == "dense"), None)
    if dense:
        dense_tpot = max(float(dense["tpot_s"]), 1e-9)
        dense_flops = max(int(dense["decode_attention_flops"]), 1)
        for row in rows:
            row["tpot_speedup_vs_dense"] = round(dense_tpot / max(float(row["tpot_s"]), 1e-9), 3)
            row["flops_ratio_vs_dense"] = round(int(row["decode_attention_flops"]) / dense_flops, 6)

    out_csv = args.output or (ensure_results_dir() / "generation_benchmark.csv")
    out_json = out_csv.with_suffix(".json")
    write_csv(out_csv, rows)
    write_json(
        out_json,
        {
            "metadata": {
                "task": "generation_benchmark",
                "conda_env": conda_env_hint(),
                "device": str(device),
                "cuda_available": torch.cuda.is_available(),
                "python": platform.python_version(),
                "torch_version": torch.__version__,
            },
            "rows": rows,
        },
    )
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
