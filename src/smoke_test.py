"""Smoke test all official baselines and the custom team press."""

from __future__ import annotations

from transformers import DynamicCache

try:
    from .presses import (
        build_kvpress_pipeline,
        get_average_cache_seq_len,
        get_cache_layer_lengths,
        iter_policies,
        make_press,
        wrap_press_for_pythia,
    )
    from .utils import DEFAULT_MODEL, conda_env_hint, resolve_device, seed_everything
except ImportError:
    from presses import (
        build_kvpress_pipeline,
        get_average_cache_seq_len,
        get_cache_layer_lengths,
        iter_policies,
        make_press,
        wrap_press_for_pythia,
    )
    from utils import DEFAULT_MODEL, conda_env_hint, resolve_device, seed_everything


def main() -> None:
    seed_everything(13)
    device = resolve_device("auto")
    print(f"Conda env: {conda_env_hint()}")
    print(f"Device: {device}")
    pipe = build_kvpress_pipeline(DEFAULT_MODEL, device=device, dtype_name="float32")

    context = (
        "KV cache compression keeps inference efficient while preserving the most useful context. "
        "Layer-adaptive budgets can preserve lower layers and compress higher layers more aggressively. "
    ) * 64
    policies = (
        "dense",
        "streaming",
        "snapkv",
        "pyramidkv",
        "expected",
        "expected_pyramid",
        "layer_adaptive_expected_pyramid",
        "observed",
        "tova",
        "knorm",
        "keydiff",
        "critical_expected",
        "chunk_expected",
        "hybrid_expected_observed",
        "hybrid_soft_pyramid",
        "expected_observed_residual",
        "expected_soft_pyramid",
        "residual_soft_pyramid",
    )

    lengths_by_policy = {}
    for policy in iter_policies(policies):
        press, spec = make_press(
            policy,
            context_len=192,
            target_cache_size=132,
            lazy_threshold=0.75,
        )
        cache = DynamicCache()
        result = pipe(
            context,
            question="",
            press=wrap_press_for_pythia(press),
            cache=cache,
            max_context_length=192,
            max_new_tokens=2,
        )
        lengths = get_cache_layer_lengths(cache)
        lengths_by_policy[policy] = lengths
        print(
            f"{policy:34s} ratio={spec.compression_ratio:.4f} "
            f"avg={get_average_cache_seq_len(cache):6.2f} layers={lengths} answer={result['answer']!r}"
        )

    dense_avg = get_average(lengths_by_policy["dense"])
    for policy in policies[1:]:
        assert get_average(lengths_by_policy[policy]) < dense_avg, f"{policy} did not compress"
    assert len(set(lengths_by_policy["pyramidkv"])) > 1
    assert len(set(lengths_by_policy["expected_pyramid"])) > 1
    assert len(set(lengths_by_policy["layer_adaptive_expected_pyramid"])) > 1
    assert len(set(lengths_by_policy["hybrid_soft_pyramid"])) > 1
    assert min(lengths_by_policy["hybrid_soft_pyramid"]) >= 36
    assert len(set(lengths_by_policy["expected_soft_pyramid"])) > 1
    assert len(set(lengths_by_policy["residual_soft_pyramid"])) > 1
    print("Smoke test passed.")


def get_average(lengths: list[int]) -> float:
    return sum(lengths) / len(lengths)


if __name__ == "__main__":
    main()
