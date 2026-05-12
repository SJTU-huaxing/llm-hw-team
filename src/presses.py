"""KVPress policies and the team custom press.

All compression algorithms are implemented as KVPress ``BasePress`` objects.
The only model-specific code is a small Pythia/GPT-NeoX adapter so the official
KVPress pipeline can hook ``gpt_neox.layers[*].attention``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Iterable

import torch
from torch import nn
from kvpress import ExpectedAttentionPress as KVPressExpectedAttentionPress


VALID_POLICIES: tuple[str, ...] = (
    "dense",
    "streaming",
    "snapkv",
    "pyramidkv",
    "expected",
    "expected_pyramid",
    "layer_adaptive_expected_pyramid",
)
POLICY_ALIASES = {"full": "dense", "proposed": "layer_adaptive_expected_pyramid"}


@dataclass(frozen=True)
class PressSpec:
    policy: str
    implementation: str
    compression_ratio: float
    target_cache_size: int
    sink_size: int
    snap_observation_window: int
    pyramid_window_size: int
    pyramid_beta: int
    lazy_threshold: float


class GPTNeoXQueryProjection:
    """Callable q-projection shim for KVPress utilities expecting ``q_proj``."""

    def __init__(self, attention):
        self.attention = attention

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attention = self.attention
        bsz, q_len, _ = hidden_states.shape
        n_heads = attention.config.num_attention_heads
        head_dim = attention.head_dim
        qkv = attention.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, n_heads, 3 * head_dim)
        query = qkv[..., :head_dim]
        return query.reshape(bsz, q_len, n_heads * head_dim)


class PythiaExpectedAttentionPress(KVPressExpectedAttentionPress):
    """KVPress ExpectedAttention with a GPT-NeoX partial-RoPE correction.

    KVPress' implementation assumes all head dimensions are rotary-embedded.
    Pythia/GPT-NeoX rotates only ``rotary_ndims`` dimensions and leaves the
    remaining channels unchanged.  This override keeps the same expected
    attention scoring formula but applies the averaged RoPE matrix only to the
    rotated subspace.
    """

    def apply_avg_rope(self, module: nn.Module, mu: torch.Tensor, cov: torch.Tensor, q_len: int):
        position_ids = torch.arange(q_len, q_len + self.n_future_positions, device=mu.device).unsqueeze(0)
        head_dim = module.head_dim
        rotary_dim = int(getattr(module, "rotary_ndims", head_dim))

        cos, sin = module.rotary_emb(mu, position_ids)
        cos, sin = cos[0][..., :rotary_dim], sin[0][..., :rotary_dim]

        rot_eye = torch.eye(rotary_dim, device=mu.device, dtype=mu.dtype)
        rot_perm = torch.zeros((rotary_dim, rotary_dim), device=mu.device, dtype=mu.dtype)
        half = rotary_dim // 2
        rot_perm[half:, :half] = torch.eye(half, device=mu.device, dtype=mu.dtype)
        rot_perm[:half, half:] = -torch.eye(half, device=mu.device, dtype=mu.dtype)

        rot_matrix = cos.unsqueeze(1) * rot_eye + sin.unsqueeze(1) * rot_perm
        rot_matrix = rot_matrix.mean(dim=0).to(mu.device)

        avg_matrix = torch.eye(head_dim, device=mu.device, dtype=mu.dtype)
        avg_matrix[:rotary_dim, :rotary_dim] = rot_matrix

        mu = torch.matmul(mu, avg_matrix.T)
        if cov is not None:
            cov = torch.matmul(avg_matrix, torch.matmul(cov, avg_matrix.T))
        return mu, cov


def kvpress_version() -> str:
    try:
        return version("kvpress")
    except PackageNotFoundError:
        return "not-installed"


def normalize_policy(policy: str) -> str:
    return POLICY_ALIASES.get(policy, policy)


def iter_policies(policies: Iterable[str] | None = None) -> list[str]:
    if policies is None:
        return list(VALID_POLICIES)
    normalized = []
    for policy in policies:
        policy_name = normalize_policy(policy)
        if policy_name not in VALID_POLICIES:
            raise ValueError(f"Unknown policy {policy!r}; choose from {VALID_POLICIES}.")
        normalized.append(policy_name)
    return normalized


def compression_ratio_for_target(context_len: int, target_cache_size: int) -> float:
    if context_len <= 0:
        return 0.0
    target = max(1, min(target_cache_size, context_len))
    if target >= context_len:
        return 0.0
    keep_fraction = (target + 0.5) / context_len
    return max(0.0, min(0.999, 1.0 - keep_fraction))


def _is_pythia(model) -> bool:
    return hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers")


def patch_pythia_for_kvpress(model) -> None:
    if not _is_pythia(model):
        return
    config = model.config
    if not hasattr(config, "num_key_value_heads") or config.num_key_value_heads is None:
        config.num_key_value_heads = config.num_attention_heads
    head_dim = config.hidden_size // config.num_attention_heads
    model.__dict__["model"] = model.gpt_neox
    for idx, layer in enumerate(model.gpt_neox.layers):
        attention = layer.attention
        layer.__dict__["self_attn"] = attention
        attention.config = config
        attention.layer_idx = idx
        attention.head_dim = head_dim
        attention.num_key_value_heads = config.num_key_value_heads
        attention.__dict__["rotary_emb"] = model.gpt_neox.rotary_emb
        attention.__dict__["q_proj"] = GPTNeoXQueryProjection(attention)


class ExpectedPyramidPress:
    """Expected-attention scoring with PyramidKV-style layer budgets."""

    def __init__(
        self,
        compression_ratio: float = 0.0,
        n_future_positions: int = 512,
        n_sink: int = 4,
        window_size: int = 4,
        beta: int = 20,
        use_covariance: bool = True,
        use_vnorm: bool = True,
        epsilon: float = 0.0,
    ) -> None:
        self.scorer = PythiaExpectedAttentionPress(
            compression_ratio=compression_ratio,
            n_future_positions=n_future_positions,
            n_sink=n_sink,
            use_covariance=use_covariance,
            use_vnorm=use_vnorm,
            epsilon=epsilon,
        )
        self.compression_ratio = compression_ratio
        self.n_sink = n_sink
        self.window_size = window_size
        self.beta = beta

    def post_init_from_model(self, model):
        self.scorer.post_init_from_model(model)

    def get_layer_budget(self, module: nn.Module, q_len: int) -> int:
        max_capacity = self.window_size + q_len * (1 - self.compression_ratio)
        min_num = (max_capacity - self.window_size) / self.beta
        max_num = (max_capacity - self.window_size) * 2 - min_num
        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (max_capacity - self.window_size) * 2 - max_num
        if not (q_len >= max_num >= min_num >= self.window_size):
            return max(1, round(q_len * (1 - self.compression_ratio)))
        steps = (max_num - min_num) / max(1, module.config.num_hidden_layers - 1)
        return max(1, round(max_num - module.layer_idx * steps))

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.compression_ratio == 0:
            return keys, values
        k_len = keys.shape[2]
        n_kept = min(k_len, self.get_layer_budget(module, k_len))
        if n_kept >= k_len:
            return keys, values
        scores = self.scorer.score(module, hidden_states, keys, values, attentions, kwargs)
        return gather_topk(keys, values, scores, n_kept, module.head_dim)

    def forward_hook(self, module, input_args, kwargs, output):
        return apply_press_to_cache(self, module, input_args, kwargs, output)


class LayerAdaptiveExpectedPyramidPress(ExpectedPyramidPress):
    """ExpectedPyramid with an observed-attention lazy-layer gate.

    If the latest queries concentrate enough mass on sink and recent positions,
    the layer uses a deterministic sink+recent cache. Otherwise it keeps the
    expected-attention top-k tokens under a PyramidKV layer budget.
    """

    def __init__(
        self,
        compression_ratio: float = 0.0,
        n_future_positions: int = 512,
        n_sink: int = 4,
        window_size: int = 4,
        beta: int = 20,
        lazy_threshold: float = 0.75,
        n_last: int = 4,
        use_covariance: bool = True,
        use_vnorm: bool = True,
        epsilon: float = 0.0,
    ) -> None:
        super().__init__(
            compression_ratio=compression_ratio,
            n_future_positions=n_future_positions,
            n_sink=n_sink,
            window_size=window_size,
            beta=beta,
            use_covariance=use_covariance,
            use_vnorm=use_vnorm,
            epsilon=epsilon,
        )
        self.lazy_threshold = lazy_threshold
        self.n_last = n_last
        self.lazy_layers: list[int] = []

    def _lazy_mass(self, attentions: torch.Tensor, n_kept: int) -> float | None:
        if attentions is None:
            return None
        recent = max(1, n_kept - self.n_sink)
        scores = attentions[..., -self.n_last :, :].float().mean(dim=(0, 1, 2))
        sink_mass = scores[: self.n_sink].sum()
        recent_mass = scores[-recent:].sum()
        return float((sink_mass + recent_mass).item())

    def _sink_recent(self, keys: torch.Tensor, values: torch.Tensor, n_kept: int):
        k_len = keys.shape[2]
        n_kept = min(k_len, max(1, n_kept))
        sink_end = min(self.n_sink, n_kept)
        recent = max(0, n_kept - sink_end)
        if recent == 0:
            indices = torch.arange(sink_end, device=keys.device)
        else:
            recent_start = max(sink_end, k_len - recent)
            indices = torch.cat(
                [
                    torch.arange(0, sink_end, device=keys.device),
                    torch.arange(recent_start, k_len, device=keys.device),
                ]
            ).unique(sorted=True)
        return index_select_common(keys, values, indices)

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if module.layer_idx == 0:
            self.lazy_layers = []
        if self.compression_ratio == 0:
            return keys, values
        k_len = keys.shape[2]
        n_kept = min(k_len, self.get_layer_budget(module, k_len))
        if n_kept >= k_len:
            return keys, values
        lazy_mass = self._lazy_mass(attentions, n_kept)
        if lazy_mass is not None and lazy_mass >= self.lazy_threshold:
            self.lazy_layers.append(module.layer_idx)
            return self._sink_recent(keys, values, n_kept)
        return super().compress(module, hidden_states, keys, values, attentions, kwargs)


def gather_topk(
    keys: torch.Tensor,
    values: torch.Tensor,
    scores: torch.Tensor,
    n_kept: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = scores.topk(n_kept, dim=-1).indices
    indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    return keys.gather(2, indices).contiguous(), values.gather(2, indices).contiguous()


def index_select_common(
    keys: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return keys.index_select(2, indices), values.index_select(2, indices)


def apply_press_to_cache(press, module, input_args, kwargs, output):
    from transformers import QuantizedCache
    from kvpress.utils import extract_keys_and_values

    hidden_states = kwargs["hidden_states"]
    cache = kwargs["past_key_values"]
    cache_layer = cache.layers[module.layer_idx]
    q_len = hidden_states.shape[1]
    if kwargs["cache_position"][-1] > q_len:
        return output
    keys, values = extract_keys_and_values(cache, module.layer_idx)
    keys, values = press.compress(module, hidden_states, keys, values, output[1], kwargs)
    if isinstance(cache, QuantizedCache):
        cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
        cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
        cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)
        cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)
        cache_layer.cumulative_length = keys.shape[2]
    else:
        cache_layer.keys = keys
        cache_layer.values = values
    return output


def make_press(
    policy: str,
    context_len: int,
    target_cache_size: int,
    sink_size: int = 4,
    snap_observation_window: int = 32,
    pyramid_window_size: int = 4,
    pyramid_beta: int = 20,
    lazy_threshold: float = 0.75,
):
    policy = normalize_policy(policy)
    ratio = 0.0 if policy == "dense" else compression_ratio_for_target(context_len, target_cache_size)
    spec = PressSpec(
        policy=policy,
        implementation=f"kvpress-{kvpress_version()}",
        compression_ratio=round(ratio, 6),
        target_cache_size=target_cache_size,
        sink_size=sink_size,
        snap_observation_window=snap_observation_window,
        pyramid_window_size=pyramid_window_size,
        pyramid_beta=pyramid_beta,
        lazy_threshold=lazy_threshold,
    )
    if policy == "dense":
        return None, spec

    from kvpress import PyramidKVPress, SnapKVPress, StreamingLLMPress

    if policy == "streaming":
        return StreamingLLMPress(compression_ratio=ratio, n_sink=sink_size), spec
    if policy == "snapkv":
        return SnapKVPress(compression_ratio=ratio, window_size=snap_observation_window), spec
    if policy == "pyramidkv":
        return PyramidKVPress(compression_ratio=ratio, window_size=pyramid_window_size, beta=pyramid_beta), spec
    if policy == "expected":
        return PythiaExpectedAttentionPress(compression_ratio=ratio, n_sink=sink_size), spec
    if policy == "expected_pyramid":
        return ExpectedPyramidPress(
            compression_ratio=ratio,
            n_sink=sink_size,
            window_size=pyramid_window_size,
            beta=pyramid_beta,
        ), spec
    if policy == "layer_adaptive_expected_pyramid":
        return LayerAdaptiveExpectedPyramidPress(
            compression_ratio=ratio,
            n_sink=sink_size,
            window_size=pyramid_window_size,
            beta=pyramid_beta,
            lazy_threshold=lazy_threshold,
        ), spec
    raise ValueError(f"Unknown policy {policy!r}; choose from {VALID_POLICIES}.")


def press_needs_attentions(press) -> bool:
    return press is not None and press.__class__.__name__ in {
        "SnapKVPress",
        "PyramidKVPress",
        "LayerAdaptiveExpectedPyramidPress",
    }


class PythiaPressAdapter:
    def __init__(self, press):
        self.press = press

    def __getattr__(self, name):
        return getattr(self.press, name)

    @contextmanager
    def __call__(self, model):
        with kvpress_model_context(model, self.press):
            yield


def wrap_press_for_pythia(press):
    return None if press is None else PythiaPressAdapter(press)


def build_kvpress_pipeline(model_name: str, device: torch.device, dtype_name: str):
    import kvpress.pipeline  # noqa: F401 - registers the custom pipeline task.
    from transformers import pipeline

    try:
        from .utils import load_model_and_tokenizer
    except ImportError:
        from utils import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(
        model_name,
        device=device,
        dtype_name=dtype_name,
        attn_implementation="eager",
    )
    patch_pythia_for_kvpress(model)
    return pipeline(
        "kv-press-text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device.type == "cuda" else -1,
    )


@contextmanager
def kvpress_model_context(model, press):
    if press is None:
        yield
        return

    old_output_attentions = getattr(model.config, "output_attentions", False)
    if press_needs_attentions(press):
        model.config.output_attentions = True

    if not _is_pythia(model):
        try:
            with press(model):
                yield
        finally:
            model.config.output_attentions = old_output_attentions
        return

    patch_pythia_for_kvpress(model)
    press.post_init_from_model(model)
    hooks = []

    def hook(module, input_args, kwargs, output):
        cache = kwargs.get("layer_past")
        if cache is None:
            return output
        adapted_kwargs = dict(kwargs)
        adapted_kwargs["hidden_states"] = input_args[0]
        adapted_kwargs["past_key_values"] = cache
        return press.forward_hook(module, input_args, adapted_kwargs, output)

    try:
        for layer in model.gpt_neox.layers:
            hooks.append(layer.attention.register_forward_hook(hook, with_kwargs=True))
        yield
    finally:
        for forward_hook in hooks:
            forward_hook.remove()
        model.config.output_attentions = old_output_attentions


def get_cache_layer_lengths(past_key_values) -> list[int]:
    if past_key_values is None:
        return []
    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        return [
            int(layer.keys.shape[-2])
            for layer in layers
            if torch.is_tensor(getattr(layer, "keys", None))
        ]
    if hasattr(past_key_values, "to_legacy_cache"):
        return get_cache_layer_lengths(past_key_values.to_legacy_cache())
    return [int(layer[0].shape[-2]) for layer in past_key_values if layer]


def get_average_cache_seq_len(past_key_values) -> float:
    lengths = get_cache_layer_lengths(past_key_values)
    return sum(lengths) / len(lengths) if lengths else 0.0


def get_cache_seq_len(past_key_values) -> int:
    lengths = get_cache_layer_lengths(past_key_values)
    return lengths[0] if lengths else 0


def estimate_kv_cache_mb(past_key_values) -> float:
    layers = getattr(past_key_values, "layers", None)
    total_bytes = 0
    if layers is not None:
        for layer in layers:
            for tensor in (getattr(layer, "keys", None), getattr(layer, "values", None)):
                if torch.is_tensor(tensor):
                    total_bytes += tensor.numel() * tensor.element_size()
        return total_bytes / (1024**2)
    if hasattr(past_key_values, "to_legacy_cache"):
        return estimate_kv_cache_mb(past_key_values.to_legacy_cache())
    for layer in past_key_values or ():
        for tensor in layer[:2]:
            if torch.is_tensor(tensor):
                total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024**2)


def estimate_decode_attention_flops(layer_lengths_by_step: list[list[int]], n_heads: int, head_dim: int) -> int:
    return int(sum(sum(4 * n_heads * head_dim * length for length in step) for step in layer_lengths_by_step))
