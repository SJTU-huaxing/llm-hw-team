"""Plot PPL, latency, cache, and FLOPs tradeoffs from result CSVs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .utils import RESULTS_DIR, ensure_results_dir
except ImportError:
    from utils import RESULTS_DIR, ensure_results_dir


POLICY_LABELS = {
    "dense": "Dense",
    "streaming": "StreamingLLM",
    "snapkv": "SnapKV",
    "pyramidkv": "PyramidKV",
    "expected": "ExpectedAttention",
    "expected_pyramid": "Expected+Pyramid",
    "layer_adaptive_expected_pyramid": "Layer-adaptive Expected+Pyramid",
    "observed": "ObservedAttention",
    "tova": "TOVA",
    "knorm": "K-norm",
    "keydiff": "KeyDiff",
    "critical_expected": "Critical+Expected",
    "chunk_expected": "Chunk+Expected",
    "hybrid_expected_observed": "Hybrid Expected+Observed",
    "hybrid_soft_pyramid": "Hybrid Soft Pyramid",
    "expected_observed_residual": "Expected+Observed Residual",
    "expected_soft_pyramid": "Expected Soft Pyramid",
    "residual_soft_pyramid": "Residual Soft Pyramid",
}


def label(policy: str) -> str:
    return POLICY_LABELS.get(policy, policy)


def plot_ppl(ppl: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    pivot = ppl.pivot_table(index="policy", columns="dataset", values="ppl", aggfunc="mean")
    pivot = pivot.loc[[p for p in POLICY_LABELS if p in pivot.index]]
    pivot.rename(index=label).plot(kind="bar", ax=ax)
    ax.set_ylabel("Continuation perplexity")
    ax.set_xlabel("")
    ax.set_title("PPL after prefill KV compression")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Dataset")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_generation(bench: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    ordered = bench.set_index("policy").loc[[p for p in POLICY_LABELS if p in set(bench["policy"])]].reset_index()
    names = [label(p) for p in ordered["policy"]]

    axes[0].bar(names, ordered["ttft_s"], color="#4C78A8")
    axes[0].set_ylabel("TTFT (s)")
    axes[0].set_title("First token")

    axes[1].bar(names, ordered["tpot_s"], color="#F58518")
    axes[1].set_ylabel("TPOT (s)")
    axes[1].set_title("Per-token latency")

    axes[2].bar(names, ordered["kept_fraction_prompt"], color="#54A24B")
    axes[2].set_ylabel("Prompt KV kept fraction")
    axes[2].set_title("Memory proxy")

    for ax in axes:
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_tradeoff(ppl: pd.DataFrame, bench: pd.DataFrame, out: Path) -> None:
    avg_ppl = ppl.groupby("policy", as_index=False)["ppl"].mean()
    merged = bench.merge(avg_ppl, on="policy", how="inner")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for _, row in merged.iterrows():
        ax.scatter(row["tpot_s"], row["ppl"], s=80 * max(0.15, row["kept_fraction_prompt"]), alpha=0.85)
        ax.annotate(label(row["policy"]), (row["tpot_s"], row["ppl"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("TPOT (s, lower is better)")
    ax.set_ylabel("Average PPL (lower is better)")
    ax.set_title("Quality-latency-memory tradeoff")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _budget_from_path(path: Path) -> int | None:
    match = re.search(r"budget_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def plot_budget_sweep(out_dir: Path) -> None:
    ppl_frames = []
    bench_frames = []
    for path in sorted(out_dir.glob("ppl_budget_*.csv")):
        budget = _budget_from_path(path)
        if budget is None:
            continue
        frame = pd.read_csv(path)
        frame["budget"] = budget
        ppl_frames.append(frame)
    for path in sorted(out_dir.glob("generation_budget_*.csv")):
        budget = _budget_from_path(path)
        if budget is None:
            continue
        frame = pd.read_csv(path)
        frame["budget"] = budget
        bench_frames.append(frame)
    if not ppl_frames or not bench_frames:
        return

    ppl = pd.concat(ppl_frames, ignore_index=True).groupby(["budget", "policy"], as_index=False)["ppl"].mean()
    bench = pd.concat(bench_frames, ignore_index=True)
    merged = bench.merge(ppl, on=["budget", "policy"], how="inner")

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    markers = {128: "o", 192: "s", 256: "^"}
    for policy, group in merged.groupby("policy"):
        ax.plot(group["tpot_s"], group["ppl"], marker=markers.get(int(group["budget"].iloc[0]), "o"), label=label(policy), alpha=0.85)
        for _, row in group.iterrows():
            ax.annotate(str(int(row["budget"])), (row["tpot_s"], row["ppl"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("TPOT (s)")
    ax.set_ylabel("Average PPL")
    ax.set_title("Budget sweep: quality vs decoding latency")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "budget_sweep_tradeoff.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppl", type=Path, default=RESULTS_DIR / "ppl_results.csv")
    parser.add_argument("--benchmark", type=Path, default=RESULTS_DIR / "generation_benchmark.csv")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_results_dir() if args.out_dir == RESULTS_DIR else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ppl = pd.read_csv(args.ppl)
    bench = pd.read_csv(args.benchmark)
    plot_ppl(ppl, out_dir / "ppl_tradeoff.png")
    plot_generation(bench, out_dir / "generation_tradeoff.png")
    plot_tradeoff(ppl, bench, out_dir / "quality_latency_tradeoff.png")
    plot_budget_sweep(out_dir)
    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
