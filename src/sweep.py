"""Run a reproducible hyperparameter sweep over KV cache budgets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from .utils import RESULTS_DIR, ensure_results_dir, write_json
except ImportError:
    from utils import RESULTS_DIR, ensure_results_dir, write_json


def run_command(cmd: list[str], cwd: Path) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", nargs="+", type=int, default=[128, 192, 256])
    parser.add_argument("--lazy-thresholds", nargs="+", type=float, default=[0.6, 0.75, 0.9])
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--continuation-len", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--allow-fallback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out_dir = ensure_results_dir() / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    commands: list[list[str]] = []
    fixed_policies = ["dense", "streaming", "snapkv", "pyramidkv", "expected", "expected_pyramid"]
    for budget in args.budgets:
        common_flag = ["--allow-fallback"] if args.allow_fallback else []
        commands.append(
            [
                sys.executable,
                "-B",
                "-m",
                "src.evaluate_ppl",
                "--policies",
                *fixed_policies,
                "--context-len",
                str(args.context_len),
                "--continuation-len",
                str(args.continuation_len),
                "--target-cache-size",
                str(budget),
                "--output",
                str(out_dir / f"ppl_budget_{budget}.csv"),
                *common_flag,
            ]
        )
        commands.append(
            [
                sys.executable,
                "-B",
                "-m",
                "src.benchmark_generate",
                "--policies",
                *fixed_policies,
                "--context-len",
                str(args.context_len),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--target-cache-size",
                str(budget),
                "--output",
                str(out_dir / f"bench_budget_{budget}.csv"),
                *common_flag,
            ]
        )

    for threshold in args.lazy_thresholds:
        common_flag = ["--allow-fallback"] if args.allow_fallback else []
        commands.append(
            [
                sys.executable,
                "-B",
                "-m",
                "src.evaluate_ppl",
                "--policies",
                "layer_adaptive_expected_pyramid",
                "--context-len",
                str(args.context_len),
                "--continuation-len",
                str(args.continuation_len),
                "--target-cache-size",
                "192",
                "--lazy-threshold",
                str(threshold),
                "--output",
                str(out_dir / f"ppl_custom_lazy_{threshold}.csv"),
                *common_flag,
            ]
        )
        commands.append(
            [
                sys.executable,
                "-B",
                "-m",
                "src.benchmark_generate",
                "--policies",
                "layer_adaptive_expected_pyramid",
                "--context-len",
                str(args.context_len),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--target-cache-size",
                "192",
                "--lazy-threshold",
                str(threshold),
                "--output",
                str(out_dir / f"bench_custom_lazy_{threshold}.csv"),
                *common_flag,
            ]
        )

    for cmd in commands:
        run_command(cmd, root)

    write_json(
        out_dir / "sweep_manifest.json",
        {
            "budgets": args.budgets,
            "lazy_thresholds": args.lazy_thresholds,
            "context_len": args.context_len,
            "continuation_len": args.continuation_len,
            "max_new_tokens": args.max_new_tokens,
            "commands": commands,
        },
    )
    print(f"Wrote {out_dir / 'sweep_manifest.json'}")


if __name__ == "__main__":
    main()
