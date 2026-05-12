"""Select the balanced Pareto policy from PPL and generation results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .utils import RESULTS_DIR, write_csv, write_json
except ImportError:
    from utils import RESULTS_DIR, write_csv, write_json


def is_pareto_frontier(row: pd.Series, frame: pd.DataFrame) -> bool:
    metrics = ["avg_ppl", "tpot_s", "flops_ratio_vs_dense"]
    for _, other in frame.iterrows():
        if other["policy"] == row["policy"]:
            continue
        no_worse = all(float(other[m]) <= float(row[m]) for m in metrics)
        strictly_better = any(float(other[m]) < float(row[m]) for m in metrics)
        if no_worse and strictly_better:
            return False
    return True


def analyze(ppl_path: Path, benchmark_path: Path, max_kept_fraction: float) -> tuple[list[dict], dict]:
    ppl = pd.read_csv(ppl_path)
    benchmark = pd.read_csv(benchmark_path)
    avg_ppl = ppl.groupby("policy", as_index=False).agg(
        avg_ppl=("ppl", "mean"),
        ppl_wikitext=("ppl", lambda values: values[ppl.loc[values.index, "dataset"].eq("wikitext")].mean()),
        ppl_pg19=("ppl", lambda values: values[ppl.loc[values.index, "dataset"].eq("pg19")].mean()),
        kept_fraction_ppl=("kept_fraction_prompt", "mean"),
    )
    merged = avg_ppl.merge(
        benchmark[
            [
                "policy",
                "tpot_s",
                "ttft_s",
                "throughput_tok_s",
                "kept_fraction_prompt",
                "flops_ratio_vs_dense",
            ]
        ],
        on="policy",
        how="inner",
    )
    candidates = merged[
        (merged["policy"] != "dense")
        & (merged["kept_fraction_prompt"] <= max_kept_fraction)
    ].copy()
    for metric in ("avg_ppl", "tpot_s", "flops_ratio_vs_dense"):
        candidates[f"{metric}_rank"] = candidates[metric].rank(method="min", ascending=True)
    candidates["balanced_rank"] = candidates[
        ["avg_ppl_rank", "tpot_s_rank", "flops_ratio_vs_dense_rank"]
    ].mean(axis=1)
    candidates["pareto_frontier"] = candidates.apply(lambda row: is_pareto_frontier(row, candidates), axis=1)
    candidates = candidates.sort_values(["pareto_frontier", "balanced_rank", "avg_ppl"], ascending=[False, True, True])
    best = candidates.iloc[0].to_dict() if not candidates.empty else {}
    candidates["selected"] = candidates["policy"].eq(best.get("policy"))
    expected_soft = candidates[candidates["policy"] == "expected_soft_pyramid"]
    expected_soft_record = expected_soft.iloc[0].to_dict() if not expected_soft.empty else {}
    hybrid_soft = candidates[candidates["policy"] == "hybrid_soft_pyramid"]
    hybrid_soft_record = hybrid_soft.iloc[0].to_dict() if not hybrid_soft.empty else {}
    best_rank = float(best.get("balanced_rank", 0.0)) or 1.0
    expected_soft_within_10pct = bool(
        expected_soft_record and float(expected_soft_record["balanced_rank"]) <= best_rank * 1.10
    )
    hybrid_soft_within_10pct = bool(
        hybrid_soft_record and float(hybrid_soft_record["balanced_rank"]) <= best_rank * 1.10
    )
    summary = {
        "ppl_path": str(ppl_path),
        "benchmark_path": str(benchmark_path),
        "max_kept_fraction": max_kept_fraction,
        "selected_policy": best.get("policy"),
        "expected_soft_pyramid_within_10pct": expected_soft_within_10pct,
        "expected_soft_pyramid_balanced_rank": expected_soft_record.get("balanced_rank"),
        "hybrid_soft_pyramid_within_10pct": hybrid_soft_within_10pct,
        "hybrid_soft_pyramid_balanced_rank": hybrid_soft_record.get("balanced_rank"),
        "best_balanced_rank": best.get("balanced_rank"),
        "selection_rule": "lowest mean rank over average PPL, TPOT, and FLOPs ratio among policies with kept_fraction <= max_kept_fraction and on the Pareto frontier",
    }
    return candidates.to_dict(orient="records"), summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppl", type=Path, default=RESULTS_DIR / "ppl_results.csv")
    parser.add_argument("--benchmark", type=Path, default=RESULTS_DIR / "generation_benchmark.csv")
    parser.add_argument("--max-kept-fraction", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "balanced_selection.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = analyze(args.ppl, args.benchmark, args.max_kept_fraction)
    write_csv(args.output, rows)
    write_json(args.output.with_suffix(".json"), {"summary": summary, "rows": rows})
    print(f"Selected policy: {summary['selected_policy']}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
