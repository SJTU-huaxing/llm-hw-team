# Balanced KVPress Compression for Pythia-70M

This group project studies training-free KV cache compression with NVIDIA KVPress and Transformers. It reproduces official baselines, then adds KVPress-compatible innovation variants for a PPL-speed tradeoff on `EleutherAI/pythia-70m`. The final recommended aggressive-compression method is `expected_soft_pyramid`, a conservative layer-budget extension of ExpectedAttention.

Remote repository: <https://github.com/SJTU-huaxing/llm-hw-team.git>

## Setup

- Model: `EleutherAI/pythia-70m`
- Training: none
- Environment: `llm_kvpress_team`
- Dependencies: `kvpress==0.5.3`, `transformers==4.57.6`, `torch==2.10.0+cu130`
- Datasets: WikiText-2 test and PG-19 test sample from HuggingFace, without fallback for reported results
- Hardware: NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB VRAM

```powershell
conda env create -f environment.yml
conda activate llm_kvpress_team
conda run -n llm_kvpress_team python -c "import sys, torch, transformers; from importlib.metadata import version; print(sys.executable); print(version('kvpress'), transformers.__version__, torch.__version__, torch.cuda.is_available())"
```

## Methods

| Assignment layer | Implemented component |
| --- | --- |
| Attention mechanism | ExpectedAttention, ObservedAttention/TOVA-style scores, value-norm scoring |
| Per-layer KV compression | StreamingLLM, SnapKV, PyramidKV, ExpectedAttention, ObservedAttention, TOVA |
| Inter-layer KV compression | Pyramid budgets, layer-adaptive gate, Expected Soft Pyramid budgets |
| Macro architecture | KVPress Transformers pipeline, prefill-once compression, unified latency/FLOPs measurement |

Official baselines include `dense`, `streaming`, `snapkv`, `pyramidkv`, `expected`, `observed`, `tova`, `knorm`, `keydiff`, `critical_expected`, and `chunk_expected`. `critical_expected` and `chunk_expected` are used in screening; `knorm` and `keydiff` are smoke-tested alternatives. The final tables keep the strongest two new baselines, `observed` and `tova`.

Custom presses:

- `expected_soft_pyramid`: the main innovation after iteration. It keeps ExpectedAttention as the token scorer and applies a mild 1.25-to-0.75 layer budget schedule, avoiding the aggressive high-layer pruning that hurt earlier pyramid variants.
- `hybrid_expected_observed`: combines ExpectedAttention, observed attention, and value norm scores with weights 0.55/0.30/0.15, while protecting sink and recent tokens.
- `hybrid_soft_pyramid`: adds mild non-uniform layer budgets to the hybrid score, with a `sink + recent` floor so high layers are not over-compressed.
- `expected_observed_residual` and `residual_soft_pyramid`: exploratory residual variants with ExpectedAttention-dominant weights. They are retained as ablations in `results/candidate_*` because the observed residual still degraded WikiText PPL.

The Pythia adapter only maps GPT-NeoX module names and partial RoPE semantics to KVPress expectations. It does not train, change model weights, or alter attention computation.

## Reproduce

Checks:

```powershell
conda run -n llm_kvpress_team python -B -m compileall -q src
conda run -n llm_kvpress_team python -B src\smoke_test.py
```

Screening run:

```powershell
$policies = 'dense expected observed tova critical_expected chunk_expected expected_observed_residual expected_soft_pyramid residual_soft_pyramid hybrid_expected_observed hybrid_soft_pyramid'
foreach ($budget in 96,128,192) {
  conda run -n llm_kvpress_team python -B -m src.evaluate_ppl --datasets wikitext pg19 --policies $policies.Split(' ') --context-len 256 --continuation-len 64 --target-cache-size $budget --output results\iteration_screening_ppl_$budget.csv
  conda run -n llm_kvpress_team python -B -m src.benchmark_generate --dataset pg19 --policies $policies.Split(' ') --context-len 256 --max-new-tokens 16 --target-cache-size $budget --output results\iteration_screening_generation_$budget.csv
}
```

Final run:

```powershell
$policies = 'dense streaming snapkv pyramidkv expected expected_soft_pyramid expected_pyramid layer_adaptive_expected_pyramid observed tova hybrid_expected_observed hybrid_soft_pyramid'
conda run -n llm_kvpress_team python -B -m src.evaluate_ppl --datasets wikitext pg19 --policies $policies.Split(' ') --context-len 512 --continuation-len 128 --target-cache-size 132 --output results\ppl_results.csv
conda run -n llm_kvpress_team python -B -m src.benchmark_generate --dataset pg19 --policies $policies.Split(' ') --context-len 512 --max-new-tokens 32 --target-cache-size 132 --output results\generation_benchmark.csv
conda run -n llm_kvpress_team python -B -m src.analyze_results --ppl results\ppl_results.csv --benchmark results\generation_benchmark.csv --output results\balanced_selection.csv
conda run -n llm_kvpress_team python -B -m src.plot_results
```

The same final commands were repeated for target cache sizes 192 and 256, producing `results/*_budget_192.*` and `results/*_budget_256.*`.

## Results

### Fixed target cache size 132

Continuation PPL after compressing a 512-token context:

| Policy | WikiText-2 | PG-19 | Avg PPL | Kept |
| --- | ---: | ---: | ---: | ---: |
| Dense | 9.750 | 32.938 | 21.344 | 1.000 |
| ExpectedAttention | 17.791 | 34.860 | 26.326 | 0.258 |
| Expected Soft Pyramid | 22.450 | 34.235 | 28.342 | 0.258 |
| Expected+Pyramid | 36.236 | 33.896 | 35.066 | 0.259 |
| Layer-adaptive Expected+Pyramid | 41.307 | 33.251 | 37.279 | 0.259 |
| Hybrid Expected+Observed | 42.667 | 33.072 | 37.870 | 0.258 |
| Hybrid Soft Pyramid | 45.052 | 32.715 | 38.884 | 0.258 |
| ObservedAttention | 48.345 | 31.870 | 40.107 | 0.258 |
| PyramidKV | 43.884 | 32.286 | 38.085 | 0.259 |
| SnapKV | 51.625 | 33.056 | 42.340 | 0.258 |
| StreamingLLM | 59.632 | 32.663 | 46.148 | 0.258 |
| TOVA | 74.139 | 32.634 | 53.386 | 0.258 |

PG-19 generation benchmark with 32 generated tokens:

| Policy | TTFT (s) | TPOT (s) | Throughput | FLOPs ratio |
| --- | ---: | ---: | ---: | ---: |
| Dense | 0.0541 | 0.006071 | 132.053 | 1.000 |
| SnapKV | 0.0061 | 0.005959 | 167.726 | 0.281 |
| Expected+Pyramid | 0.0060 | 0.006056 | 165.149 | 0.282 |
| Hybrid Soft Pyramid | 0.0059 | 0.006064 | 165.030 | 0.281 |
| Expected Soft Pyramid | 0.0063 | 0.006066 | 164.644 | 0.281 |
| Hybrid Expected+Observed | 0.0063 | 0.006080 | 164.339 | 0.281 |
| ExpectedAttention | 0.0060 | 0.006373 | 157.232 | 0.281 |
| ObservedAttention | 0.0073 | 0.006453 | 154.376 | 0.281 |

Balanced selection uses equal ranks over average PPL, TPOT, and FLOPs ratio among methods with kept fraction <= 0.5. At target cache size 132, `expected_soft_pyramid` is the selected Pareto method. `ExpectedAttention` remains the best pure-quality baseline, while `snapkv` has the lowest measured TPOT in this run.

### Budget sweep

| Target cache | Balanced selection | Avg PPL | TPOT (s) | FLOPs ratio | Kept |
| ---: | --- | ---: | ---: | ---: | ---: |
| 132 | Expected Soft Pyramid | 28.342 | 0.006066 | 0.281 | 0.258 |
| 192 | ExpectedAttention | 22.407 | 0.006006 | 0.395 | 0.375 |
| 256 | Hybrid Expected+Observed | 26.329 | 0.006105 | 0.516 | 0.500 |

Interpretation: the best innovation was not the more complex observed-attention hybrid. The stronger balanced design is a conservative ExpectedAttention-based soft pyramid: it sacrifices about 2.0 average PPL relative to ExpectedAttention at target 132, but recovers faster TPOT while keeping the same 0.281 FLOPs ratio. At larger budgets, pure ExpectedAttention remains the quality anchor and the observed hybrid can be attractive when latency receives equal weight.

Plots are saved in `results/`:

- `ppl_tradeoff.png`
- `generation_tradeoff.png`
- `quality_latency_tradeoff.png`
- `budget_sweep_tradeoff.png`

## Paper

The English NeurIPS-style paper is under `paper/` and compiles to `paper/main.pdf`:

```powershell
cd paper
latexmk -pdf main.tex
```

## References

- NVIDIA KVPress: <https://github.com/NVIDIA/kvpress>
- StreamingLLM: <https://arxiv.org/abs/2309.17453>
- SnapKV: <https://arxiv.org/abs/2404.14469>
- PyramidKV: <https://arxiv.org/abs/2406.02069>
- H2O / Observed attention: <https://arxiv.org/abs/2306.14048>
- TOVA: <https://arxiv.org/abs/2401.06104>
