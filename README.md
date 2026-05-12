# KVPress Multi-layer KV Cache Compression for Pythia-70M

This repository is the group project for the LLM inference acceleration assignment. It uses NVIDIA KVPress and Transformers to reproduce official KV cache compression baselines, then adds a minimal KVPress-compatible composition method for Pythia/GPT-NeoX.

Remote repository: <https://github.com/SJTU-huaxing/llm-hw-team.git>

## Scope

- Model: `EleutherAI/pythia-70m`
- Training: none
- Framework: `kvpress==0.5.3`, `transformers==4.57.6`, `torch==2.10.0+cu130`
- Datasets: WikiText-2 test and PG-19 test sample, loaded from HuggingFace without fallback for reported results
- Metrics: continuation PPL, TTFT, TPOT, throughput, KV kept fraction, peak CUDA memory, and decode attention FLOPs proxy
- Hardware used for reported runs: NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB VRAM

## Methods

| Assignment layer | Implemented component |
| --- | --- |
| Attention mechanism layer | ExpectedAttention scoring; observed attention is used by SnapKV/PyramidKV and the custom lazy-layer gate |
| Per-layer KV cache compression | StreamingLLM, SnapKV, PyramidKV, ExpectedAttention top-k |
| Inter-layer KV cache compression | Non-uniform PyramidKV budgets and the custom layer-adaptive lazy gate |
| Macro compute architecture | KVPress `kv-press-text-generation` pipeline, prefill-once compression, unified latency/FLOPs measurement |

Official KVPress baselines:

- `dense`: no press, full KV cache
- `streaming`: `StreamingLLMPress`
- `snapkv`: `SnapKVPress`
- `pyramidkv`: `PyramidKVPress`
- `expected`: KVPress ExpectedAttention scoring with a GPT-NeoX partial-RoPE compatibility shim

Custom method:

- `layer_adaptive_expected_pyramid`: a KVPress-style press that combines ExpectedAttention scores, PyramidKV layer budgets, and a conservative observed-attention lazy-layer gate. The adapter only handles Pythia/GPT-NeoX module naming and partial RoPE; it does not change model weights or attention semantics.

`SimLayerKVPress` was tested but not included as a baseline because its current KVPress implementation assumes full-dimensional RoPE, while Pythia uses partial RoPE.

## Reproduce

Create the isolated environment. This does not modify `base`:

```powershell
conda env create -f environment.yml
conda activate llm_kvpress_team
```

Verify isolation and dependency versions:

```powershell
conda run -n llm_kvpress_team python -c "import sys, torch, transformers; from importlib.metadata import version; print(sys.executable); print(version('kvpress'), transformers.__version__, torch.__version__, torch.cuda.is_available())"
```

Run checks:

```powershell
conda run -n llm_kvpress_team python -B -m compileall -q src
conda run -n llm_kvpress_team python -B src\smoke_test.py
```

Run the fixed 4x-compression experiment:

```powershell
conda run -n llm_kvpress_team python -B -m src.evaluate_ppl --datasets wikitext pg19 --context-len 512 --continuation-len 128 --target-cache-size 132 --output results\ppl_results.csv
conda run -n llm_kvpress_team python -B -m src.benchmark_generate --dataset pg19 --context-len 512 --max-new-tokens 32 --target-cache-size 132 --output results\generation_benchmark.csv
```

Run the budget sweep used in the paper:

```powershell
conda run -n llm_kvpress_team python -B -m src.evaluate_ppl --datasets wikitext pg19 --target-cache-size 192 --lazy-threshold 0.99 --output results\ppl_budget_192.csv
conda run -n llm_kvpress_team python -B -m src.evaluate_ppl --datasets wikitext pg19 --target-cache-size 256 --lazy-threshold 0.99 --output results\ppl_budget_256.csv
conda run -n llm_kvpress_team python -B -m src.benchmark_generate --target-cache-size 192 --lazy-threshold 0.99 --output results\generation_budget_192.csv
conda run -n llm_kvpress_team python -B -m src.benchmark_generate --target-cache-size 256 --lazy-threshold 0.99 --output results\generation_budget_256.csv
conda run -n llm_kvpress_team python -B -m src.plot_results
```

## Results

### Fixed target cache size 132

Continuation PPL after compressing a 512-token context:

| Policy | WikiText-2 PPL | PG-19 PPL | Avg kept |
| --- | ---: | ---: | ---: |
| Dense | 9.750 | 32.938 | 1.000 |
| StreamingLLM | 59.632 | 32.663 | 0.258 |
| SnapKV | 51.625 | 33.056 | 0.258 |
| PyramidKV | 43.884 | 32.286 | 0.259 |
| ExpectedAttention | 17.791 | 34.860 | 0.258 |
| Expected+Pyramid | 36.236 | 33.896 | 0.259 |
| Layer-adaptive Expected+Pyramid | 41.307 | 33.251 | 0.259 |

PG-19 generation benchmark with 32 generated tokens:

| Policy | TTFT (s) | TPOT (s) | Throughput (tok/s) | FLOPs ratio |
| --- | ---: | ---: | ---: | ---: |
| Dense | 0.0611 | 0.00660 | 120.445 | 1.000 |
| StreamingLLM | 0.00744 | 0.00682 | 146.224 | 0.281 |
| SnapKV | 0.00705 | 0.00683 | 146.284 | 0.281 |
| PyramidKV | 0.00721 | 0.00628 | 158.514 | 0.282 |
| ExpectedAttention | 0.00604 | 0.00589 | 169.578 | 0.281 |
| Expected+Pyramid | 0.00546 | 0.00533 | 187.537 | 0.282 |
| Layer-adaptive Expected+Pyramid | 0.00591 | 0.00611 | 163.791 | 0.282 |

### Budget sweep summary

Average PPL over WikiText-2 and PG-19:

| Target cache | Best policy | Avg PPL | Avg kept |
| ---: | --- | ---: | ---: |
| 132 | ExpectedAttention | 26.326 | 0.258 |
| 192 | Layer-adaptive Expected+Pyramid / Expected+Pyramid | 22.399 | 0.376 |
| 256 | ExpectedAttention | 21.967 | 0.500 |

Main observation: at aggressive 4x prompt-cache compression, ExpectedAttention gives the best average PPL, while Expected+Pyramid has the best TPOT in the generation run. With a 2x cache budget, ExpectedAttention nearly matches Dense PPL while halving KV memory and decode attention FLOPs.

Plots are saved in `results/`:

- `ppl_tradeoff.png`
- `generation_tradeoff.png`
- `quality_latency_tradeoff.png`
- `budget_sweep_tradeoff.png`

## Paper

The English NeurIPS-style paper is under `paper/`. Compile with:

```powershell
cd paper
latexmk -pdf main.tex
```

The paper includes the method mapping, experiment settings, tables, limitations, and placeholder team contribution shares that can be edited before final submission.

## References

- NVIDIA KVPress: <https://github.com/NVIDIA/kvpress>
- StreamingLLM: <https://arxiv.org/abs/2309.17453>
- SnapKV: <https://arxiv.org/abs/2404.14469>
- PyramidKV: <https://arxiv.org/abs/2406.02069>
- Expected Attention: implemented through NVIDIA KVPress `ExpectedAttentionPress`
