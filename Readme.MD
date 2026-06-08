# LLM Quantization Tradeoff Analyzer

Benchmarks FP16 / INT8 / INT4 quantization across **VRAM**, **latency (TTFT)**, **throughput (TPS)**, and **semantic quality** on a single model. Runs on Kaggle (T4 GPU) with a Gradio dashboard for results visualization.

---

## What it measures

| Metric | Method |
| :--- | :--- |
| VRAM | `memory_allocated()` delta — baseline subtracted post-unload to exclude allocator overhead |
| TTFT | `LogitsProcessor` hook timestamps first token emit |
| TPS | Generated tokens / wall time (warm passes only, pass 1 discarded) |
| Semantic quality | Cosine similarity of `all-MiniLM-L6-v2` embeddings vs FP16 baseline |
| Formatting | Regex checks for JSON / table / bullet structure against prompt intent |

---

## Results (TinyLlama-1.1B, Kaggle T4)

| Precision | VRAM (MB) | TTFT (ms) | TPS | Semantic Score |
| :--- | ---: | ---: | ---: | ---: |
| FP16 | 965 | 43.0 | 26.96 | 1.000 |
| INT8 | 469 | 156.2 | 8.28 | 0.878 |
| INT4 | 744 | 85.7 | 16.86 | 0.845 |

**Key finding:** FP16 outperforms both quantized formats on TPS on T4. Bitsandbytes dequantization overhead fully negates compute savings on hardware without native INT4/INT8 tensor cores. INT8 wins on VRAM (51% reduction) with the least quality regression.

---

## Project structure

```
.
├── quantization-tradeoff-analysis.ipynb   # Main benchmark notebook (run this)
├── app.py                                 # Gradio dashboard (imported by notebook)
└── benchmark_results/                     # Auto-created on run
    ├── perf_results.json
    └── quality_results.json
```

---

## Setup

**Kaggle** (recommended — free T4 GPU):

1. Upload both files to a Kaggle notebook
2. Enable GPU: *Settings → Accelerator → GPU T4 x1*
3. Run all cells in order

**Local** (GPU required):

```bash
pip install transformers bitsandbytes accelerate gradio \
            sentence-transformers pandas matplotlib seaborn
```

Then open the notebook or run:

```bash
python app.py --results ./benchmark_results
python app.py --demo    # no GPU needed, uses synthetic data
```

---

## Supported models

| Key | Model |
| :--- | :--- |
| `TinyLlama-1.1B` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `Phi-3-mini` | `microsoft/Phi-3-mini-4k-instruct` |
| `Qwen2.5-3B` | `Qwen/Qwen2.5-3B-Instruct` |

Change the model in the entry point cell:

```python
app = build_app(results_dir="./benchmark_results", demo_mode=False)
```

or in `run_full_benchmark()`:

```python
perf_results = run_full_benchmark(model_key="Phi-3-mini")
```

---

## Prompt suite

Quality is evaluated across 6 categories (3 prompts each by default):

- `reasoning` — multi-step logic problems
- `summarization` — concise technical explanation
- `structured_output` — JSON / YAML / markdown table generation
- `sql` — query writing with CTEs and window functions
- `long_context` — retrieval from inline context
- `hallucination_sensitive` — fact-critical questions requiring uncertainty handling

---

## Known limitations

- **VRAM measurement:** `memory_allocated()` delta still includes some bitsandbytes metadata for INT4 with `bnb_4bit_use_double_quant=True`, causing INT4 to read higher than INT8. True weight-only INT4 footprint should be ~241 MB (~75% reduction vs FP16 for 1.1B params). Use the parameter-sum method for precise accounting.
- **TPS / TTFT are hardware-specific:** Results on T4 (no INT4 tensor cores) will differ significantly from A100/H100/MI300X where quantized kernels are hardware-accelerated.
- **Semantic scoring saturates:** Cosine similarity of MiniLM embeddings does not catch factual degradation or hallucinations — only surface-level semantic drift from the FP16 reference.
- **Single prompt for perf benchmark:** TTFT reflects one fixed ~20-token prefill. Not representative of variable-length production traffic.

---

## Engineering findings

- **INT8** is the dominant choice on bandwidth-bound GPUs: 51% VRAM reduction with only 12% semantic quality drop and no throughput regression beyond dequant overhead.
- **INT4** gains only materialise on hardware with native quantized compute — use AutoAWQ / GPTQ via ROCm on AMD (MI300X) or TensorRT-LLM on Hopper+.
- At INT4, inference shifts from **memory-bandwidth-bound to compute-bound** due to dequantization ALU load — this is why latency improvements plateau faster than VRAM reductions suggest.

---

## Requirements

```
torch >= 2.0
transformers
bitsandbytes >= 0.46.1
accelerate
gradio
sentence-transformers
pandas
matplotlib
seaborn
``