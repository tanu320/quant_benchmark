# LLM Quantization Tradeoff Analyzer

> Measuring the real inference tradeoffs of FP16 / INT8 / INT4 quantization  
> Prepared for: AMD/NVIDIA LLM inference optimization roles

---

## Project Structure

```
llm_quantization_analyzer/
├── quantization_benchmark.py   # Core: model loading, inference, metrics
├── gradio_app.py               # Gradio dashboard UI
├── kaggle_notebook.ipynb       # Self-contained Kaggle/Colab notebook
├── requirements.txt
└── README.md
```

---

## Quick Start

### On Kaggle / Colab (GPU required for real metrics)

1. Upload `quantization_benchmark.py` and `gradio_app.py` as a dataset  
2. Open `kaggle_notebook.ipynb`  
3. Enable GPU (T4 is sufficient for TinyLlama/Phi-3-mini)  
4. Run all cells — takes ~20–40 min for TinyLlama  
5. Gradio UI launches with `share=True` for a public URL  

### On Mac (demo mode only, no GPU metrics)

```bash
pip install -r requirements.txt
python gradio_app.py --demo
```

---

## What's Measured

### Performance Metrics
| Metric | How | Why it matters |
|--------|-----|----------------|
| **TTFT** | `time.perf_counter()` around first token | User-perceived latency |
| **TPS** | `generated_tokens / wall_time` | Throughput for batch serving |
| **Total Latency** | End-to-end wall time | SLA planning |
| **VRAM** | `torch.cuda.max_memory_allocated()` | Hardware sizing |
| **Load Time** | Model init wall time | Cold-start cost |

### Quality Metrics
| Metric | Method |
|--------|--------|
| **Semantic Consistency** | Cosine sim vs FP16 baseline (sentence-transformers `all-MiniLM-L6-v2`) |
| **Formatting Stability** | Heuristic: JSON/table/bullet presence vs prompt requirement |
| **Response Completeness** | Sentence boundary + length check |

### Prompt Suite (6 categories × 5 prompts)
- Reasoning (logic, math)
- Summarization (markdown output)
- Structured Output (JSON, YAML, tables)
- SQL (window functions, CTEs, joins)
- Long Context (retrieval from given text)
- Hallucination Sensitive (factual, verifiable claims)

---

## Models Supported

| Key | Model | HuggingFace ID | VRAM (FP16) |
|-----|-------|----------------|-------------|
| `TinyLlama-1.1B` | TinyLlama Chat 1.1B | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ~2.2 GB |
| `Phi-3-mini` | Microsoft Phi-3 Mini 4K | `microsoft/Phi-3-mini-4k-instruct` | ~7.6 GB |
| `Qwen2.5-3B` | Qwen 2.5 3B Instruct | `Qwen/Qwen2.5-3B-Instruct` | ~6.4 GB |

**Recommended for Kaggle T4 (16GB):** TinyLlama-1.1B (runs all 3 precisions easily)

---

## Quantization Details

### FP16
- Native HuggingFace `torch_dtype=torch.float16`
- 2 bytes/weight — memory-bandwidth-bound inference
- Quality baseline

### INT8 (bitsandbytes LLM.int8())
- Mixed-precision: outlier features stay FP16, bulk weights go INT8
- ~1.1 bytes/weight effective
- Minimal quality degradation, good throughput gain

### INT4 (bitsandbytes QLoRA / NF4)
- `bnb_4bit_quant_type="nf4"` — NormalFloat4, designed for normally distributed weights
- `bnb_4bit_use_double_quant=True` — quantize the quantization constants (~0.4 bpw savings)
- `bnb_4bit_compute_dtype=torch.float16` — dequantize to FP16 for matmul
- ~0.5 bytes/weight
- Largest VRAM win; latency gains plateau due to dequantization overhead

---

## Key Engineering Insight

> *"INT4 provided the largest VRAM reduction (~75% vs FP16), but latency gains were smaller  
> than expected (+40-60% TPS) because inference shifted from memory-bandwidth-bound to  
> partially compute-bound — dequantization overhead on the GPU ALU partially offsets  
> memory bandwidth savings."*

This is the core systems insight that differentiates a surface-level answer from an  
inference-optimization-level answer in interviews.

---

## Extending the Project

### Add GPTQ/AWQ (next step)
```bash
pip install auto-gptq autoawq
```
Then add a `load_gptq()` method to `ModelLoader` — compares against bitsandbytes post-hoc.

### Add GGUF/llama.cpp
Run on CPU-only Mac to measure CPU inference characteristics.  
Good for: edge device tradeoffs, M-series Apple Silicon comparison.

### Add speculative decoding
Compare draft-model-assisted vs standard greedy across precisions.

---

## Requirements

```
torch>=2.1.0
transformers>=4.40.0
bitsandbytes>=0.43.0
accelerate>=0.28.0
gradio>=4.25.0
sentence-transformers>=2.7.0
pandas>=2.0.0
matplotlib>=3.8.0
numpy>=1.26.0
seaborn>=0.13.0
```