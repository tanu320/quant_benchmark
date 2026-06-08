"""
LLM Quantization Tradeoff Analyzer
====================================
Run on Kaggle (GPU T4/P100) or Google Colab.

Install first:
    pip install transformers bitsandbytes accelerate torch pandas

Usage:
    python quantisation_benchmark.py
    # or import and call run_full_benchmark()
"""

import os, gc, json, time, warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd

# ── Prompt Suite ──────────────────────────────────────────────────────────────

PROMPT_SUITE = {
    "reasoning": [
        "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies? Explain step by step.",
        "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost? Show your reasoning.",
        "Three friends split a restaurant bill. Alice pays twice what Bob pays. Bob pays $5 more than Carol. Total is $85. How much does each pay?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
        "There are 23 people in a room. What is the probability two share a birthday? Is it above or below 50%? Explain intuitively.",
    ],
    "summarization": [
        "Summarize the key differences between TCP and UDP in under 100 words using a markdown table.",
        "Explain transformer attention mechanism to a software engineer in 3 bullet points.",
        "Summarize what quantization is and why it matters for LLM deployment in 4 sentences.",
        "Write a concise technical summary of gradient descent with momentum vs Adam optimizer.",
        "Summarize the CAP theorem for a distributed systems beginner in plain English, under 80 words.",
    ],
    "structured_output": [
        'Return a JSON object with keys: name, category, risk_level for "deploying INT4 quantized models in production".',
        "Generate a YAML config for a FastAPI service with fields: host, port, workers, log_level, model_path.",
        'Return a JSON array of 3 ML hyperparameters with fields: name, type, typical_range, effect_on_training.',
        'Output a markdown table comparing GPTQ, AWQ, and GGUF quantization formats across: speed, quality, tooling.',
        'Return JSON: {"steps": [...]} for fine-tuning a small LLM on domain data. Each step: name, description.',
    ],
    "sql": [
        "Write a SQL query to find the top 5 customers by total order value from tables: orders(id, customer_id, amount) and customers(id, name).",
        "Write a SQL window function to compute a 7-day rolling average of daily_sales from a sales table.",
        "Write a SQL query to find duplicate emails in a users table, showing email and count.",
        "Given tables: products(id, name, price) and inventory(product_id, quantity), write SQL to find products with quantity below 10.",
        "Write a SQL CTE that finds the second highest salary per department from an employees table.",
    ],
    "long_context": [
        "Given this context: 'The Attention mechanism was introduced in 2014 by Bahdanau et al., later refined in the 2017 Transformer paper by Vaswani et al.' — Who introduced attention and when?",
        "Context: 'RLHF stands for Reinforcement Learning from Human Feedback. It was popularized by InstructGPT (2022) and is used to align LLMs with human preferences.' — What year was RLHF popularized for LLMs?",
        "Context: 'FP16 uses 2 bytes per weight, INT8 uses 1 byte, INT4 uses 0.5 bytes.' — How many GB of VRAM does a 7B parameter model need in each precision?",
        "Context: 'LoRA adds trainable rank decomposition matrices to frozen weights. It was introduced by Hu et al. in 2021.' — What is the core idea of LoRA and who proposed it?",
        "Context: 'Speculative decoding uses a small draft model to propose tokens verified by the larger model.' — What is the role of the draft model in speculative decoding?",
    ],
    "hallucination_sensitive": [
        "Who is the CEO of Anthropic? Answer only if you are certain.",
        "What is the exact parameter count of Llama 3.1 8B? Be precise.",
        "When exactly was GPT-4 released? Provide only the date you are confident about.",
        "What is the context window of Mistral 7B v0.1? If unsure, say so.",
        "Name the paper that introduced the Mixture of Experts architecture for LLMs. Be specific.",
    ],
}

# ── Model Loader ───────────────────────────────────────────────────────────────

class ModelLoader:
    """Loads HuggingFace models in FP16, INT8, INT4 via bitsandbytes."""

    SUPPORTED_MODELS = {
        "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "Phi-3-mini":     "microsoft/Phi-3-mini-4k-instruct",
        "Qwen2.5-3B":     "Qwen/Qwen2.5-3B-Instruct",
    }

    def __init__(self, model_key: str = "TinyLlama-1.1B"):
        self.model_key = model_key
        self.model_id = self.SUPPORTED_MODELS[model_key]
        self.model = None
        self.tokenizer = None

    def load(self, precision: str) -> float:
        """Load model at given precision. Returns load time in seconds.
        
        Args:
            precision: One of 'FP16' | 'INT8' | 'INT4'
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        print(f"\n{'='*60}\nLoading {self.model_key} @ {precision}\n{'='*60}")

        self.unload()
        t0 = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        # Do NOT alias pad_token to eos_token — the model treats eos_token_id
        # as a stop signal during generation. If pad==eos, passing pad_token_id
        # to generate() causes the model to stop after 1 token.
        # A dedicated [PAD] token avoids this entirely.
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        common_kwargs = dict(device_map="auto", trust_remote_code=True)

        if precision == "FP16":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, dtype=torch.float16, **common_kwargs
            )
        elif precision == "INT8":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                **common_kwargs,
            )
        elif precision == "INT4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,  # quantize scale factors too (~0.4 bpw saving)
                bnb_4bit_quant_type="nf4",        # NormalFloat4: optimal for normally distributed weights
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, quantization_config=bnb_config, **common_kwargs
            )
        else:
            raise ValueError(f"Unknown precision '{precision}'. Choose from: FP16, INT8, INT4")

        self.model.eval()
        # If we added a new [PAD] token, resize the embedding table to match
        self.model.resize_token_embeddings(len(self.tokenizer))
        load_time = time.perf_counter() - t0
        print(f"Loaded in {load_time:.1f}s")
        return load_time

    def unload(self):
        """Release model and tokenizer, free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Inference Runner ───────────────────────────────────────────────────────────

class InferenceRunner:
    def __init__(self, model, tokenizer, max_new_tokens: int = 200):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        # Use first parameter's device as the input device.
        # Note: with device_map="auto" on multi-GPU, layers may span devices;
        # inputs go to the device of the embedding layer (always first).
        self.device = next(model.parameters()).device

    def run(self, prompt: str) -> dict:
        """Run a single inference pass. Returns timing, throughput, and response."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=True,  # explicit mask avoids pad/eos ambiguity
        ).to(self.device)
        input_len = inputs["input_ids"].shape[1]

        first_token_time: list[float | None] = [None]

        # Use a LogitsProcessor as a timing hook — the correct abstraction
        # for observing generation step-by-step without interfering with stopping.
        from transformers import LogitsProcessor, LogitsProcessorList

        class TTFTProbe(LogitsProcessor):
            """Records wall-clock time of the first decode step."""
            def __init__(self_inner):
                self_inner.step = 0

            def __call__(self_inner, input_ids, scores):
                if self_inner.step == 0:
                    # Sync GPU so the timestamp reflects actual compute completion
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    first_token_time[0] = time.perf_counter()
                self_inner.step += 1
                return scores  # pass logits through unchanged

        with torch.no_grad():
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_start = time.perf_counter()

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,                         # greedy — reproducible across runs
                logits_processor=LogitsProcessorList([TTFTProbe()]),
            )

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_end = time.perf_counter()

        generated_ids = outputs[0][input_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_tokens = len(generated_ids)

        total_latency_ms = (t_end - t_start) * 1000
        ttft_ms = (first_token_time[0] - t_start) * 1000 if first_token_time[0] else total_latency_ms
        tps = generated_tokens / (t_end - t_start) if (t_end - t_start) > 0 else 0.0

        return {
            "ttft_ms":           round(ttft_ms, 2),
            "tps":               round(tps, 2),
            "total_latency_ms":  round(total_latency_ms, 2),
            "generated_tokens":  generated_tokens,
            "response":          response,
        }


# ── Metrics Collector ──────────────────────────────────────────────────────────

class MetricsCollector:

    @staticmethod
    def get_peak_vram_mb() -> float:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
        return 0.0

    @staticmethod
    def reset_peak():
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def benchmark_performance(runner: InferenceRunner, n_runs: int = 5) -> dict:
        """Run a fixed prompt N times and return averaged performance stats.
        
        The first run is discarded to eliminate JIT compilation and cold-cache effects.
        Peak VRAM is measured across all runs (including the warmup).
        """
        BENCHMARK_PROMPT = (
            "Explain the difference between model quantization and model pruning "
            "for LLM inference optimization. Be concise."
        )

        MetricsCollector.reset_peak()
        results = []

        print(f"  Running {n_runs} inference passes...")
        for i in range(n_runs):
            r = runner.run(BENCHMARK_PROMPT)
            results.append(r)
            print(f"    Pass {i+1}: TTFT={r['ttft_ms']:.1f}ms  TPS={r['tps']:.1f}  Tokens={r['generated_tokens']}")

        # Discard run 0: first pass pays JIT compilation + cold GPU/HBM cache cost
        warm = results[1:] if len(results) > 1 else results
        return {
            "ttft_ms":           np.mean([r["ttft_ms"] for r in warm]),
            "tps":               np.mean([r["tps"] for r in warm]),
            "total_latency_ms":  np.mean([r["total_latency_ms"] for r in warm]),
            "generated_tokens":  np.mean([r["generated_tokens"] for r in warm]),
            "vram_mb":           MetricsCollector.get_peak_vram_mb(),
        }


# ── Quality Evaluator ──────────────────────────────────────────────────────────

class QualityEvaluator:
    """
    Evaluates INT8/INT4 response quality relative to the FP16 baseline.

    Primary signal: cosine similarity of sentence-transformer embeddings.
    Fallback (no sentence-transformers): Jaccard word overlap (weaker — ignores order).
    Structural signals: JSON/table/bullet/code detection, completeness, format compliance.
    
    Limitation: semantic score measures deviation from FP16, not absolute correctness.
    If FP16 itself hallucinates, lower precisions that agree will score high.
    """

    def __init__(self, use_sentence_transformers: bool = True):
        self.encoder = None
        if use_sentence_transformers:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
                print("  ✓ Sentence-transformers loaded for semantic scoring")
            except ImportError:
                print("  ⚠ sentence-transformers not installed — falling back to Jaccard similarity")

    def score_response(self, response: str, prompt: str, baseline_response: str = None) -> dict:
        """Score a single response against the FP16 baseline."""
        import re

        has_json    = bool(re.search(r'\{.*?\}', response, re.DOTALL))
        has_table   = "|" in response and "---" in response
        has_bullets = bool(re.search(r'^\s*[-*•]', response, re.MULTILINE))
        has_code    = "```" in response or "    " in response[:50]

        is_complete = (
            response.strip().endswith((".", "!", "?", "```", "}"))
            or len(response.split()) > 30
        )

        formatting_ok = True
        if "json"   in prompt.lower() and not has_json:    formatting_ok = False
        if "table"  in prompt.lower() and not has_table:   formatting_ok = False
        if "bullet" in prompt.lower() and not has_bullets: formatting_ok = False

        semantic_score = 1.0
        if baseline_response:
            if self.encoder:
                emb_base = self.encoder.encode([baseline_response])
                emb_curr = self.encoder.encode([response])
                cos_sim = float(
                    np.dot(emb_base, emb_curr.T)
                    / (np.linalg.norm(emb_base) * np.linalg.norm(emb_curr))
                )
                semantic_score = max(0.0, min(1.0, cos_sim))
            else:
                # Jaccard fallback: ignores word order and synonyms — treat as approximate
                a = set(baseline_response.lower().split())
                b = set(response.lower().split())
                semantic_score = len(a & b) / max(len(a | b), 1)

        return {
            "response_length": len(response.split()),
            "has_structure":   has_json or has_table or has_bullets or has_code,
            "is_complete":     is_complete,
            "semantic_score":  round(semantic_score, 4),
            "formatting_ok":   formatting_ok,
        }

    def run_suite(
        self,
        runner: InferenceRunner,
        precision: str,
        baseline_responses: dict = None,
        max_per_category: int = 3,
    ) -> list:
        """Run the full prompt suite and return a list of scored result dicts."""
        results = []
        with torch.no_grad():
            for category, prompts in PROMPT_SUITE.items():
                print(f"  [{precision}] Category: {category}")
                for idx, prompt in enumerate(prompts[:max_per_category]):
                    r = runner.run(prompt)
                    baseline = (baseline_responses or {}).get(category, {}).get(idx)
                    scores = self.score_response(r["response"], prompt, baseline)
                    results.append({
                        "precision":   precision,
                        "category":    category,
                        "prompt_idx":  idx,
                        "prompt":      prompt[:80] + "...",
                        "response":    r["response"],
                        **scores,
                    })
        return results


# ── Full Benchmark Orchestrator ────────────────────────────────────────────────

class QuantizationBenchmark:
    def __init__(
        self,
        model_key: str = "TinyLlama-1.1B",
        precisions: list = None,
        n_perf_runs: int = 5,
        max_quality_per_category: int = 3,
        output_dir: str = "./benchmark_results",
    ):
        self.model_key   = model_key
        self.precisions  = precisions or ["FP16", "INT8", "INT4"]
        self.n_perf_runs = n_perf_runs
        self.max_quality = max_quality_per_category
        self.output_dir  = output_dir
        self.loader      = ModelLoader(model_key)
        self.evaluator   = QualityEvaluator()

        os.makedirs(output_dir, exist_ok=True)

        self.perf_results:       list[dict] = []
        self.quality_results:    list[dict] = []
        self.baseline_responses: dict       = {}

    def _collect_baseline_responses(self, runner: InferenceRunner, precision: str):
        """Cache FP16 outputs so lower-precision runs can compute semantic similarity."""
        if precision != "FP16":
            return
        for category, prompts in PROMPT_SUITE.items():
            self.baseline_responses[category] = {}
            for idx, prompt in enumerate(prompts[:self.max_quality]):
                self.baseline_responses[category][idx] = runner.run(prompt)["response"]

    def run(self):
        for precision in self.precisions:
            print(f"\n{'#'*60}\n  PRECISION: {precision}\n{'#'*60}")

            load_time = self.loader.load(precision)
            runner    = InferenceRunner(self.loader.model, self.loader.tokenizer)

            perf = MetricsCollector.benchmark_performance(runner, self.n_perf_runs)
            self.perf_results.append({
                "precision":        precision,
                "model":            self.model_key,
                "ttft_ms":          round(perf["ttft_ms"], 2),
                "tps":              round(perf["tps"], 2),
                "total_latency_ms": round(perf["total_latency_ms"], 2),
                "vram_mb":          round(perf["vram_mb"], 1),
                "load_time_s":      round(load_time, 2),
                "generated_tokens": int(perf["generated_tokens"]),
            })

            self._collect_baseline_responses(runner, precision)

            self.quality_results.extend(
                self.evaluator.run_suite(runner, precision, self.baseline_responses, self.max_quality)
            )

            self.loader.unload()

        self._save_results()
        self._print_summary()
        return self.perf_results, self.quality_results

    def _save_results(self):
        with open(os.path.join(self.output_dir, "perf_results.json"), "w") as f:
            json.dump(self.perf_results, f, indent=2)
        with open(os.path.join(self.output_dir, "quality_results.json"), "w") as f:
            # Omit full responses to keep file size manageable
            json.dump(
                [{k: v for k, v in r.items() if k != "response"} for r in self.quality_results],
                f, indent=2,
            )
        print(f"\n✓ Results saved to {self.output_dir}/")

    def _print_summary(self):
        print("\n" + "="*60 + "\nBENCHMARK SUMMARY\n" + "="*60)
        df = pd.DataFrame(self.perf_results)
        print(df[["precision", "vram_mb", "ttft_ms", "tps", "total_latency_ms"]].to_string(index=False))

        qdf = pd.DataFrame([{k: v for k, v in r.items() if k != "response"} for r in self.quality_results])
        if not qdf.empty:
            print("\nQuality Summary:")
            print(
                qdf.groupby("precision")
                   .agg(semantic_score=("semantic_score", "mean"),
                        formatting_ok=("formatting_ok", "mean"),
                        is_complete=("is_complete", "mean"))
                   .round(3)
                   .to_string()
            )


# ── Entry Point ────────────────────────────────────────────────────────────────

def run_full_benchmark(
    model_key:   str  = "TinyLlama-1.1B",
    precisions:  list = None,
    n_perf_runs: int  = 5,
    output_dir:  str  = "./benchmark_results",
):
    """
    Main entry point.

    Args:
        model_key:   One of TinyLlama-1.1B | Phi-3-mini | Qwen2.5-3B
        precisions:  Subset of ["FP16", "INT8", "INT4"] or None for all three
        n_perf_runs: Inference passes per precision (first is discarded as warmup)
        output_dir:  Directory for JSON result files
    """
    if not torch.cuda.is_available():
        print("⚠  No CUDA detected — VRAM stats will be 0. Run on a GPU for meaningful results.")

    return QuantizationBenchmark(
        model_key=model_key,
        precisions=precisions or ["FP16", "INT8", "INT4"],
        n_perf_runs=n_perf_runs,
        output_dir=output_dir,
    ).run()


if __name__ == "__main__":
    perf, quality = run_full_benchmark()
