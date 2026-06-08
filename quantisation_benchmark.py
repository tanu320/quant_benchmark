import os, gc, json, time, warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd

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

class ModelLoader:
    SUPPORTED_MODELS = {
        "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "Phi-3-mini":     "microsoft/Phi-3-mini-4k-instruct",
        "Qwen2.5-3B":     "Qwen/Qwen2.5-3B-Instruct",
    }

    def __init__(self, model_key: str = "TinyLlama-1.1B"):
        self.model_key = model_key
        self.model_id  = self.SUPPORTED_MODELS[model_key]
        self.model     = None
        self.tokenizer = None

    def load(self, precision: str) -> float:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        print(f"\n{'='*60}\nLoading {self.model_key} @ {precision}\n{'='*60}")
        self.unload()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        common_kwargs = dict(device_map="auto", trust_remote_code=True)

        if precision == "FP16":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, dtype=torch.float16, **common_kwargs)
        elif precision == "INT8":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                **common_kwargs)
        elif precision == "INT4":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                ),
                **common_kwargs)
        else:
            raise ValueError(f"Unknown precision '{precision}'. Choose from: FP16, INT8, INT4")

        self.model.eval()
        self.model.resize_token_embeddings(len(self.tokenizer))


        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._model_vram_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

        load_time = time.perf_counter() - t0
        print(f"Loaded in {load_time:.1f}s  |  Model VRAM: {self._model_vram_mb:.1f} MB")
        return load_time

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _apply_chat_template(tokenizer, prompt: str) -> str:
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class InferenceRunner:
    def __init__(self, model, tokenizer, max_new_tokens: int = 200):
        self.model          = model
        self.tokenizer      = tokenizer
        self.max_new_tokens = max_new_tokens
        self.device         = next(model.parameters()).device

    def run(self, prompt: str) -> dict:
        from transformers import LogitsProcessor, LogitsProcessorList
        formatted = _apply_chat_template(self.tokenizer, prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt", return_attention_mask=True).to(self.device)
        input_len = inputs["input_ids"].shape[1]
        first_token_time: list = [None]

        class TTFTProbe(LogitsProcessor):
            def __init__(self_inner):
                self_inner.step = 0
            def __call__(self_inner, input_ids, scores):
                if self_inner.step == 0:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    first_token_time[0] = time.perf_counter()
                self_inner.step += 1
                return scores

        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                logits_processor=LogitsProcessorList([TTFTProbe()]),
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_end = time.perf_counter()

        generated_ids    = outputs[0][input_len:]
        response         = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_tokens = len(generated_ids)
        total_latency_ms = (t_end - t_start) * 1000
        ttft_ms          = (first_token_time[0] - t_start) * 1000 if first_token_time[0] else total_latency_ms
        tps              = generated_tokens / (t_end - t_start) if (t_end - t_start) > 0 else 0.0

        return {
            "ttft_ms":          round(ttft_ms, 2),
            "tps":              round(tps, 2),
            "total_latency_ms": round(total_latency_ms, 2),
            "generated_tokens": generated_tokens,
            "response":         response,
        }


class MetricsCollector:

    @staticmethod
    def get_peak_vram_mb() -> float:
        return torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

    @staticmethod
    def reset_peak():
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def benchmark_performance(runner: InferenceRunner, n_runs: int = 5) -> dict:
        PROMPT = (
            "Explain the difference between model quantization and model pruning "
            "for LLM inference optimization. Be concise."
        )
        MetricsCollector.reset_peak()
        results = []

        print(f"  Running {n_runs} inference passes...")
        for i in range(n_runs):
            r = runner.run(PROMPT)
            results.append(r)
            print(f"    Pass {i+1}: TTFT={r['ttft_ms']:.1f}ms  TPS={r['tps']:.1f}  Tokens={r['generated_tokens']}")


        ttfts = [r["ttft_ms"] for r in results]
        if len(ttfts) >= 3:
            # check if last half is >10% slower than first half — sign of throttling
            mid = len(ttfts) // 2
            first_half_mean = sum(ttfts[:mid]) / mid
            last_half_mean  = sum(ttfts[mid:]) / (len(ttfts) - mid)
            if last_half_mean > first_half_mean * 1.10:
                print(f"  ⚠  TTFT climbing detected ({first_half_mean:.1f}ms → {last_half_mean:.1f}ms). "
                      f"Possible thermal throttle. Using passes 2–3 only for TTFT average.")
                warm = results[1:3]  # use the stable middle window
            else:
                warm = results[1:]   # normal: discard only warmup pass
        else:
            warm = results[1:] if len(results) > 1 else results

        return {
            "ttft_ms":          np.mean([r["ttft_ms"]          for r in warm]),
            "tps":              np.mean([r["tps"]               for r in warm]),
            "total_latency_ms": np.mean([r["total_latency_ms"]  for r in warm]),
            "generated_tokens": np.mean([r["generated_tokens"]  for r in warm]),
            "vram_mb":          MetricsCollector.get_peak_vram_mb(),
        }


class QualityEvaluator:
    def __init__(self, use_sentence_transformers: bool = True):
        self.encoder = None
        if use_sentence_transformers:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
                print("  ✓ Sentence-transformers loaded for semantic scoring")
            except ImportError:
                print("  ⚠ sentence-transformers not installed — falling back to Jaccard similarity")

    def score_response(self, response: str, prompt: str, baseline: str = None) -> dict:
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
        if baseline:
            if self.encoder:
                eb = self.encoder.encode([baseline])
                ec = self.encoder.encode([response])
                cos = float(np.dot(eb, ec.T) / (np.linalg.norm(eb) * np.linalg.norm(ec)))
                semantic_score = max(0.0, min(1.0, cos))
            else:
                a = set(baseline.lower().split())
                b = set(response.lower().split())
                semantic_score = len(a & b) / max(len(a | b), 1)

        return {
            "response_length": len(response.split()),
            "has_structure":   has_json or has_table or has_bullets or has_code,
            "is_complete":     is_complete,
            "semantic_score":  round(semantic_score, 4),
            "formatting_ok":   formatting_ok,
        }

    def run_suite(self, runner, precision, baseline_responses=None, max_per_category=3):
        results = []
        with torch.no_grad():
            for category, prompts in PROMPT_SUITE.items():
                print(f"  [{precision}] Category: {category}")
                for idx, prompt in enumerate(prompts[:max_per_category]):
                    r        = runner.run(prompt)
                    baseline = (baseline_responses or {}).get(category, {}).get(idx)
                    scores   = self.score_response(r["response"], prompt, baseline)
                    results.append({
                        "precision":  precision,
                        "category":   category,
                        "prompt_idx": idx,
                        "prompt":     prompt[:80] + "...",
                        "response":   r["response"],
                        **scores,
                    })
        return results


class QuantizationBenchmark:
    def __init__(
        self,
        model_key:                str  = "TinyLlama-1.1B",
        precisions:               list = None,
        n_perf_runs:              int  = 5,
        max_quality_per_category: int  = 3,
        output_dir:               str  = "./benchmark_results",
    ):
        self.model_key   = model_key
        self.precisions  = precisions or ["FP16", "INT8", "INT4"]
        self.n_perf_runs = n_perf_runs
        self.max_quality = max_quality_per_category
        self.output_dir  = output_dir
        self.loader      = ModelLoader(model_key)
        self.evaluator   = QualityEvaluator()
        os.makedirs(output_dir, exist_ok=True)
        self.perf_results:       list = []
        self.quality_results:    list = []
        self.baseline_responses: dict = {}

    def _collect_baseline(self, runner, precision):
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

            model_vram_mb = getattr(self.loader, "_model_vram_mb", 0.0)

            runner = InferenceRunner(self.loader.model, self.loader.tokenizer)
            perf   = MetricsCollector.benchmark_performance(runner, self.n_perf_runs)

            self.perf_results.append({
                "precision":        precision,
                "model":            self.model_key,
                "ttft_ms":          round(perf["ttft_ms"], 2),
                "tps":              round(perf["tps"], 2),
                "total_latency_ms": round(perf["total_latency_ms"], 2),
                "vram_mb":          round(model_vram_mb, 1),   # clean reading
                "load_time_s":      round(load_time, 2),
                "generated_tokens": int(perf["generated_tokens"]),
            })

            self._collect_baseline(runner, precision)
            self.quality_results.extend(
                self.evaluator.run_suite(runner, precision, self.baseline_responses, self.max_quality)
            )
            self.loader.unload()

        self._save()
        self._summary()
        return self.perf_results, self.quality_results

    def _save(self):
        with open(os.path.join(self.output_dir, "perf_results.json"), "w") as f:
            json.dump(self.perf_results, f, indent=2)
        with open(os.path.join(self.output_dir, "quality_results.json"), "w") as f:
            json.dump(
                [{k: v for k, v in r.items() if k != "response"} for r in self.quality_results],
                f, indent=2,
            )
        print(f"\n✓ Results saved to {self.output_dir}/")

    def _summary(self):
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
                   .round(3).to_string()
            )


def run_full_benchmark(
    model_key:   str  = "TinyLlama-1.1B",
    precisions:  list = None,
    n_perf_runs: int  = 5,
    output_dir:  str  = "./benchmark_results",
):
    if not torch.cuda.is_available():
        print("⚠  No CUDA detected — VRAM stats will be 0. Run on a GPU.")
    return QuantizationBenchmark(
        model_key=model_key,
        precisions=precisions or ["FP16", "INT8", "INT4"],
        n_perf_runs=n_perf_runs,
        output_dir=output_dir,
    ).run()


if __name__ == "__main__":
    run_full_benchmark()