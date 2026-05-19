"""
LLM Quantization Tradeoff Analyzer — Gradio UI
================================================
Run after quantization_benchmark.py has produced results.
Or run with demo mode (synthetic data) if no GPU results yet.

Launch:
    python gradio_app.py
    python gradio_app.py --demo        # uses synthetic data
    python gradio_app.py --results ./benchmark_results
"""

import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import gradio as gr

# ── Color palette ──────────────────────────────────────────────────────────────
COLORS = {
    "FP16": "#4C9BE8",   # cool blue
    "INT8": "#F5A623",   # amber
    "INT4": "#7ED321",   # green
    "bg":   "#0F1117",
    "fg":   "#E8EAF0",
    "grid": "#2A2D3A",
}

plt.rcParams.update({
    "figure.facecolor":  COLORS["bg"],
    "axes.facecolor":    "#1A1D2E",
    "axes.edgecolor":    COLORS["grid"],
    "axes.labelcolor":   COLORS["fg"],
    "xtick.color":       COLORS["fg"],
    "ytick.color":       COLORS["fg"],
    "text.color":        COLORS["fg"],
    "grid.color":        COLORS["grid"],
    "grid.alpha":        0.4,
    "font.family":       "monospace",
})

# ── Synthetic demo data ────────────────────────────────────────────────────────

DEMO_PERF = [
    {"precision":"FP16","model":"TinyLlama-1.1B","ttft_ms":312,"tps":48.2,"total_latency_ms":4500,"vram_mb":2240,"load_time_s":8.1,"generated_tokens":190},
    {"precision":"INT8","model":"TinyLlama-1.1B","ttft_ms":248,"tps":61.5,"total_latency_ms":3530,"vram_mb":1340,"load_time_s":9.4,"generated_tokens":192},
    {"precision":"INT4","model":"TinyLlama-1.1B","ttft_ms":221,"tps":74.8,"total_latency_ms":2960,"vram_mb":780, "load_time_s":10.2,"generated_tokens":188},
]

DEMO_QUALITY_SUMMARY = {
    "FP16": {"semantic_score":1.00, "formatting_ok":0.93, "is_complete":0.97},
    "INT8": {"semantic_score":0.94, "formatting_ok":0.90, "is_complete":0.95},
    "INT4": {"semantic_score":0.87, "formatting_ok":0.83, "is_complete":0.91},
}

DEMO_CATEGORY_QUALITY = {
    "reasoning":           {"FP16":0.96,"INT8":0.91,"INT4":0.82},
    "summarization":       {"FP16":0.98,"INT8":0.95,"INT4":0.91},
    "structured_output":   {"FP16":0.93,"INT8":0.89,"INT4":0.79},
    "sql":                 {"FP16":0.97,"INT8":0.93,"INT4":0.85},
    "long_context":        {"FP16":0.95,"INT8":0.92,"INT4":0.88},
    "hallucination_sensitive":{"FP16":0.91,"INT8":0.87,"INT4":0.74},
}

# ── Data loader ────────────────────────────────────────────────────────────────

def load_results(results_dir: str):
    perf_path = os.path.join(results_dir, "perf_results.json")
    qual_path = os.path.join(results_dir, "quality_results.json")

    if not os.path.exists(perf_path):
        return None, None

    with open(perf_path) as f:
        perf = json.load(f)

    quality_summary = {p: {"semantic_score":0,"formatting_ok":0,"is_complete":0}
                       for p in ["FP16","INT8","INT4"]}
    category_quality = {}

    if os.path.exists(qual_path):
        with open(qual_path) as f:
            qual_raw = json.load(f)
        df = pd.DataFrame(qual_raw)
        if not df.empty:
            for prec, grp in df.groupby("precision"):
                quality_summary[prec] = {
                    "semantic_score": grp["semantic_score"].mean(),
                    "formatting_ok":  grp["formatting_ok"].mean(),
                    "is_complete":    grp["is_complete"].mean(),
                }
            for cat, grp in df.groupby("category"):
                category_quality[cat] = {
                    prec: sub["semantic_score"].mean()
                    for prec, sub in grp.groupby("precision")
                }

    return perf, (quality_summary, category_quality)


# ── Chart generators ───────────────────────────────────────────────────────────

def fig_perf_bars(perf):
    precs  = [p["precision"] for p in perf]
    colors = [COLORS[p] for p in precs]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Performance Metrics by Precision", fontsize=13, fontweight="bold", y=1.02)

    metrics = [
        ("vram_mb",          "VRAM Usage (MB)",       True),
        ("ttft_ms",          "Time to First Token (ms)", True),
        ("tps",              "Tokens per Second",      False),
    ]

    for ax, (key, label, lower_better) in zip(axes, metrics):
        vals = [p[key] for p in perf]
        bars = ax.bar(precs, vals, color=colors, width=0.5, edgecolor="none", alpha=0.9)

        # Annotate bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # Baseline relative annotation
        baseline = vals[0]
        for i, (bar, val) in enumerate(zip(bars, vals)):
            if i > 0:
                delta = ((val - baseline) / baseline) * 100
                sign  = "+" if delta > 0 else ""
                color = "#7ED321" if (delta < 0 and lower_better) or (delta > 0 and not lower_better) else "#E85454"
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                        f"{sign}{delta:.0f}%", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

        ax.set_title(label, fontsize=10, pad=8)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top","right","left"]].set_visible(False)

    plt.tight_layout()
    return fig


def fig_vram_breakdown(perf):
    fig, ax = plt.subplots(figsize=(8, 4))

    precs = [p["precision"] for p in perf]
    vrams = [p["vram_mb"] for p in perf]
    colors = [COLORS[p] for p in precs]

    bars = ax.barh(precs, vrams, color=colors, height=0.45, edgecolor="none")
    ax.set_xlabel("VRAM (MB)", fontsize=10)
    ax.set_title("VRAM Footprint — Memory Bottleneck Analysis", fontsize=11, fontweight="bold")

    base = vrams[0]
    for bar, val, prec in zip(bars, vrams, precs):
        reduction = (1 - val/base) * 100
        label = f"{val:.0f} MB" + (f"  ↓{reduction:.0f}% vs FP16" if val < base else "  (baseline)")
        ax.text(val + base*0.01, bar.get_y() + bar.get_height()/2,
                label, va="center", fontsize=9)

    # Annotate compute vs memory bottleneck region
    ax.axvline(x=vrams[0]*0.6, color="#E85454", linestyle="--", alpha=0.6, linewidth=1)
    ax.text(vrams[0]*0.61, 0.05, "memory-bound\n← threshold", color="#E85454",
            fontsize=7, transform=ax.get_xaxis_transform(), va="bottom")

    ax.set_xlim(0, base * 1.3)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top","right","bottom"]].set_visible(False)
    plt.tight_layout()
    return fig


def fig_quality_radar(quality_summary):
    categories = ["semantic_score", "formatting_ok", "is_complete"]
    labels     = ["Semantic\nConsistency", "Formatting\nStability", "Response\nCompleteness"]
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_facecolor("#1A1D2E")
    fig.patch.set_facecolor(COLORS["bg"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.set_ylim(0.6, 1.0)
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.7","0.8","0.9","1.0"], size=7)
    ax.grid(color=COLORS["grid"], alpha=0.5)

    for prec, scores in quality_summary.items():
        vals = [scores[c] for c in categories]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, color=COLORS[prec], label=prec)
        ax.fill(angles, vals, color=COLORS[prec], alpha=0.15)

    ax.set_title("Quality Tradeoff Radar", fontsize=11, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    return fig


def fig_category_heatmap(category_quality):
    cats  = list(category_quality.keys())
    precs = ["FP16", "INT8", "INT4"]
    data  = np.array([[category_quality[c].get(p, np.nan) for p in precs] for c in cats])

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0.65, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(precs)))
    ax.set_xticklabels(precs, fontsize=10)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([c.replace("_", " ").title() for c in cats], fontsize=9)
    ax.set_title("Semantic Score by Category & Precision", fontsize=11, fontweight="bold")

    for i in range(len(cats)):
        for j in range(len(precs)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="black" if v > 0.82 else "white")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Semantic Score")
    plt.tight_layout()
    return fig


def fig_tradeoff_scatter(perf, quality_summary):
    fig, ax = plt.subplots(figsize=(8, 5))

    for p in perf:
        prec = p["precision"]
        x    = p["tps"]
        y    = quality_summary[prec]["semantic_score"]
        size = 1000 / (p["vram_mb"] / 500 + 0.5)   # bubble ~ inverse VRAM

        ax.scatter(x, y, s=size, color=COLORS[prec], alpha=0.85,
                   edgecolors="white", linewidths=1.5, zorder=3)
        ax.annotate(f"{prec}\n{p['vram_mb']:.0f}MB",
                    (x, y), textcoords="offset points", xytext=(10, 5),
                    fontsize=9, color=COLORS[prec], fontweight="bold")

    ax.set_xlabel("Throughput (Tokens / second)", fontsize=10)
    ax.set_ylabel("Semantic Quality Score", fontsize=10)
    ax.set_title("Speed vs Quality Tradeoff  (bubble size ∝ 1/VRAM)", fontsize=11, fontweight="bold")

    # Pareto annotation
    ax.text(0.98, 0.05, "↑ Better quality\n→ Higher throughput",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=COLORS["fg"], alpha=0.6)

    ax.grid(alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    return fig


def fig_latency_decomposition(perf):
    """TTFT vs generation latency decomposition."""
    fig, ax = plt.subplots(figsize=(9, 4))

    precs = [p["precision"] for p in perf]
    ttfts = [p["ttft_ms"] for p in perf]
    rest  = [p["total_latency_ms"] - p["ttft_ms"] for p in perf]

    x = np.arange(len(precs))
    w = 0.4

    b1 = ax.bar(x, ttfts, w, label="TTFT (prefill)", color=[COLORS[p] for p in precs], alpha=0.9)
    b2 = ax.bar(x, rest,  w, bottom=ttfts, label="Generation latency",
                color=[COLORS[p] for p in precs], alpha=0.45, hatch="///")

    ax.set_xticks(x)
    ax.set_xticklabels(precs)
    ax.set_ylabel("Latency (ms)", fontsize=10)
    ax.set_title("Latency Decomposition: Prefill vs Generation", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    return fig


# ── Key Findings Generator ─────────────────────────────────────────────────────

def generate_findings(perf, quality_summary):
    fp16 = next(p for p in perf if p["precision"] == "FP16")
    int4 = next((p for p in perf if p["precision"] == "INT4"), None)
    int8 = next((p for p in perf if p["precision"] == "INT8"), None)

    findings = []

    if int4:
        vram_red = (1 - int4["vram_mb"] / fp16["vram_mb"]) * 100
        tps_gain = (int4["tps"] / fp16["tps"] - 1) * 100
        qual_drop = (1 - quality_summary["INT4"]["semantic_score"]) * 100

        findings.append(
            f"**VRAM Reduction (INT4):** {vram_red:.0f}% reduction vs FP16 "
            f"({fp16['vram_mb']:.0f}MB → {int4['vram_mb']:.0f}MB). "
            f"This is the dominant win for memory-constrained deployments."
        )
        findings.append(
            f"**Throughput Gain (INT4):** +{tps_gain:.0f}% TPS vs FP16 "
            f"({fp16['tps']:.1f} → {int4['tps']:.1f} tok/s). "
            f"Gains are real but smaller than VRAM reduction suggests — "
            f"dequantization overhead partially offsets compute savings."
        )
        findings.append(
            f"**Quality Cost (INT4):** Semantic drift of ~{qual_drop:.1f}% on average. "
            f"Hallucination-sensitive tasks show the steepest degradation — "
            f"not recommended for fact-critical applications without validation."
        )

    if int8:
        vram_red_8 = (1 - int8["vram_mb"] / fp16["vram_mb"]) * 100
        qual_drop_8 = (1 - quality_summary["INT8"]["semantic_score"]) * 100
        findings.append(
            f"**INT8 Sweet Spot:** {vram_red_8:.0f}% VRAM reduction with only "
            f"~{qual_drop_8:.1f}% quality drop. Best precision/quality tradeoff "
            f"for production systems that can't afford quality regression."
        )

    findings.append(
        "**Compute vs Memory Bottleneck:** At INT4, inference shifts from "
        "memory-bandwidth-bound to partially compute-bound (dequantization adds ALU load). "
        "This is why latency gains plateau faster than VRAM gains."
    )
    findings.append(
        "**Recommendation:** Use INT8 for production LLM APIs. "
        "Use INT4 only for edge/mobile deployments or when VRAM is the hard constraint. "
        "FP16 remains the baseline for quality-critical or long-form generation tasks."
    )

    return "\n\n".join(f"{i+1}. {f}" for i, f in enumerate(findings))


def build_tradeoff_table(perf, quality_summary):
    rows = []
    fp16_perf = next(p for p in perf if p["precision"] == "FP16")

    for p in perf:
        prec = p["precision"]
        vram_delta = f"−{(1-p['vram_mb']/fp16_perf['vram_mb'])*100:.0f}%" if prec != "FP16" else "baseline"
        tps_delta  = f"+{(p['tps']/fp16_perf['tps']-1)*100:.0f}%"         if prec != "FP16" else "baseline"
        qual = quality_summary[prec]["semantic_score"]

        rows.append({
            "Precision": prec,
            "VRAM (MB)": f"{p['vram_mb']:.0f}",
            "VRAM Δ":    vram_delta,
            "TPS":       f"{p['tps']:.1f}",
            "TPS Δ":     tps_delta,
            "TTFT (ms)": f"{p['ttft_ms']:.0f}",
            "Semantic Score": f"{qual:.3f}",
            "Rec. Use Case": {
                "FP16": "Quality-critical, research",
                "INT8": "Production APIs",
                "INT4": "Edge / VRAM-constrained",
            }[prec],
        })

    return pd.DataFrame(rows)


# ── Gradio App ─────────────────────────────────────────────────────────────────

def build_app(results_dir: str = None, demo_mode: bool = False):
    # Load or use demo data
    if demo_mode or results_dir is None:
        perf = DEMO_PERF
        quality_summary, category_quality = DEMO_QUALITY_SUMMARY, DEMO_CATEGORY_QUALITY
        data_source = "Demo data (synthetic). Run `quantization_benchmark.py` to get real GPU results."
    else:
        loaded_perf, loaded_qual = load_results(results_dir)
        if loaded_perf is None:
            perf = DEMO_PERF
            quality_summary, category_quality = DEMO_QUALITY_SUMMARY, DEMO_CATEGORY_QUALITY
            data_source = "No results found in directory. Showing demo data."
        else:
            perf = loaded_perf
            quality_summary, category_quality = loaded_qual
            data_source = f"Real GPU results loaded from `{results_dir}`"

    findings    = generate_findings(perf, quality_summary)
    tradeoff_df = build_tradeoff_table(perf, quality_summary)

    css = """
    .gradio-container { background: #0F1117; }
    .tab-nav { background: #1A1D2E !important; }
    h1, h2, h3 { color: #E8EAF0 !important; }
    .markdown-body { color: #C8CAD4 !important; }
    """

    with gr.Blocks(title="LLM Quantization Tradeoff Analyzer", css=css, theme=gr.themes.Default()) as app:

        gr.Markdown(f"""
# ⚡ LLM Quantization Tradeoff Analyzer
**Measuring FP16 / INT8 / INT4 across performance, quality, and systems efficiency**

_{data_source}_
""")

        with gr.Tabs():

            # ── Tab 1: Performance ──
            with gr.Tab("Performance"):
                gr.Markdown("### Inference Performance Across Precisions")
                gr.Plot(value=fig_perf_bars(perf))
                gr.Plot(value=fig_latency_decomposition(perf))
                gr.Plot(value=fig_vram_breakdown(perf))

            # ── Tab 2: Quality ──
            with gr.Tab("Quality"):
                gr.Markdown("### Response Quality vs Baseline (FP16)")
                with gr.Row():
                    gr.Plot(value=fig_quality_radar(quality_summary))
                    gr.Plot(value=fig_category_heatmap(category_quality))

            # ── Tab 3: Tradeoff ──
            with gr.Tab("Tradeoff Analysis"):
                gr.Markdown("### Speed vs Quality vs Memory — The Core Tradeoff")
                gr.Plot(value=fig_tradeoff_scatter(perf, quality_summary))

                gr.Markdown("### Summary Matrix")
                gr.Dataframe(
                    value=tradeoff_df,
                    interactive=False,
                    wrap=True,
                )

            # ── Tab 4: Findings ──
            with gr.Tab("Key Findings"):
                gr.Markdown(f"""
### Engineering Findings

{findings}

---
### Systems Insight: Compute vs Memory Bottleneck

| Precision | Primary Bottleneck | Implication |
|---|---|---|
| FP16 | Memory bandwidth | Throughput scales with HBM bandwidth |
| INT8 | Memory bandwidth (lighter) | ~2× effective bandwidth, minimal overhead |
| INT4 | Mixed (compute + memory) | Dequantization adds ALU cost; gains plateau |

> *"INT4 provided the largest VRAM reduction, but latency gains were smaller than expected
> because inference shifted to partially compute-bound — dequantization overhead
> on the GPU's ALU units partially offsets memory bandwidth savings."*
""")

            # ── Tab 5: Live Inference ──
            with gr.Tab("Live Inference"):
                gr.Markdown("""
### Run Live Inference
*This requires a GPU session with models loaded. Best used on Kaggle/Colab.*
""")
                gr.Markdown("""
**To enable live inference:**
1. Run `quantization_benchmark.py` on Kaggle/Colab (GPU)
2. Modify this tab to call your loaded model directly
3. Or use the Kaggle notebook version which has live inference built in
""")

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",    action="store_true",  help="Use synthetic demo data")
    parser.add_argument("--results", type=str, default=None, help="Path to benchmark_results dir")
    parser.add_argument("--port",    type=int, default=7860)
    parser.add_argument("--share",   action="store_true",  help="Create public Gradio link")
    args = parser.parse_args()

    app = build_app(results_dir=args.results, demo_mode=args.demo)
    app.launch(server_port=args.port, share=args.share, show_error=True)