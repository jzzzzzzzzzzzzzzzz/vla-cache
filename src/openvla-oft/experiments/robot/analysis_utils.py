"""
analysis_utils.py

Data structures and visualization tools for token classification analysis.
Used to validate Phase 1.1 hypotheses before implementing Prune class.

Token classification (per camera, 256 patches each):
  Class A (Cache):    high pixel-sim AND low text-attention  → VLA-Cache caches these
  Class B (Prune):    low  pixel-sim AND low text-attention  → Our proposed Prune class
  Class C (Recompute): high pixel-sim AND high text-attention → Must recompute (task-relevant)
  Class D (Recompute): low  pixel-sim AND high text-attention → Must recompute (dynamic+relevant)

Key questions this tool answers:
  Q1. What fraction of tokens are Class B (prune candidates)? → Scheme A feasibility
  Q2. How low are Class B attention scores?                   → Pruning safety
  Q3. Is Class B membership stable across frames?             → Temporal consistency
  Q4. What fraction of VLA-Cache's task-relevant set has medium-low attention?
"""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# GPU memory management: streaming attention capture via forward hooks
# ─────────────────────────────────────────────────────────────────────────────

class AttentionHookCapture:
    """
    Context manager that captures LLaMA decoder-layer attention weights on-the-fly.

    Problem: output_attentions=True forces SDPA → eager attention fallback, accumulating
    32 layers × (1, 32, 605, 605) × fp32 = ~1.5 GB of GPU attention maps simultaneously
    before the forward pass returns.  On a 16 GB GPU running INT4 LLaMA-7B this causes OOM.

    Solution: Register a forward hook on each LlamaDecoderLayer. The hook immediately
    moves that layer's attention tensor to CPU and returns None in its place.  PyTorch's
    allocator can then reuse the freed ~47 MB GPU block for the NEXT layer's computation,
    so peak GPU memory from attention drops from 1.5 GB to ~47 MB.

    The reconstructed attentions tuple (all CPU tensors + original cache_position) is
    drop-in compatible with token_attention_merge and get_layer_mask_schedule, which
    already call .to(torch.float32) / .cpu() internally.

    Usage:
        with AttentionHookCapture(vla.language_model) as cap:
            action, _, last_caches = vla.predict_action(...)
        # last_caches['attentions'] still has the (None, ..., None, cache_pos) tuple
        # from LlamaModel; replace it with CPU tensors captured by the hooks:
        cache_pos = last_caches['attentions'][-1]          # real cache_position tensor
        last_caches['attentions'] = cap.make_attentions_tuple(cache_pos)
    """

    def __init__(self, llm_model, num_layers: int = 32):
        self.llm_model = llm_model
        self.num_layers = num_layers
        self.captured: List = [None] * num_layers
        self._handles: List = []

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, outputs):
            # LlamaDecoderLayer output with output_attentions=True, use_cache=True:
            #   (hidden_states, attn_weights, present_key_value)
            # attn_weights is a 4-D float tensor; present_key_value is a tuple/Cache.
            if (isinstance(outputs, tuple) and len(outputs) >= 2
                    and isinstance(outputs[1], torch.Tensor)
                    and outputs[1].dim() == 4):
                # Move to CPU immediately; GPU block becomes available for next layer.
                self.captured[layer_idx] = outputs[1].detach().cpu()
                return (outputs[0], None) + outputs[2:]
            return outputs
        return hook

    def __enter__(self):
        layers = self.llm_model.model.layers
        for i, layer in enumerate(layers[:self.num_layers]):
            handle = layer.register_forward_hook(self._make_hook(i))
            self._handles.append(handle)
        return self

    def __exit__(self, *args):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def make_attentions_tuple(self, cache_position):
        """
        Reconstruct the 33-element attentions tuple expected by the siyuhsu LLaMA fork:
          (attn_layer_0, ..., attn_layer_31, cache_position)
        All 32 attention tensors are CPU; cache_position is moved to CPU so that
        token_attention_merge can index into CPU attention maps without a device mismatch.
        """
        cpu_cache_pos = cache_position.cpu() if isinstance(cache_position, torch.Tensor) else cache_position
        return tuple(self.captured) + (cpu_cache_pos,)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameStats:
    """Statistics collected for one model-query step (every num_open_loop_steps env steps)."""

    # Metadata
    frame_idx: int          # model-query index within episode (0 = first query)
    task_id: int
    episode_idx: int

    # ── Primary camera (256 patches) ──────────────────────────────────────────
    primary_sim_scores: np.ndarray = field(default_factory=lambda: np.zeros(256))
    primary_attn_scores: np.ndarray = field(default_factory=lambda: np.zeros(256))
    primary_class_A: List[int] = field(default_factory=list)   # Cache
    primary_class_B: List[int] = field(default_factory=list)   # Prune candidate
    primary_class_C: List[int] = field(default_factory=list)   # Recompute (static+relevant)
    primary_class_D: List[int] = field(default_factory=list)   # Recompute (dynamic+relevant)
    primary_stable_count: int = 0                               # patches with sim >= threshold

    # ── Wrist camera (256 patches) ────────────────────────────────────────────
    wrist_sim_scores: np.ndarray = field(default_factory=lambda: np.zeros(256))
    wrist_attn_scores: np.ndarray = field(default_factory=lambda: np.zeros(256))
    wrist_class_A: List[int] = field(default_factory=list)
    wrist_class_B: List[int] = field(default_factory=list)
    wrist_class_C: List[int] = field(default_factory=list)
    wrist_class_D: List[int] = field(default_factory=list)
    wrist_stable_count: int = 0

    # ── Layer-adaptive reuse schedule ─────────────────────────────────────────
    proportion_attn_var: Optional[np.ndarray] = None  # (31,); None for frame_idx==0

    # ── Compute stats ─────────────────────────────────────────────────────────
    llm_flops: float = 0.0
    time_ms: float = 0.0

    # ── Derived convenience properties ───────────────────────────────────────

    @property
    def primary_counts(self) -> Dict[str, int]:
        return dict(A=len(self.primary_class_A), B=len(self.primary_class_B),
                    C=len(self.primary_class_C), D=len(self.primary_class_D))

    @property
    def wrist_counts(self) -> Dict[str, int]:
        return dict(A=len(self.wrist_class_A), B=len(self.wrist_class_B),
                    C=len(self.wrist_class_C), D=len(self.wrist_class_D))

    def primary_ratios(self) -> Dict[str, float]:
        total = 256.0
        c = self.primary_counts
        return {k: v / total for k, v in c.items()}

    def wrist_ratios(self) -> Dict[str, float]:
        total = 256.0
        c = self.wrist_counts
        return {k: v / total for k, v in c.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation
# ─────────────────────────────────────────────────────────────────────────────

def save_analysis_data(all_stats: List[FrameStats], path: str) -> None:
    """Pickle a list of FrameStats to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(all_stats, f)
    print(f"[analysis] Saved {len(all_stats)} FrameStats → {path}")


def load_analysis_data(path: str) -> List[FrameStats]:
    """Load pickled FrameStats."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"[analysis] Loaded {len(data)} FrameStats from {path}")
    return data


def export_summary_json(all_stats: List[FrameStats], path: str) -> None:
    """Save a human-readable JSON summary (ratios and counts, no raw arrays)."""
    records = []
    for s in all_stats:
        records.append({
            "frame_idx": s.frame_idx,
            "task_id": s.task_id,
            "episode_idx": s.episode_idx,
            "primary": {
                "counts": s.primary_counts,
                "ratios": {k: round(v, 4) for k, v in s.primary_ratios().items()},
                "stable_count": s.primary_stable_count,
            },
            "wrist": {
                "counts": s.wrist_counts,
                "ratios": {k: round(v, 4) for k, v in s.wrist_ratios().items()},
                "stable_count": s.wrist_stable_count,
            },
            "llm_flops": s.llm_flops,
            "time_ms": round(s.time_ms, 2),
        })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"[analysis] Exported JSON summary → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate statistics
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_class_ratios(all_stats: List[FrameStats], camera: str = "primary") -> Dict[str, np.ndarray]:
    """
    Returns per-frame ratio arrays for each class.
    camera: "primary" | "wrist" | "both" (average of the two)
    """
    def _get_ratios(s: FrameStats, cam: str) -> Dict[str, float]:
        return s.primary_ratios() if cam == "primary" else s.wrist_ratios()

    if camera == "both":
        keys = list("ABCD")
        ratios = {k: [] for k in keys}
        for s in all_stats:
            pr = s.primary_ratios()
            wr = s.wrist_ratios()
            for k in keys:
                ratios[k].append((pr[k] + wr[k]) / 2.0)
        return {k: np.array(v) for k, v in ratios.items()}
    else:
        keys = list("ABCD")
        ratios = {k: [] for k in keys}
        for s in all_stats:
            r = _get_ratios(s, camera)
            for k in keys:
                ratios[k].append(r[k])
        return {k: np.array(v) for k, v in ratios.items()}


def collect_attn_by_class(all_stats: List[FrameStats], camera: str = "primary") -> Dict[str, np.ndarray]:
    """
    Returns attention scores grouped by class, pooled across all frames.
    Answers: "How low are Class B attention scores compared to other classes?"
    """
    out = {k: [] for k in "ABCD"}
    for s in all_stats:
        if camera == "primary":
            attn = s.primary_attn_scores
            cls_map = {"A": s.primary_class_A, "B": s.primary_class_B,
                       "C": s.primary_class_C, "D": s.primary_class_D}
        else:
            attn = s.wrist_attn_scores
            cls_map = {"A": s.wrist_class_A, "B": s.wrist_class_B,
                       "C": s.wrist_class_C, "D": s.wrist_class_D}
        for k, ids in cls_map.items():
            if len(ids) > 0:
                out[k].extend(attn[ids].tolist())
    return {k: np.array(v) for k, v in out.items()}


def collect_sim_by_class(all_stats: List[FrameStats], camera: str = "primary") -> Dict[str, np.ndarray]:
    """
    Returns cosine similarity scores grouped by class, pooled across all frames.
    Answers: "Do Class A/C have higher sim than Class B/D?"
    """
    out = {k: [] for k in "ABCD"}
    for s in all_stats:
        if camera == "primary":
            sim = s.primary_sim_scores
            cls_map = {"A": s.primary_class_A, "B": s.primary_class_B,
                       "C": s.primary_class_C, "D": s.primary_class_D}
        else:
            sim = s.wrist_sim_scores
            cls_map = {"A": s.wrist_class_A, "B": s.wrist_class_B,
                       "C": s.wrist_class_C, "D": s.wrist_class_D}
        for k, ids in cls_map.items():
            if len(ids) > 0:
                out[k].extend(sim[ids].tolist())
    return {k: np.array(v) for k, v in out.items()}


def compute_class_b_stability(all_stats: List[FrameStats], camera: str = "primary") -> np.ndarray:
    """
    For each of the 256 patch positions, compute the fraction of frames in which
    it was classified as Class B (prune candidate).
    High values → consistently dynamic+irrelevant → safe to prune.
    Low values  → only occasionally Class B → pruning might occasionally hurt.
    """
    counts = np.zeros(256, dtype=float)
    n = len(all_stats)
    for s in all_stats:
        ids = s.primary_class_B if camera == "primary" else s.wrist_class_B
        for pid in ids:
            counts[pid] += 1.0
    return counts / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

CLASS_COLORS = {"A": "#2ECC71", "B": "#E74C3C", "C": "#3498DB", "D": "#F39C12"}
CLASS_LABELS = {
    "A": "A: Cache (high-sim, low-attn)",
    "B": "B: Prune candidate (low-sim, low-attn)",
    "C": "C: Recompute (high-sim, high-attn)",
    "D": "D: Recompute (low-sim, high-attn)",
}


def plot_class_ratios_bar(all_stats: List[FrameStats], out_path: str) -> None:
    """
    Bar chart: mean class ratios for primary and wrist cameras.
    Key metric: how large is Class B?
    """
    classes = list("ABCD")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, cam in zip(axes, ["primary", "wrist"]):
        ratios = aggregate_class_ratios(all_stats, camera=cam)
        means  = [ratios[k].mean() * 100 for k in classes]
        stds   = [ratios[k].std()  * 100 for k in classes]
        colors = [CLASS_COLORS[k] for k in classes]
        bars = ax.bar(classes, means, yerr=stds, color=colors, capsize=5, alpha=0.85)
        ax.set_title(f"{cam.capitalize()} camera", fontsize=13)
        ax.set_ylabel("Fraction of 256 patches (%)")
        ax.set_ylim(0, 100)
        ax.set_xlabel("Token Class")
        for bar, m, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, m + std + 1,
                    f"{m:.1f}%", ha="center", va="bottom", fontsize=9)

    axes[0].legend(
        handles=[plt.Rectangle((0, 0), 1, 1, fc=CLASS_COLORS[k]) for k in classes],
        labels=[CLASS_LABELS[k] for k in classes],
        loc="upper right", fontsize=8,
    )
    fig.suptitle("Mean Token Class Distribution (averaged over all frames)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analysis] → {out_path}")


def plot_attention_distributions(all_stats: List[FrameStats], out_path: str) -> None:
    """
    Histograms of text-to-vision attention scores per class, for both cameras.
    Key metric: is Class B attention score truly low (< 0.01)?
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=False)
    bins = np.linspace(0, 0.04, 60)  # zoom in on low end

    for row, cam in enumerate(["primary", "wrist"]):
        attn_by_class = collect_attn_by_class(all_stats, camera=cam)
        for col, k in enumerate("ABCD"):
            ax = axes[row][col]
            data = attn_by_class[k]
            if len(data) > 0:
                ax.hist(data, bins=bins, color=CLASS_COLORS[k], alpha=0.8, edgecolor="white")
                ax.axvline(np.median(data), color="black", linestyle="--", linewidth=1.2,
                           label=f"median={np.median(data):.4f}")
                ax.axvline(np.mean(data), color="gray", linestyle=":", linewidth=1.2,
                           label=f"mean={np.mean(data):.4f}")
                ax.legend(fontsize=7)
            ax.set_title(f"{cam} – Class {k}\n(n={len(data):,})", fontsize=9)
            ax.set_xlabel("Text-to-vision attention score")
            ax.set_ylabel("Count" if col == 0 else "")

    fig.suptitle("Attention Score Distributions by Token Class", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analysis] → {out_path}")


def plot_similarity_distributions(all_stats: List[FrameStats], out_path: str) -> None:
    """
    Histograms of pixel cosine similarity per class, for both cameras.
    Validates the high-sim / low-sim separation assumption.
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=False)
    bins = np.linspace(0, 1, 50)

    for row, cam in enumerate(["primary", "wrist"]):
        sim_by_class = collect_sim_by_class(all_stats, camera=cam)
        for col, k in enumerate("ABCD"):
            ax = axes[row][col]
            data = sim_by_class[k]
            if len(data) > 0:
                ax.hist(data, bins=bins, color=CLASS_COLORS[k], alpha=0.8, edgecolor="white")
                ax.axvline(np.median(data), color="black", linestyle="--", linewidth=1.2,
                           label=f"median={np.median(data):.3f}")
                ax.legend(fontsize=7)
            ax.set_title(f"{cam} – Class {k}\n(n={len(data):,})", fontsize=9)
            ax.set_xlabel("Pixel cosine similarity")
            ax.set_ylabel("Count" if col == 0 else "")

    fig.suptitle("Pixel Cosine Similarity Distributions by Token Class", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analysis] → {out_path}")


def plot_temporal_evolution(all_stats: List[FrameStats], out_path: str, max_frames: int = 200) -> None:
    """
    Line chart: class ratios over model-query steps within an episode.
    Shows how VLA-Cache's token distribution evolves during a manipulation.
    """
    stats_to_plot = all_stats[:max_frames]
    frame_indices = [s.frame_idx for s in stats_to_plot]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    for ax, cam in zip(axes, ["primary", "wrist"]):
        ratios = aggregate_class_ratios(stats_to_plot, camera=cam)
        for k in "ABCD":
            ax.plot(frame_indices, ratios[k] * 100, label=CLASS_LABELS[k],
                    color=CLASS_COLORS[k], linewidth=1.5, alpha=0.85)
        ax.set_title(f"{cam.capitalize()} camera", fontsize=12)
        ax.set_xlabel("Model-query step (within episode)")
        ax.set_ylabel("Fraction of patches (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=7)

    fig.suptitle("Token Class Ratios over Time", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analysis] → {out_path}")


def plot_class_b_stability_heatmap(all_stats: List[FrameStats], out_path: str) -> None:
    """
    16×16 heatmap showing the fraction of frames each patch was Class B.
    Bright patches = consistently prune-candidate → safe to prune.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cam in zip(axes, ["primary", "wrist"]):
        stability = compute_class_b_stability(all_stats, camera=cam)
        im = ax.imshow(stability.reshape(16, 16), vmin=0, vmax=1,
                       cmap="RdYlGn_r", interpolation="nearest")
        plt.colorbar(im, ax=ax, label="Class B fraction")
        ax.set_title(f"{cam.capitalize()} camera\nClass B consistency across {len(all_stats)} frames",
                     fontsize=11)
        ax.set_xlabel("Patch column")
        ax.set_ylabel("Patch row")
        # Annotate mean
        mean_val = stability.mean()
        ax.text(0.5, -0.12, f"Mean Class B fraction: {mean_val:.3f}",
                transform=ax.transAxes, ha="center", fontsize=10)

    fig.suptitle("Spatial Stability of Class B (Prune Candidate) Tokens", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analysis] → {out_path}")


def plot_per_layer_reuse(all_stats: List[FrameStats], out_path: str) -> None:
    """
    Visualises the layer-adaptive reuse schedule (proportion_attn_var).
    Shows mean ± std across all frames, with pruning_loc highlighted.
    """
    schedules = [s.proportion_attn_var for s in all_stats if s.proportion_attn_var is not None]
    if not schedules:
        print("[analysis] No proportion_attn_var data, skipping per-layer plot.")
        return

    arr = np.stack(schedules)          # (N_frames, 31)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    layers = np.arange(len(mean))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(layers, mean, "b-o", markersize=4, label="Mean reuse ratio")
    ax.fill_between(layers, mean - std, mean + std, alpha=0.25, color="blue", label="±1 std")

    pruning_locs = [2, 6, 9, 11]
    for loc in pruning_locs:
        if loc < len(mean):
            ax.axvline(loc, color="red", linestyle="--", alpha=0.6,
                       label=f"pruning_loc" if loc == pruning_locs[0] else "")

    ax.set_xlabel("LLaMA Layer Index")
    ax.set_ylabel("KV Reuse Proportion (α)")
    ax.set_title("Layer-Adaptive KV Reuse Schedule (proportion_attn_var)")
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analysis] → {out_path}")


def plot_flops_comparison(all_stats: List[FrameStats], baseline_flops: Optional[float], out_path: str) -> None:
    """
    Bar chart comparing actual LLM FLOPs:
      - VLA-Cache measured FLOPs
      - Estimated FLOPs if we also prune Class B
      - Baseline FLOPs (passed in as argument, or estimated from full sequence)
    """
    if not all_stats:
        return

    cache_flops = np.array([s.llm_flops for s in all_stats if s.llm_flops > 0])
    if len(cache_flops) == 0:
        print("[analysis] No FLOPs data, skipping FLOPs plot.")
        return

    mean_cache_flops = cache_flops.mean()

    # Estimate Prune-saved FLOPs: each pruned token saves FLOPs across all 32 layers
    # FLOPs per layer ≈ 4n·d² + 2n²·d + 3n·d·m  (from modeling_llama.py)
    # Pruning reduces n by |class_B_primary + class_B_wrist|
    # Approximate relative saving: Δn/n
    prune_ratios = []
    for s in all_stats:
        if s.llm_flops > 0:
            n_pruned = len(s.primary_class_B) + len(s.wrist_class_B)
            n_total  = 512  # visual tokens
            prune_ratios.append(n_pruned / n_total)
    mean_prune_ratio = np.mean(prune_ratios) if prune_ratios else 0.0
    # Very rough: FLOPs scale ~linearly for attention+MLP combined at moderate n
    est_prune_flops = mean_cache_flops * (1 - mean_prune_ratio * 0.7)  # conservative

    labels = ["VLA-Cache", "VLA-Cache + Prune\n(estimated)"]
    values = [mean_cache_flops, est_prune_flops]
    colors = ["#3498DB", "#E74C3C"]

    if baseline_flops is not None:
        labels = ["Baseline"] + labels
        values = [baseline_flops] + values
        colors = ["#95A5A6"] + colors

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, [v / 1e12 for v in values], color=colors, alpha=0.85)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v/1e12:.2f}T", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("LLM FLOPs (TFLOPs per inference step)")
    ax.set_title(f"LLM Computation Comparison\nEst. Class B prune ratio: {mean_prune_ratio*100:.1f}%")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analysis] → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Master report generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_full_report(all_stats: List[FrameStats], out_dir: str,
                         baseline_flops: Optional[float] = None) -> None:
    """
    Run all visualizations and print a text summary.
    Call this after collecting analysis data from an evaluation run.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("  TOKEN CLASSIFICATION ANALYSIS REPORT")
    print("="*70)
    print(f"  Total frames analysed: {len(all_stats)}")

    # ── Text summary ──────────────────────────────────────────────────────────
    for cam in ["primary", "wrist"]:
        ratios = aggregate_class_ratios(all_stats, camera=cam)
        print(f"\n  [{cam.upper()}]")
        for k in "ABCD":
            m = ratios[k].mean() * 100
            s = ratios[k].std()  * 100
            print(f"    Class {k}: {m:5.1f}% ± {s:.1f}%  {CLASS_LABELS[k]}")

    # Class B attention score summary
    print("\n  [CLASS B ATTENTION SCORES]")
    for cam in ["primary", "wrist"]:
        attn = collect_attn_by_class(all_stats, camera=cam)
        b = attn["B"]
        if len(b) > 0:
            print(f"    {cam}: median={np.median(b):.5f}  mean={np.mean(b):.5f}  "
                  f"p95={np.percentile(b,95):.5f}  max={b.max():.5f}")

    print("\n  [SCHEME A FEASIBILITY]")
    for cam in ["primary", "wrist"]:
        ratios = aggregate_class_ratios(all_stats, camera=cam)
        b_mean = ratios["B"].mean() * 100
        if b_mean > 15:
            verdict = "VIABLE  (Class B > 15%)"
        elif b_mean > 5:
            verdict = "MARGINAL (Class B 5-15%, worth investigating)"
        else:
            verdict = "QUESTIONABLE (Class B < 5%, gain very limited)"
        print(f"    {cam}: Class B = {b_mean:.1f}%  →  {verdict}")

    print("="*70 + "\n")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_class_ratios_bar       (all_stats, str(out_dir / "01_class_ratios.png"))
    plot_attention_distributions(all_stats, str(out_dir / "02_attention_distributions.png"))
    plot_similarity_distributions(all_stats, str(out_dir / "03_similarity_distributions.png"))
    plot_temporal_evolution     (all_stats, str(out_dir / "04_temporal_evolution.png"))
    plot_class_b_stability_heatmap(all_stats, str(out_dir / "05_class_b_stability.png"))
    plot_per_layer_reuse        (all_stats, str(out_dir / "06_per_layer_reuse.png"))
    plot_flops_comparison       (all_stats, baseline_flops, str(out_dir / "07_flops_comparison.png"))

    export_summary_json(all_stats, str(out_dir / "summary.json"))
    save_analysis_data (all_stats, str(out_dir / "frame_stats.pkl"))

    print(f"[analysis] All outputs saved to: {out_dir}")
