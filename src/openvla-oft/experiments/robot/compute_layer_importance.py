"""
compute_layer_importance.py

Computes per-layer importance scores for the Llama2-7B decoder in OpenVLA-OFT.

Methodology (EfficientVLA §3.1):
    I^(ℓ) = 1 - cosine_sim(layer_input, layer_output)
    averaged over all token positions and all calibration frames.

    Low I^(ℓ)  →  layer barely transforms the representation  →  redundant  →  safe to skip.
    High I^(ℓ) →  layer performs critical computation         →  important  →  keep.

Usage:
    cd /home/jzzz/vla-ws/vla-cache/src/openvla-oft/
    PYTHONPATH=/home/jzzz/vla-ws/LIBERO CUDA_VISIBLE_DEVICES=0 \\
    /home/jzzz/miniconda3/envs/openvla-oft/bin/python \\
        experiments/robot/compute_layer_importance.py \\
        --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \\
        --task_suite_name libero_spatial \\
        --num_calibration_episodes 5 \\
        --max_queries_per_episode 40 \\
        --load_in_4bit True

Outputs:
    experiments/analysis/layer_importance/
        layer_importance.json   - {layer_idx: importance_score}
        layer_importance_ranked.txt - sorted layer ranking (most redundant first)
        layer_importance.png    - bar chart with colour-coded tiers
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import draccus
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from libero.libero import benchmark

sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla_action,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    get_model,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT  = "libero_object"
    LIBERO_GOAL    = "libero_goal"
    LIBERO_10      = "libero_10"


TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT:  280,
    TaskSuite.LIBERO_GOAL:    300,
    TaskSuite.LIBERO_10:      520,
}


@dataclass
class CalibConfig:
    # fmt: off
    model_family: str                   = "openvla"
    pretrained_checkpoint: Union[str, Path] = "checkpoints/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
    use_l1_regression: bool             = True
    use_diffusion: bool                 = False
    num_diffusion_steps_train: int      = 50
    num_diffusion_steps_inference: int  = 50
    use_film: bool                      = False
    num_images_in_input: int            = 2
    use_proprio: bool                   = True
    center_crop: bool                   = True
    num_open_loop_steps: int            = 8
    lora_rank: int                      = 32
    unnorm_key: Union[str, Path]        = ""
    load_in_8bit: bool                  = False
    load_in_4bit: bool                  = True

    # Disable all cache / pruning for clean calibration
    use_vla_cache: bool                 = False
    use_prune: bool                     = False
    use_preprune: bool                  = False
    use_preprune_v3: bool               = False
    use_adaptive: bool                  = False
    preprune_k_local: int               = 80
    skip_layers: str                    = ""

    # LIBERO
    task_suite_name: str                = TaskSuite.LIBERO_SPATIAL
    num_steps_wait: int                 = 10
    num_calibration_episodes: int       = 5      # number of episodes across which to collect hidden states
    max_queries_per_episode: int        = 40     # cap forward passes per episode (speed)
    env_img_res: int                    = 256

    # Output
    output_dir: str                     = "experiments/analysis/layer_importance"

    seed: int                           = 7
    run_id_note: Optional[str]          = None
    # fmt: on


# ---------------------------------------------------------------------------
# Hook infrastructure
# ---------------------------------------------------------------------------

class LayerImportanceCollector:
    """
    Registers forward hooks on every LlamaDecoderLayer.
    For each call: captures (input_hidden_states, output_hidden_states),
    computes mean cosine similarity over all token positions, accumulates.
    """

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self._cos_sum  = [0.0] * n_layers   # sum of per-step mean cosine sim
        self._call_cnt = [0]   * n_layers   # number of forward-pass calls
        self._handles  = []

    def register(self, llama_model):
        """llama_model = vla.language_model.model  (LlamaModel)"""
        for idx, layer in enumerate(llama_model.layers):
            h = layer.register_forward_hook(self._make_hook(idx))
            self._handles.append(h)
        logger.info(f"Registered importance hooks on {len(llama_model.layers)} decoder layers.")

    def _make_hook(self, layer_idx: int):
        def hook(module, inp, out):
            # inp[0]: hidden_states before this layer  [B, T, D]
            # out[0]: hidden_states after  this layer  [B, T, D]
            h_in  = inp[0].float().detach()
            h_out = out[0].float().detach()
            # cosine similarity per (batch, token) pair, then mean
            cos = F.cosine_similarity(h_in, h_out, dim=-1)   # [B, T]
            self._cos_sum[layer_idx]  += cos.mean().item()
            self._call_cnt[layer_idx] += 1
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def importance_scores(self) -> List[float]:
        """Returns I^(ℓ) = 1 - mean_cosine_sim for each layer."""
        return [
            1.0 - (self._cos_sum[i] / max(self._call_cnt[i], 1))
            for i in range(self.n_layers)
        ]

    def call_counts(self) -> List[int]:
        return list(self._call_cnt)


# ---------------------------------------------------------------------------
# Helper: model setup
# ---------------------------------------------------------------------------

def check_unnorm_key(cfg, model):
    key = cfg.task_suite_name
    if key not in model.norm_stats and f"{key}_no_noops" in model.norm_stats:
        key = f"{key}_no_noops"
    assert key in model.norm_stats, f"unnorm_key {key} not found in norm_stats"
    cfg.unnorm_key = key


# ---------------------------------------------------------------------------
# Calibration loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_calibration(cfg: CalibConfig):
    set_seed_everywhere(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    logger.info("Loading model...")
    model       = get_model(cfg)
    processor   = get_processor(cfg) if cfg.model_family == "openvla" else None
    check_unnorm_key(cfg, model)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if cfg.use_proprio else None
    action_head = get_action_head(cfg, model.llm_dim) if (cfg.use_l1_regression or cfg.use_diffusion) else None

    # ── Register hooks ──────────────────────────────────────────────────────
    llama_model = model.language_model.model          # LlamaModel (contains .layers)
    n_layers    = len(llama_model.layers)
    logger.info(f"Decoder has {n_layers} layers.")
    collector   = LayerImportanceCollector(n_layers)
    collector.register(llama_model)

    # ── LIBERO env setup ────────────────────────────────────────────────────
    resize_size = get_image_resize_size(cfg)
    task_suite  = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    n_tasks     = task_suite.n_tasks

    episode_count = 0
    task_id       = 0

    with tqdm.tqdm(total=cfg.num_calibration_episodes, desc="Calibration episodes") as pbar:
        while episode_count < cfg.num_calibration_episodes:
            task_id_use = task_id % n_tasks
            task        = task_suite.get_task(task_id_use)
            init_states = task_suite.get_task_init_states(task_id_use)

            env, task_desc = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
            env.reset()
            obs = env.set_init_state(init_states[0])

            action_queue   = deque(maxlen=cfg.num_open_loop_steps)
            prev_img       = None
            prev_img_wrist = None
            last_caches    = None
            query_idx      = 0
            t              = 0
            max_steps      = TASK_MAX_STEPS[cfg.task_suite_name]

            try:
                while t < max_steps + cfg.num_steps_wait:
                    if t < cfg.num_steps_wait:
                        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    img       = resize_image_for_policy(get_libero_image(obs), resize_size)
                    wrist_img = resize_image_for_policy(get_libero_wrist_image(obs), resize_size)
                    if prev_img is None:
                        prev_img       = img
                        prev_img_wrist = wrist_img

                    observation = {
                        "full_image":  img,
                        "wrist_image": wrist_img,
                        "prev_images": [prev_img, prev_img_wrist],
                        "state": np.concatenate((
                            obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )),
                    }

                    if len(action_queue) == 0:
                        if query_idx >= cfg.max_queries_per_episode:
                            break   # cap reached — stop this episode

                        actions, last_caches, _, _ = get_vla_action(
                            cfg, model, processor, observation, task_desc,
                            action_head=action_head,
                            proprio_projector=proprio_projector,
                            last_caches=last_caches,
                        )
                        for a in actions:
                            action_queue.append(a)
                        query_idx += 1

                    action   = action_queue.popleft()
                    obs, _, done, _ = env.step(action.tolist())
                    prev_img       = img
                    prev_img_wrist = wrist_img
                    t += 1

                    if done:
                        break

            except Exception as e:
                logger.warning(f"Episode error (task {task_id_use}): {e}")

            env.close()
            episode_count += 1
            task_id       += 1
            pbar.update(1)
            logger.info(
                f"Episode {episode_count}/{cfg.num_calibration_episodes} done "
                f"({query_idx} model queries, task={task_desc[:40]})"
            )

    collector.remove()

    # ── Compute & report ────────────────────────────────────────────────────
    scores = collector.importance_scores()
    counts = collector.call_counts()

    logger.info("\n=== Layer Importance Scores (I^(ℓ) = 1 - cos_sim) ===")
    logger.info(f"{'Layer':>6}  {'I^(l)':>8}  {'calls':>7}")
    for i, (s, c) in enumerate(zip(scores, counts)):
        logger.info(f"{i:>6}  {s:>8.5f}  {c:>7}")

    # Ranked (most redundant first)
    ranked = sorted(enumerate(scores), key=lambda x: x[1])
    logger.info("\n=== Ranking: most redundant → most important ===")
    for rank, (layer_idx, score) in enumerate(ranked):
        logger.info(f"  rank {rank+1:2d}  layer {layer_idx:2d}  I={score:.5f}")

    # ── Save JSON ────────────────────────────────────────────────────────────
    result = {str(i): float(s) for i, s in enumerate(scores)}
    json_path = os.path.join(cfg.output_dir, "layer_importance.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved: {json_path}")

    # ── Save ranked TXT ─────────────────────────────────────────────────────
    txt_path = os.path.join(cfg.output_dir, "layer_importance_ranked.txt")
    with open(txt_path, "w") as f:
        f.write("rank  layer  importance  (most redundant first)\n")
        f.write("-" * 45 + "\n")
        for rank, (layer_idx, score) in enumerate(ranked):
            f.write(f"{rank+1:4d}  {layer_idx:5d}  {score:.6f}\n")
    logger.info(f"Saved: {txt_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    _plot_importance(scores, cfg.output_dir)

    return scores


def _plot_importance(scores: List[float], output_dir: str):
    n = len(scores)
    x = list(range(n))

    # colour tiers: bottom 25% = green (safe to skip), top 25% = red (keep)
    threshold_low  = np.percentile(scores, 25)
    threshold_high = np.percentile(scores, 75)
    colors = [
        "#2ecc71" if s <= threshold_low else
        "#e74c3c" if s >= threshold_high else
        "#3498db"
        for s in scores
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x, scores, color=colors, edgecolor="white", linewidth=0.4)
    ax.axhline(threshold_low,  color="#2ecc71", linestyle="--", linewidth=1, label=f"25th pct (low={threshold_low:.4f})")
    ax.axhline(threshold_high, color="#e74c3c", linestyle="--", linewidth=1, label=f"75th pct (high={threshold_high:.4f})")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Importance  I⁽ˡ⁾ = 1 − cos_sim(input, output)")
    ax.set_title("Per-layer Importance Scores — OpenVLA-OFT (Llama2-7B decoder, libero_spatial)")
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    plt.tight_layout()
    png_path = os.path.join(output_dir, "layer_importance.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {png_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@draccus.wrap()
def main(cfg: CalibConfig):
    logger.info("=== Layer Importance Calibration ===")
    logger.info(f"  episodes : {cfg.num_calibration_episodes}")
    logger.info(f"  max_queries_per_episode : {cfg.max_queries_per_episode}")
    logger.info(f"  task suite : {cfg.task_suite_name}")
    logger.info(f"  output dir : {cfg.output_dir}")
    run_calibration(cfg)


if __name__ == "__main__":
    main()
