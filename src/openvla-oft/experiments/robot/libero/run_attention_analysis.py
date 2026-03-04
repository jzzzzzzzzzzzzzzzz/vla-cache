"""
run_attention_analysis.py

Phase 1.1 analysis script: runs VLA-Cache inference while collecting per-frame
token classification statistics to validate the three-way (Cache/Prune/Recompute)
framework before implementing the Prune class.

Usage (from vla-cache/src/openvla-oft/):
  CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_attention_analysis.py \
      --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-spatial \
      --task_suite_name libero_spatial \
      --num_trials_per_task 1 \
      --load_in_4bit True \
      --analysis_output_dir experiments/analysis/libero_spatial

Outputs (in analysis_output_dir/):
  frame_stats.pkl          - raw FrameStats objects (all frames)
  summary.json             - human-readable per-frame summary
  01_class_ratios.png      - mean A/B/C/D distribution bar chart
  02_attention_distributions.png  - attention score histograms per class
  03_similarity_distributions.png - cosine similarity histograms per class
  04_temporal_evolution.png       - class ratios over model-query steps
  05_class_b_stability.png        - 16x16 spatial stability heatmap
  06_per_layer_reuse.png          - layer-adaptive KV reuse schedule
  07_flops_comparison.png         - FLOPs: VLA-Cache vs VLA-Cache+Prune

NOTE: This script requires --use_vla_cache True (analysis is only meaningful
      when VLA-Cache is active; the first frame of each episode is skipped).
"""

import json
import logging
import os
import sys
import traceback
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import draccus
import numpy as np
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
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.robot.analysis_utils import (
    FrameStats,
    generate_full_report,
    save_analysis_data,
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
class AnalysisConfig:
    # fmt: off
    # ── Model ─────────────────────────────────────────────────────────────────
    model_family: str            = "openvla"
    pretrained_checkpoint: Union[str, Path] = "checkpoints/openvla-7b-oft-finetuned-libero-spatial"
    use_l1_regression: bool      = True
    use_diffusion: bool          = False
    num_diffusion_steps_train: int   = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool               = False
    num_images_in_input: int     = 2
    use_proprio: bool            = True
    center_crop: bool            = True
    num_open_loop_steps: int     = 8
    lora_rank: int               = 32
    unnorm_key: Union[str, Path] = ""
    load_in_8bit: bool           = False
    load_in_4bit: bool           = False

    # ── VLA-Cache (must be True for analysis) ─────────────────────────────────
    use_vla_cache: bool          = True

    # ── LIBERO ────────────────────────────────────────────────────────────────
    task_suite_name: str         = TaskSuite.LIBERO_SPATIAL
    num_steps_wait: int          = 10
    num_trials_per_task: int     = 1    # keep small; 1-3 trials per task is enough
    env_img_res: int             = 256
    initial_states_path: str     = "DEFAULT"

    # ── Analysis output ───────────────────────────────────────────────────────
    analysis_output_dir: str     = "./experiments/analysis/run"

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed: int                    = 7
    run_id_note: Optional[str]   = None
    # fmt: on


def check_unnorm_key(cfg: AnalysisConfig, model) -> None:
    unnorm_key = cfg.task_suite_name
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in model.norm_stats, f"unnorm_key {unnorm_key} not in norm_stats!"
    cfg.unnorm_key = unnorm_key


def run_analysis_episode(
    cfg: AnalysisConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor,
    action_head,
    proprio_projector,
    task_id: int,
    episode_idx: int,
    initial_state=None,
) -> List[FrameStats]:
    """Run one episode, collecting FrameStats at every model-query step."""
    env.reset()
    obs = env.set_init_state(initial_state)

    action_queue   = deque(maxlen=cfg.num_open_loop_steps)
    prev_img       = None
    prev_img_wrist = None
    last_caches    = None
    episode_stats: List[FrameStats] = []
    query_idx      = 0   # model-query counter within episode (for frame_idx)
    t = 0
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    try:
        while t < max_steps + cfg.num_steps_wait:
            # Let objects settle
            if t < cfg.num_steps_wait:
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # ── Build observation ─────────────────────────────────────────────
            img       = resize_image_for_policy(get_libero_image(obs),       resize_size)
            wrist_img = resize_image_for_policy(get_libero_wrist_image(obs), resize_size)

            if prev_img is None:
                prev_img       = img
                prev_img_wrist = wrist_img

            observation = {
                "full_image":   img,
                "wrist_image":  wrist_img,
                "prev_images":  [prev_img, prev_img_wrist],
                "state": np.concatenate((
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )),
            }

            # ── Query model ───────────────────────────────────────────────────
            if len(action_queue) == 0:
                actions, last_caches, _, metrics = get_vla_action(
                    cfg, model, processor, observation, task_description,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    use_film=cfg.use_film,
                    last_caches=last_caches,
                    collect_analysis=True,
                    analysis_frame_idx=query_idx,
                    analysis_task_id=task_id,
                    analysis_episode_idx=episode_idx,
                )
                # ── Diagnostics (query_idx 0 and 1 only) ─────────────────────
                if query_idx <= 1:
                    attn = last_caches['attentions'] if last_caches is not None else None
                    if attn is None:
                        logger.warning(f"  [DIAG q={query_idx}] last_caches['attentions'] is None!")
                    else:
                        a15 = attn[15] if len(attn) > 15 else "MISSING"
                        a15_type = type(a15).__name__ if not isinstance(a15, str) else a15
                        a15_shape = tuple(a15.shape) if hasattr(a15, 'shape') else "no shape"
                        a15_dev = str(a15.device) if hasattr(a15, 'device') else "?"
                        logger.info(f"  [DIAG q={query_idx}] attentions len={len(attn)}, attn[15] type={a15_type} shape={a15_shape} device={a15_dev}")
                    frame_stat_diag = metrics.get("frame_stats")
                    logger.info(f"  [DIAG q={query_idx}] frame_stats={frame_stat_diag is not None}")
                # ─────────────────────────────────────────────────────────────
                frame_stat = metrics.get("frame_stats")
                if frame_stat is not None:          # None for query_idx==0 (no prev_attn yet)
                    episode_stats.append(frame_stat)

                query_idx += 1
                action_queue.extend(actions)

            # ── Step environment ──────────────────────────────────────────────
            prev_img       = img
            prev_img_wrist = wrist_img
            action = action_queue.popleft()
            action = normalize_gripper_action(action, binarize=True)
            if cfg.model_family == "openvla":
                action = invert_gripper_action(action)
            obs, _, done, _ = env.step(action.tolist())
            if done:
                break
            t += 1

    except Exception as e:
        logger.warning(f"Episode error at t={t}: {e}\n{traceback.format_exc()}")

    logger.info(f"  Episode done: {len(episode_stats)} frames with analysis data "
                f"(first query skipped, no prev_attn)")
    return episode_stats


@draccus.wrap()
def run_analysis(cfg: AnalysisConfig) -> None:
    assert cfg.use_vla_cache, "Analysis requires --use_vla_cache True"

    set_seed_everywhere(cfg.seed)

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info("Loading model…")
    model = get_model(cfg)
    processor = get_processor(cfg)
    check_unnorm_key(cfg, model)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)

    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    resize_size = get_image_resize_size(cfg)

    # ── Collect FLOPs baseline (first episode without VLA-Cache) ─────────────
    # We collect baseline FLOPs by temporarily disabling VLA-Cache for one episode.
    # This is used only for the FLOPs comparison plot.
    baseline_flops: Optional[float] = None
    logger.info("Collecting baseline FLOPs (one episode without VLA-Cache)…")
    try:
        cfg.use_vla_cache = False
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite_tmp  = benchmark_dict[cfg.task_suite_name]()
        task_tmp        = task_suite_tmp.get_task(0)
        env_tmp, task_desc_tmp = get_libero_env(task_tmp, cfg.model_family, resolution=cfg.env_img_res)

        env_tmp.reset()
        init_states_tmp = task_suite_tmp.get_task_init_states(0)
        obs_tmp = env_tmp.set_init_state(init_states_tmp[0])
        model.language_model.all_FLOPs = 0
        img_tmp  = resize_image_for_policy(get_libero_image(obs_tmp),       resize_size)
        wrist_tmp = resize_image_for_policy(get_libero_wrist_image(obs_tmp), resize_size)
        obs_dict_tmp = {
            "full_image":  img_tmp,
            "wrist_image": wrist_tmp,
            "prev_images": [img_tmp, wrist_tmp],
            "state": np.concatenate((obs_tmp["robot0_eef_pos"],
                                     quat2axisangle(obs_tmp["robot0_eef_quat"]),
                                     obs_tmp["robot0_gripper_qpos"])),
        }
        get_vla_action(cfg, model, processor, obs_dict_tmp, task_desc_tmp,
                       action_head=action_head, proprio_projector=proprio_projector,
                       collect_analysis=True)
        baseline_flops = float(model.language_model.all_FLOPs)
        logger.info(f"  Baseline FLOPs: {baseline_flops/1e12:.3f} TFLOPs")
        cfg.use_vla_cache = True
    except Exception as e:
        logger.warning(f"Could not collect baseline FLOPs: {e}")
        cfg.use_vla_cache = True

    # ── Main evaluation + analysis ────────────────────────────────────────────
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite     = benchmark_dict[cfg.task_suite_name]()
    num_tasks      = task_suite.n_tasks

    all_stats: List[FrameStats] = []

    for task_id in tqdm.tqdm(range(num_tasks), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

        for episode_idx in range(cfg.num_trials_per_task):
            logger.info(f"\n[Task {task_id}/{num_tasks-1}  Episode {episode_idx}] {task_description}")
            init_state = initial_states[episode_idx] if cfg.initial_states_path == "DEFAULT" else None

            episode_stats = run_analysis_episode(
                cfg, env, task_description, model, resize_size,
                processor, action_head, proprio_projector,
                task_id=task_id, episode_idx=episode_idx,
                initial_state=init_state,
            )
            all_stats.extend(episode_stats)

    logger.info(f"\nTotal FrameStats collected: {len(all_stats)}")

    # ── Generate report ───────────────────────────────────────────────────────
    out_dir = cfg.analysis_output_dir
    generate_full_report(all_stats, out_dir, baseline_flops=baseline_flops)


if __name__ == "__main__":
    run_analysis()
