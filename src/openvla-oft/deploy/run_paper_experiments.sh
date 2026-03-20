#!/usr/bin/env bash
# =============================================================================
# run_paper_experiments.sh  —  论文完整实验批次（BF16，RTX 4090）
#
# 实验清单（来自 experiments/论文写作框架_VLA推理三维冗余分析.md）:
#
# ── Phase 1: 主实验（3个配置 × 4 suite × 20-trial ≈ 24h）───────────────────
#   E0  Baseline              × 4 suites
#   E1  VLA-Cache only        × 4 suites
#   E5f VLA-Cache+Skip{3-6}  × 4 suites
#
# ── Phase 2: Extended Cache 实验（2个配置 × Spatial × 20-trial ≈ 4h）────────
#   E3  Extended Cache / v3 only              × Spatial
#   E_full VLA-Cache + Extended + Skip{3-6}  × Spatial（新实验）
#
# ── Phase 3: 消融实验（Spatial only，3-trial ≈ 3h）──────────────────────────
#   E5_skip_only  Layer Skip alone (无VLA-Cache)   × Spatial
#   E5a           Skip{16-19} 对照组               × Spatial
#   E5b           Skip{24-27} 对照组               × Spatial
#   E5h           Skip{3,5,23,28} spread 对照组   × Spatial
#   E5i           VLA-Cache + Skip{3,4} 2-layer   × Spatial
#
# ── Phase 4: Token Deletion 机制验证（可选，Spatial 3-trial ≈ 2h）──────────
#   Ev1  zero-KV (v1)         × Spatial
#   Ev2b B-only block (v2b)   × Spatial
#   Ev3  stale-KV (v3/E3)     × Spatial（E3已跑，可复用结果）
#
# 使用方式：
#   export MUJOCO_GL=egl
#   export PYTHONPATH=/root/sj-tmp/workspace/LIBERO:$PYTHONPATH
#   cd /root/sj-tmp/workspace/vla-cache/src/openvla-oft
#   nohup bash deploy/run_paper_experiments.sh > experiments/results/run_paper.log 2>&1 &
#
# 选择性运行某个 Phase：
#   bash deploy/run_paper_experiments.sh --phase 1
#   bash deploy/run_paper_experiments.sh --phase 2
#   bash deploy/run_paper_experiments.sh --phase 3
#   bash deploy/run_paper_experiments.sh --phase 4
#   bash deploy/run_paper_experiments.sh --phase 1,2   （逗号分隔多Phase）
# =============================================================================
set -e

# ── 路径配置（服务器硬编码）──────────────────────────────────────────────────
PYTHON="/root/sj-tmp/conda-envs/openvla-oft/bin/python"
CHECKPOINT="checkpoints/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
LIBERO_PATH="/root/sj-tmp/workspace/LIBERO"
RESULTS_DIR="experiments/results/paper_bf16"
WORK_DIR="/root/sj-tmp/workspace/vla-cache/src/openvla-oft"

# ── 命令行参数解析 ────────────────────────────────────────────────────────────
RUN_PHASES="1,2,3"   # 默认运行 Phase 1+2+3；Phase 4 (token deletion) 可选
for arg in "$@"; do
    case "${arg}" in
        --phase) PHASE_ARG_NEXT=1 ;;
        *) [[ -n "${PHASE_ARG_NEXT}" ]] && RUN_PHASES="${arg}" && PHASE_ARG_NEXT="" ;;
    esac
done

should_run_phase() {
    local phase="$1"
    echo "${RUN_PHASES}" | grep -qE "(^|,)${phase}(,|$)"
}

# ── 环境检查 ──────────────────────────────────────────────────────────────────
cd "${WORK_DIR}"
export PYTHONPATH="${LIBERO_PATH}:${PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "============================================================"
echo "  VLA-Cache Paper Experiments (BF16)"
echo "  Date:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Phases:     ${RUN_PHASES}"
echo "  Python:     ${PYTHON}"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Results:    ${RESULTS_DIR}"
echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "============================================================"
echo ""

mkdir -p "${RESULTS_DIR}"

# ── Helper 函数 ───────────────────────────────────────────────────────────────
run_experiment() {
    local exp_id="$1"
    local suite="$2"
    local trials="$3"
    local log_file="${RESULTS_DIR}/${exp_id}_${suite}.log"
    shift 3

    echo "──────────────────────────────────────────────────────────"
    echo "  [$(date '+%H:%M:%S')]  START: ${exp_id}  suite=${suite}  trials=${trials}"
    echo "  Log: ${log_file}"
    echo "──────────────────────────────────────────────────────────"

    PYTHONPATH="${LIBERO_PATH}:${PYTHONPATH}" \
    MUJOCO_GL="${MUJOCO_GL}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    "${PYTHON}" experiments/robot/libero/run_libero_eval.py \
        --pretrained_checkpoint "${CHECKPOINT}" \
        --task_suite_name "${suite}" \
        --num_trials_per_task "${trials}" \
        --load_in_4bit False \
        --load_in_8bit False \
        --run_id_note "${exp_id}" \
        "$@" \
        2>&1 | tee "${log_file}"

    # 提取 SR
    local succ total
    succ=$(grep -oP "Total successes[:\s]+\K[0-9]+" "${log_file}" 2>/dev/null | tail -1 || echo "?")
    total=$((trials * 10))
    echo ""
    echo "  ✓  ${exp_id}/${suite} DONE — ${succ}/${total}"
    echo ""
}

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: 主实验 (3 configs × 4 suites × 20-trial)
# ══════════════════════════════════════════════════════════════════════════════
if should_run_phase 1; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  PHASE 1: Main Experiments (3 configs × 4 suites × 20t) ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    for SUITE in libero_spatial libero_object libero_goal libero_10; do

        # E0: Baseline
        run_experiment "E0_baseline_bf16_20t" "${SUITE}" 20 \
            --use_vla_cache False \
            --use_preprune_v3 False

        # E1: VLA-Cache only
        run_experiment "E1_vlacache_bf16_20t" "${SUITE}" 20 \
            --use_vla_cache True \
            --use_preprune_v3 False

        # E5f: VLA-Cache + Layer Skip{3,4,5,6}
        run_experiment "E5f_vlacache_skip3456_bf16_20t" "${SUITE}" 20 \
            --use_vla_cache True \
            --use_preprune_v3 False \
            --skip_layers "3,4,5,6"

    done
fi

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Extended Cache 实验 (Spatial × 20-trial)
# ══════════════════════════════════════════════════════════════════════════════
if should_run_phase 2; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  PHASE 2: Extended Cache (Spatial × 20t)                ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    # E3: Extended Cache only (v3, B-class stale KV, no skip)
    run_experiment "E3_extcache_v3_bf16_20t" "libero_spatial" 20 \
        --use_preprune_v3 True \
        --use_vla_cache False

    # E_full: VLA-Cache + Extended Cache + Layer Skip{3-6}（新组合）
    run_experiment "E_full_cache_ext_skip3456_bf16_20t" "libero_spatial" 20 \
        --use_vla_cache True \
        --use_preprune_v3 True \
        --skip_layers "3,4,5,6"

fi

# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: 消融实验 (Spatial × 3-trial)
# ══════════════════════════════════════════════════════════════════════════════
if should_run_phase 3; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  PHASE 3: Ablation (Spatial × 3t)                       ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    # Layer Skip alone（无VLA-Cache，验证skip单独效果）
    run_experiment "E5_skip_only_skip3456_3t" "libero_spatial" 3 \
        --use_vla_cache False \
        --use_preprune_v3 False \
        --skip_layers "3,4,5,6"

    # 对照组：中间层 Skip{16-19}（已有INT4结果43.3%，BF16复现确认）
    run_experiment "E5a_vlacache_skip1619_3t" "libero_spatial" 3 \
        --use_vla_cache True \
        --skip_layers "16,17,18,19"

    # 对照组：深层 Skip{24-27}（已有INT4结果70%，BF16复现确认）
    run_experiment "E5b_vlacache_skip2427_3t" "libero_spatial" 3 \
        --use_vla_cache True \
        --skip_layers "24,25,26,27"

    # 对照组：Spread Skip{3,5,23,28}（已有INT4结果90%）
    run_experiment "E5h_vlacache_skip35_23_28_3t" "libero_spatial" 3 \
        --use_vla_cache True \
        --skip_layers "3,5,23,28"

    # 2-layer skip（更保守）：VLA-Cache + Skip{3,4}
    run_experiment "E5i_vlacache_skip34_3t" "libero_spatial" 3 \
        --use_vla_cache True \
        --skip_layers "3,4"

fi

# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: Token Deletion 机制验证（可选，--phase 4 才运行）
# ══════════════════════════════════════════════════════════════════════════════
if should_run_phase 4; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  PHASE 4: Token Deletion Verification (Spatial × 3t)    ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    # v1: zero KV（预期 BF16 下同样 ~5% SR）
    run_experiment "Ev1_zero_kv_3t" "libero_spatial" 3 \
        --use_vla_cache True \
        --use_preprune True \
        --use_preprune_v3 False

    # v2b: B-only block（预期 ~66% SR）
    run_experiment "Ev2b_bonly_block_3t" "libero_spatial" 3 \
        --use_vla_cache True \
        --use_preprune_v2b True \
        --use_preprune_v3 False

    # v3: stale KV（应与 E3 相同，~93%）
    run_experiment "Ev3_stale_kv_3t" "libero_spatial" 3 \
        --use_preprune_v3 True \
        --use_vla_cache False

fi

# ── 汇总结果 ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  ALL DONE — Summary  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "============================================================"
for log_file in "${RESULTS_DIR}"/*.log; do
    [[ -f "${log_file}" ]] || continue
    exp_name=$(basename "${log_file}" .log)
    succ=$(grep -oP "Total successes[:\s]+\K[0-9]+" "${log_file}" 2>/dev/null | tail -1 || echo "?")
    total=$(grep -oP "num_trials_per_task[:\s]+\K[0-9]+" "${log_file}" 2>/dev/null | tail -1 || echo "?")
    [[ "${total}" != "?" ]] && total=$((total * 10))
    echo "  ${exp_name}: ${succ}/${total}"
done
echo ""
echo "  Logs: ${RESULTS_DIR}/"
