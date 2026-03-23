#!/usr/bin/env bash
# =============================================================================
# run_paper_experiments_v2.sh  —  论文补充实验批次（BF16，RTX 4090）
#
# 前提：run_paper_experiments.sh (v1) 已完成
#       E_full bug 已修复（openvla_utils.py 三处 use_preprune_v3 fix）
#
# ── Phase 5: E_full 重跑（bug fix后）──────────────────────────────────────
#   E_full  VLA-Cache + Extended + Skip{3-6}  × Spatial × 20-trial
#
# ── Phase 6: E5i 正式 20-trial（Spatial 已有3-trial≈93%）──────────────────
#   E5i     VLA-Cache + Skip{3,4}             × Spatial × 20-trial
#
# ── Phase 7: E3 跨 Suite（已有Spatial=97%）───────────────────────────────
#   E3      Extended Cache                    × Object / Goal / LIBERO-10 × 20-trial
#
# ── Phase 8: 延迟测量（timing only，--phase 8 才运行）──────────────────
#   对每个配置跑 3-trial，记录 time_elapsed 日志
#
# 使用方式：
#   export MUJOCO_GL=egl
#   export PYTHONPATH=/root/sj-tmp/workspace/LIBERO:$PYTHONPATH
#   cd /root/sj-tmp/workspace/vla-cache/src/openvla-oft
#   nohup bash deploy/run_paper_experiments_v2.sh > experiments/results/run_paper_v2.log 2>&1 &
#
# 选择性运行某个 Phase：
#   bash deploy/run_paper_experiments_v2.sh --phase 5
#   bash deploy/run_paper_experiments_v2.sh --phase 5,6,7
# =============================================================================
set -e

# ── 路径配置（服务器硬编码）──────────────────────────────────────────────────
PYTHON="/root/sj-tmp/conda-envs/openvla-oft/bin/python"
CHECKPOINT="checkpoints/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
LIBERO_PATH="/root/sj-tmp/workspace/LIBERO"
RESULTS_DIR="experiments/results/paper_bf16"
WORK_DIR="/root/sj-tmp/workspace/vla-cache/src/openvla-oft"

# ── 命令行参数解析 ────────────────────────────────────────────────────────────
RUN_PHASES="5,6,7"   # 默认运行 Phase 5+6+7；Phase 8 (latency) 可选
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
echo "  VLA-Cache Paper Experiments v2 (BF16)"
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
# Phase 5: E_full 重跑（修复 reshape crash）
#   原因: use_preprune_v3 时 prev_attn 未切换到 last_full_attn + task_relevant_selection
#         未接收 attn_scores_override → token_attention_merge 在 compact 空间返回 <256-d
#         → get_top_attention_patches reshape(16,16) 崩溃。
#   修复: openvla_utils.py 三处 use_preprune_v3 扩展（详见 git log）
# ══════════════════════════════════════════════════════════════════════════════
if should_run_phase 5; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  PHASE 5: E_full Re-run (bug fixed) Spatial × 20t       ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    run_experiment "E_full_v2_cache_ext_skip3456_bf16_20t" "libero_spatial" 20 \
        --use_vla_cache True \
        --use_preprune_v3 True \
        --skip_layers "3,4,5,6"

fi

# ══════════════════════════════════════════════════════════════════════════════
# Phase 6: E5i 正式 20-trial（VLA-Cache + Skip{3,4}）
#   背景: 3-trial 消融显示 E5i ≈ 93.3% SR（vs E5f=87%），2-layer skip 更保守更稳定。
#   目标: 20-trial 确认 SR ≥ 93%，如达到则作为论文主结果（替换 E5f）。
# ══════════════════════════════════════════════════════════════════════════════
if should_run_phase 6; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  PHASE 6: E5i Formal 20-trial (Cache+Skip{3,4}) Spatial  ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    run_experiment "E5i_vlacache_skip34_bf16_20t" "libero_spatial" 20 \
        --use_vla_cache True \
        --use_preprune_v3 False \
        --skip_layers "3,4"

fi

# ══════════════════════════════════════════════════════════════════════════════
# Phase 7: E3 跨 Suite（Extended Cache × Object/Goal/LIBERO-10）
#   背景: E3 on Spatial = 97% SR（vs Baseline 94.5%），优于 VLA-Cache（93.5%）。
#   目标: 验证 E3 在其他 suite 的泛化性，作为论文 Table 1 E3 行的数据。
# ══════════════════════════════════════════════════════════════════════════════
if should_run_phase 7; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  PHASE 7: E3 Cross-Suite (Object/Goal/10 × 20t)          ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    for SUITE in libero_object libero_goal libero_10; do
        run_experiment "E3_extcache_v3_bf16_20t" "${SUITE}" 20 \
            --use_preprune_v3 True \
            --use_vla_cache False
    done

fi

# ══════════════════════════════════════════════════════════════════════════════
# Phase 8: 延迟测量（--phase 8 才运行）
#   每个配置跑 libero_spatial × 3-trial（~30步/episode），
#   run_libero_eval.py 已记录 time_elapsed 到日志，grep 提取均值。
#   目标: 获取 E0/E1/E3/E5i/E_full 的 ms/step 数据用于论文 Table 2。
# ══════════════════════════════════════════════════════════════════════════════
if should_run_phase 8; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  PHASE 8: Latency Measurement (Spatial × 3t)             ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    # E0: Baseline
    run_experiment "LAT_E0_baseline_3t" "libero_spatial" 3 \
        --use_vla_cache False \
        --use_preprune_v3 False

    # E1: VLA-Cache
    run_experiment "LAT_E1_vlacache_3t" "libero_spatial" 3 \
        --use_vla_cache True \
        --use_preprune_v3 False

    # E3: Extended Cache
    run_experiment "LAT_E3_extcache_3t" "libero_spatial" 3 \
        --use_preprune_v3 True \
        --use_vla_cache False

    # E5i: Cache + Skip{3,4}
    run_experiment "LAT_E5i_skip34_3t" "libero_spatial" 3 \
        --use_vla_cache True \
        --skip_layers "3,4"

    # E5f: Cache + Skip{3-6}
    run_experiment "LAT_E5f_skip3456_3t" "libero_spatial" 3 \
        --use_vla_cache True \
        --skip_layers "3,4,5,6"

    # E_full (fixed): Cache + Extended + Skip{3-6}
    run_experiment "LAT_E_full_v2_3t" "libero_spatial" 3 \
        --use_vla_cache True \
        --use_preprune_v3 True \
        --skip_layers "3,4,5,6"

    echo ""
    echo "── Latency summary (time_elapsed) ──────────────────────────"
    for log_file in "${RESULTS_DIR}"/LAT_*.log; do
        [[ -f "${log_file}" ]] || continue
        exp_name=$(basename "${log_file}" .log)
        # run_libero_eval prints "time_elapsed: X.XXX" per step
        avg_ms=$(grep -oP "time_elapsed[:\s]+\K[0-9.]+" "${log_file}" 2>/dev/null \
                 | awk '{s+=$1; n++} END {if(n>0) printf "%.1f ms", s/n*1000; else print "?"}')
        echo "  ${exp_name}: ${avg_ms}/step"
    done
    echo ""

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
