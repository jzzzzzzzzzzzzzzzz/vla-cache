#!/usr/bin/env bash
# =============================================================================
# run_paper_experiments_v3.sh  —  论文第三批实验（BF16，RTX 4090）
#
# 前提：v1 + v2 批次已完成，服务器上已有 E5i/E3 Spatial 结果
#
# ── Phase 9: E5i 跨 Suite（最高优先级）──────────────────────────────────────
#   E5i  VLA-Cache + Skip{3,4}  × Object / Goal / LIBERO-10  × 20-trial
#   决定 E5i 能否作为论文主结果（关键：Goal suite SR 是否保持 ≥ 93%）
#
# ── Phase 10: 延迟精确测量（第二优先级）──────────────────────────────────────
#   E0 / E1 / E3 / E5i  各跑 libero_spatial × 5-trial（50 episodes）
#   从日志提取 Average CUDA latency 和 Average time per step，汇报论文 speedup
#
# ── Phase 11: E6c 验证（可选，第三优先级）────────────────────────────────────
#   E6c  ExtCache v3 + Skip{3,4}（无 VLA-Cache）× Spatial × 3-trial
#   验证空间+深度冗余可同时利用（不含 Cache 的交互问题）
#
# 使用方式（默认跑 Phase 9+10）：
#   cd /root/sj-tmp/workspace/vla-cache/src/openvla-oft
#   nohup bash deploy/run_paper_experiments_v3.sh \
#       > experiments/results/run_paper_v3.log 2>&1 &
#
# 选择性运行：
#   bash deploy/run_paper_experiments_v3.sh --phase 9        # 只跑 E5i 跨 suite
#   bash deploy/run_paper_experiments_v3.sh --phase 10       # 只跑延迟测量
#   bash deploy/run_paper_experiments_v3.sh --phase 9,10,11  # 全部
#
# 预计时长：
#   Phase 9:  ~15h（3 suite × ~5h/suite × 20-trial）
#   Phase 10: ~3h（4 config × ~45min × 5-trial）
#   Phase 11: ~1.5h（3-trial）
#   总计:     ~18-20h（建议 nohup 后台挂起）
# =============================================================================
set -e

# ── 路径配置（服务器硬编码）──────────────────────────────────────────────────
PYTHON="/root/sj-tmp/conda-envs/openvla-oft/bin/python"
CHECKPOINT="checkpoints/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
LIBERO_PATH="/root/sj-tmp/workspace/LIBERO"
RESULTS_DIR="experiments/results/paper_bf16"
WORK_DIR="/root/sj-tmp/workspace/vla-cache/src/openvla-oft"

# ── 命令行参数解析 ────────────────────────────────────────────────────────────
RUN_PHASES="9,10"
for arg in "$@"; do
    case "${arg}" in
        --phase) PHASE_ARG_NEXT=1 ;;
        *) [[ -n "${PHASE_ARG_NEXT}" ]] && RUN_PHASES="${arg}" && PHASE_ARG_NEXT="" ;;
    esac
done

should_run_phase() {
    echo "${RUN_PHASES}" | grep -qE "(^|,)${1}(,|$)"
}

# ── 环境初始化 ─────────────────────────────────────────────────────────────────
cd "${WORK_DIR}"
export PYTHONPATH="${LIBERO_PATH}:${PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "============================================================"
echo "  VLA-Cache Paper Experiments v3 (BF16)"
echo "  Date:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Phases:  ${RUN_PHASES}"
echo "  Python:  ${PYTHON}"
echo "  GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "============================================================"
echo ""

mkdir -p "${RESULTS_DIR}"

# ── Helper: 运行单个实验 ──────────────────────────────────────────────────────
run_experiment() {
    local exp_id="$1"
    local suite="$2"
    local trials="$3"
    local log_file="${RESULTS_DIR}/${exp_id}_${suite}.log"
    shift 3

    # 跳过已完成的实验（log 已存在且包含 "Total successes"）
    if [[ -f "${log_file}" ]] && grep -q "Total successes" "${log_file}" 2>/dev/null; then
        local succ
        succ=$(grep -oP "Total successes[:\s]+\K[0-9]+" "${log_file}" | tail -1)
        echo "  ⏭  SKIP (already done): ${exp_id}/${suite} — ${succ}/$((trials*10))"
        return 0
    fi

    echo "──────────────────────────────────────────────────────────"
    echo "  [$(date '+%H:%M:%S')]  START: ${exp_id}"
    echo "  Suite: ${suite}  Trials: ${trials}"
    echo "  Log:   ${log_file}"
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

    local succ
    succ=$(grep -oP "Total successes[:\s]+\K[0-9]+" "${log_file}" 2>/dev/null | tail -1 || echo "?")
    echo ""
    echo "  ✓  DONE: ${exp_id}/${suite} — ${succ}/$((trials*10))"
    echo "  [$(date '+%H:%M:%S')]"
    echo ""
}

# ── Helper: 从日志提取延迟数据 ───────────────────────────────────────────────
extract_latency() {
    local log_file="$1"
    local exp_name="$2"

    local cuda_avg step_avg reuse_ratio
    # 取日志最后 100 条 CUDA latency 读数求均值（跳过 warmup）
    cuda_avg=$(grep -oP "Average CUDA latency: \K[\d.]+" "${log_file}" 2>/dev/null \
               | tail -100 | awk '{s+=$1; n++} END {if(n>0) printf "%.1f", s/n; else print "?"}')
    step_avg=$(grep -oP "Average time per step: \K[\d.]+" "${log_file}" 2>/dev/null \
               | awk '{s+=$1; n++} END {if(n>0) printf "%.1f", s/n; else print "?"}')
    reuse_ratio=$(grep -oP "Token Reusing Ratio \(Primary\): \K[\d.]+" "${log_file}" 2>/dev/null \
                  | tail -50 | awk '{s+=$1; n++} END {if(n>0) printf "%.1f%%", s/n; else print "N/A"}')

    printf "  %-40s  CUDA: %6s ms  Step: %6s ms  Reuse: %s\n" \
        "${exp_name}" "${cuda_avg}" "${step_avg}" "${reuse_ratio}"
}

# ══════════════════════════════════════════════════════════════════════════════
# Phase 9: E5i × Object / Goal / LIBERO-10（优先级 1）
#
# 目标：验证 E5i（Cache+Skip{3,4}）在所有 4 个 suite 上的 SR。
# 关键判断条件：
#   - Goal suite SR ≥ 93%  → E5i 作为论文主结果（替代 E5f 的 80% Goal）
#   - Goal suite SR < 88%  → E5i 仅限于 Spatial/Object/LIBERO-10 结论
# ══════════════════════════════════════════════════════════════════════════════
if should_run_phase 9; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 9: E5i Cross-Suite (Cache+Skip{3,4}) × 20-trial      ║"
    echo "║  Object → Goal → LIBERO-10                                   ║"
    echo "║  预计: ~15h (3 × 5h)                                         ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    run_experiment "E5i_vlacache_skip34_bf16_20t" "libero_object" 20 \
        --use_vla_cache True \
        --skip_layers "3,4"

    run_experiment "E5i_vlacache_skip34_bf16_20t" "libero_goal" 20 \
        --use_vla_cache True \
        --skip_layers "3,4"

    run_experiment "E5i_vlacache_skip34_bf16_20t" "libero_10" 20 \
        --use_vla_cache True \
        --skip_layers "3,4"

    echo ""
    echo "── Phase 9 Summary ──────────────────────────────────────────"
    for suite in libero_object libero_goal libero_10; do
        f="${RESULTS_DIR}/E5i_vlacache_skip34_bf16_20t_${suite}.log"
        [[ -f "$f" ]] && succ=$(grep -oP "Total successes[:\s]+\K[0-9]+" "$f" | tail -1 || echo "?") \
            && echo "  E5i / ${suite}: ${succ}/200 ($(( succ*100/200 ))%)"
    done
    echo ""
fi

# ══════════════════════════════════════════════════════════════════════════════
# Phase 10: 延迟精确测量（优先级 2）
#
# 每个配置跑 libero_spatial × 5-trial（50 episodes）。
# 5-trial 足以使 "Average CUDA latency" 稳定（warmup 在前几步完成）。
# 提取指标：Average CUDA latency（LLM 纯计算）+ Average time per step（端到端）
# ══════════════════════════════════════════════════════════════════════════════
if should_run_phase 10; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 10: Latency Measurement × 5-trial                     ║"
    echo "║  E0 / E1 / E3 / E5i — libero_spatial                         ║"
    echo "║  预计: ~3h (4 config × ~45min)                               ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    # E0: Baseline（无任何优化）
    run_experiment "LAT_E0_baseline_5t" "libero_spatial" 5 \
        --use_vla_cache False \
        --use_preprune_v3 False

    # E1: VLA-Cache only（时间冗余）
    run_experiment "LAT_E1_vlacache_5t" "libero_spatial" 5 \
        --use_vla_cache True \
        --use_preprune_v3 False

    # E3: Extended Cache v3（空间冗余）
    run_experiment "LAT_E3_extcache_5t" "libero_spatial" 5 \
        --use_preprune_v3 True \
        --use_vla_cache False

    # E5i: VLA-Cache + Skip{3,4}（时间+深度冗余）
    run_experiment "LAT_E5i_skip34_5t" "libero_spatial" 5 \
        --use_vla_cache True \
        --use_preprune_v3 False \
        --skip_layers "3,4"

    echo ""
    echo "── Phase 10: Latency Report ─────────────────────────────────"
    echo ""
    for exp_id in "LAT_E0_baseline_5t" "LAT_E1_vlacache_5t" "LAT_E3_extcache_5t" "LAT_E5i_skip34_5t"; do
        f="${RESULTS_DIR}/${exp_id}_libero_spatial.log"
        [[ -f "$f" ]] && extract_latency "$f" "${exp_id}"
    done
    echo ""

    # 计算相对于 E0 的加速比
    echo "── CUDA Speedup vs E0 ───────────────────────────────────────"
    E0_CUDA=$(grep -oP "Average CUDA latency: \K[\d.]+" \
              "${RESULTS_DIR}/LAT_E0_baseline_5t_libero_spatial.log" 2>/dev/null \
              | tail -50 | awk '{s+=$1;n++} END{printf "%.1f",s/n}' || echo "0")
    for exp_id in "LAT_E1_vlacache_5t" "LAT_E3_extcache_5t" "LAT_E5i_skip34_5t"; do
        f="${RESULTS_DIR}/${exp_id}_libero_spatial.log"
        [[ -f "$f" ]] || continue
        cuda=$(grep -oP "Average CUDA latency: \K[\d.]+" "$f" | tail -50 \
               | awk '{s+=$1;n++} END{printf "%.1f",s/n}')
        speedup=$(awk "BEGIN{printf \"%.2f\", ${E0_CUDA}/${cuda}}")
        printf "  %-35s  %s ms  →  %s× speedup\n" "${exp_id}" "${cuda}" "${speedup}"
    done
    echo ""
fi

# ══════════════════════════════════════════════════════════════════════════════
# Phase 11: E6c — E3 + Skip{3,4}（无 VLA-Cache）× Spatial × 3-trial（可选）
#
# 动机：E_full_v2 (E3+Cache+Skip) 只有 84%，主要原因是 ExtCache+Cache 交互。
# E6c 去掉 Cache，只叠加空间冗余（E3）+ 深度冗余（Skip），验证两维是否可正交。
# 若 E6c SR ≥ 92% → 说明 E_full 84% 主要是 Cache 交互问题，结论更清晰。
# ══════════════════════════════════════════════════════════════════════════════
if should_run_phase 11; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 11: E6c (ExtCache+Skip{3,4}, no Cache) Spatial × 3t  ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    run_experiment "E6c_extcache_skip34_bf16_3t" "libero_spatial" 3 \
        --use_preprune_v3 True \
        --use_vla_cache False \
        --skip_layers "3,4"

fi

# ── 全局汇总 ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  ALL DONE — Final Summary  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "============================================================"
echo ""
echo "── Success Rates ────────────────────────────────────────────"
for log_file in "${RESULTS_DIR}"/*.log; do
    [[ -f "${log_file}" ]] || continue
    exp_name=$(basename "${log_file}" .log)
    succ=$(grep -oP "Total successes[:\s]+\K[0-9]+" "${log_file}" 2>/dev/null | tail -1 || echo "?")
    trials=$(grep -oP "num_trials_per_task[:\s]+\K[0-9]+" "${log_file}" 2>/dev/null | tail -1 || echo "?")
    [[ "${trials}" != "?" ]] && total=$((trials * 10)) || total="?"
    echo "  ${exp_name}: ${succ}/${total}"
done
echo ""
echo "  Logs: ${RESULTS_DIR}/"
echo "============================================================"
