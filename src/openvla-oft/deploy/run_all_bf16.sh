#!/usr/bin/env bash
# =============================================================================
# run_all_bf16.sh  —  服务器端实验批量运行脚本（BF16 模式）
# 依次运行：E0 Baseline（20t）、E1 VLA-Cache（20t）、E5f Cache+Skip{3,4,5,6}（20t）
# 每个实验自动保存结果到 experiments/results/ 并打印最终 SR
# 使用方式：
#   export MUJOCO_GL=egl
#   export PYTHONPATH=/path/to/LIBERO:$PYTHONPATH
#   cd /path/to/vla-cache/src/openvla-oft
#   bash deploy/run_all_bf16.sh [CHECKPOINT_PATH]
# =============================================================================
set -e

CHECKPOINT="${1:-checkpoints/openvla-7b-oft-finetuned-libero-spatial-object-goal-10}"
PYTHON="/path/to/conda/envs/openvla-oft/bin/python"   # ← 修改为服务器上实际路径
RESULTS_DIR="experiments/results/server_bf16"
TRIALS=20

# 自动检测 Python 路径
if command -v conda &>/dev/null; then
    CONDA_BASE=$(conda info --base)
    PYTHON="${CONDA_BASE}/envs/openvla-oft/bin/python"
fi

mkdir -p "${RESULTS_DIR}"

echo "============================================================"
echo "  VLA-Cache BF16 Server Experiments"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Python:     ${PYTHON}"
echo "  Trials/task: ${TRIALS}"
echo "  Results:    ${RESULTS_DIR}"
echo "============================================================"
echo ""

# ── 共用参数 ──────────────────────────────────────────────────
COMMON_ARGS=(
    --pretrained_checkpoint "${CHECKPOINT}"
    --task_suite_name libero_spatial
    --num_trials_per_task "${TRIALS}"
    --load_in_4bit False    # BF16 模式：不使用 INT4 量化
    --load_in_8bit False
)

run_experiment() {
    local exp_id="$1"
    local log_file="${RESULTS_DIR}/${exp_id}.log"
    shift
    echo "──────────────────────────────────────────────────────"
    echo "  Starting ${exp_id}  ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "  Log: ${log_file}"
    echo "──────────────────────────────────────────────────────"

    "${PYTHON}" experiments/robot/libero/run_libero_eval.py \
        "${COMMON_ARGS[@]}" "$@" \
        --run_id_note "${exp_id}" \
        2>&1 | tee "${log_file}"

    # 提取并打印最终 SR
    SR=$(grep "Overall success rate" "${log_file}" -A1 | tail -1 | grep -oP '[0-9]+\.[0-9]+%?' | head -1)
    TOTAL=$(grep "Total successes" "${log_file}" | grep -oP '[0-9]+' | head -1)
    echo ""
    echo "  ✓ ${exp_id} DONE — SR: ${SR}  (${TOTAL}/${TRIALS_TOTAL} successes)"
    echo ""
}

TRIALS_TOTAL=$((TRIALS * 10))   # 10 tasks

# ── E0: Baseline（无任何优化）────────────────────────────────
run_experiment "E0_baseline_bf16_20t" \
    --use_vla_cache False \
    --use_preprune_v3 False

# ── E1: VLA-Cache only ────────────────────────────────────────
run_experiment "E1_vlacache_bf16_20t" \
    --use_vla_cache True \
    --use_preprune_v3 False

# ── E5f: VLA-Cache + Layer Skip{3,4,5,6} Calib-Early-4 ───────
run_experiment "E5f_vlacache_skip3456_bf16_20t" \
    --use_vla_cache True \
    --use_preprune_v3 False \
    --skip_layers "3,4,5,6"

# ── 汇总 ─────────────────────────────────────────────────────
echo "============================================================"
echo "  All experiments done. Summary:"
echo "============================================================"
for exp in E0_baseline_bf16_20t E1_vlacache_bf16_20t E5f_vlacache_skip3456_bf16_20t; do
    LOG="${RESULTS_DIR}/${exp}.log"
    if [[ -f "${LOG}" ]]; then
        SUCC=$(grep "Total successes" "${LOG}" | grep -oP '[0-9]+' | head -1)
        echo "  ${exp}: ${SUCC}/${TRIALS_TOTAL} successes"
    fi
done
echo ""
echo "  Results saved to: ${RESULTS_DIR}/"
