#!/usr/bin/env bash
# =============================================================================
# setup_server.sh  —  VLA-Cache + Layer Skip 服务器部署脚本（4090, BF16）
# 目标环境：Ubuntu 22.04 / CUDA 12.x / Python 3.10 / RTX 4090 (24 GB)
# 无需 bitsandbytes（BF16 不需要 INT4 量化）
# 使用方式：bash setup_server.sh [YOUR_GITHUB_REPO_URL]
# =============================================================================
set -e

REPO_URL="${1:-https://github.com/jzzzzzzzzzzzzzzzz/vla-cache.git}"
OPENVLA_DIR="openvla-oft"      # 子目录（repo 内部路径）
LIBERO_REPO="https://github.com/Lifelong-Robot-Learning/LIBERO.git"
CONDA_ENV="openvla-oft"
PYTHON_VERSION="3.10"

echo "============================================================"
echo "  VLA-Cache Server Setup  (BF16, no bitsandbytes)"
echo "  Repo: ${REPO_URL}"
echo "  Conda env: ${CONDA_ENV}"
echo "============================================================"

# ── 1. Conda 环境 ──────────────────────────────────────────────
echo "[1/7] Creating conda environment: ${CONDA_ENV}"
conda create -n "${CONDA_ENV}" python="${PYTHON_VERSION}" -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ── 2. PyTorch (cu121) ────────────────────────────────────────
echo "[2/7] Installing PyTorch 2.2.0+cu121"
pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ── 3. Clone 主仓库 ───────────────────────────────────────────
echo "[3/7] Cloning main repo"
git clone "${REPO_URL}" vla-cache
cd vla-cache/src/${OPENVLA_DIR}

# ── 4. 安装 openvla-oft（editable）────────────────────────────
# 注：pyproject.toml 中 transformers 指向自定义 fork，会自动安装
echo "[4/7] Installing openvla-oft (editable) + custom transformers fork"
pip install -e "."
# 上面会安装 transformers @ git+https://github.com/siyuhsu/transformers.git@vla-cache-openvla-oft
# 验证版本
python -c "import transformers; print('transformers:', transformers.__version__)"

# ── 5. 安装 LIBERO ────────────────────────────────────────────
echo "[5/7] Installing LIBERO"
cd ../../../
git clone "${LIBERO_REPO}" LIBERO
cd LIBERO
pip install -e "."
cd ../vla-cache/src/${OPENVLA_DIR}

# ── 6. 安装额外依赖（非 bitsandbytes）────────────────────────
echo "[6/7] Installing remaining dependencies (no bitsandbytes)"
pip install \
    mujoco==2.3.7 \
    robosuite==1.4.1 \
    opencv-python==4.9.0.80 \
    wandb==0.25.0 \
    imageio imageio-ffmpeg \
    scipy scikit-image seaborn \
    tensorflow==2.15.0 tensorflow-datasets==4.9.3 \
    tensorflow-graphics==2021.12.3 \
    trimesh einops tqdm rich \
    pyopengl glfw \
    easydict pynput evdev

# ── 7. 应用 site-packages 补丁 ────────────────────────────────
echo "[7/7] Applying transformers patches (Layer Skip + cache fix)"
SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")

LLAMA_PATH="${SITE_PKG}/transformers/models/llama/modeling_llama.py"
CACHE_PATH="${SITE_PKG}/transformers/cache_utils.py"

echo "  Patching: ${LLAMA_PATH}"
cp "deploy/modified_site_packages/modeling_llama.py" "${LLAMA_PATH}"

echo "  Patching: ${CACHE_PATH}"
cp "deploy/modified_site_packages/cache_utils.py" "${CACHE_PATH}"

# 验证补丁
python -c "
import subprocess, sys
result = subprocess.run(['grep', '-c', 'Layer Skip', '${LLAMA_PATH}'], capture_output=True, text=True)
assert int(result.stdout.strip()) >= 1, 'Layer Skip patch NOT found in modeling_llama.py!'
print('✓ modeling_llama.py patch verified')

result2 = subprocess.run(['grep', '-c', 'isinstance.*torch.Tensor', '${CACHE_PATH}'], capture_output=True, text=True)
assert int(result2.stdout.strip()) >= 1, 'cache_utils.py patch NOT found!'
print('✓ cache_utils.py patch verified')
"

# ── 环境变量（headless rendering）────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Add to ~/.bashrc:"
echo "    export MUJOCO_GL=egl"
echo "    export PYTHONPATH=/path/to/LIBERO:\$PYTHONPATH"
echo ""
echo "  Model checkpoint: upload checkpoints/ dir or download from HuggingFace"
echo "  See deploy/bf16_changes.md for BF16 vs INT4 differences"
echo "============================================================"
