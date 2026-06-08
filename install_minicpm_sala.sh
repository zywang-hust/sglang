#!/bin/bash
set -e

# Get the directory where this script resides (i.e., the sglang repo root)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_ROOT}/sglang_minicpm_sala_env"

# PyPI mirror: prefer CLI argument, then env var, default to official source
if [ -n "$1" ]; then
    export UV_INDEX_URL="$1"
elif [ -z "${UV_INDEX_URL}" ]; then
    export UV_INDEX_URL="https://pypi.org/simple"
fi

echo "============================================"
echo " MiniCPM-SALA Installation (uv)"
echo "============================================"
echo "Root Directory: ${REPO_ROOT}"
echo "PyPI mirror:    ${UV_INDEX_URL}"

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "'uv' not found. Installing it via pip..."
    pip install uv || python -m pip install uv
    if ! command -v uv &> /dev/null; then
        echo "Error: failed to install 'uv'. Please install it manually (e.g., pip install uv) and re-run."
        exit 1
    fi
fi

# ---- Ensure uv-managed Python ----
REQUIRED_PY="3.12"
echo "[1/4] Ensuring Python ${REQUIRED_PY} is managed by uv..."
uv python install "${REQUIRED_PY}"
UV_PYTHON=$(uv python find --python-preference only-managed "${REQUIRED_PY}")
echo "  uv-managed Python: ${UV_PYTHON} ($(${UV_PYTHON} --version))"

# ---- Create venv ----
if [ -d "${VENV_DIR}" ]; then
    VENV_PY_VER=$("${VENV_DIR}/bin/python" --version 2>&1 | awk '{print $2}')
    if [[ "${VENV_PY_VER}" != ${REQUIRED_PY}.* ]]; then
        echo "  venv exists but Python version is ${VENV_PY_VER} (expected ${REQUIRED_PY}.x), recreating..."
        rm -rf "${VENV_DIR}"
        uv venv --python "${UV_PYTHON}" "${VENV_DIR}"
    else
        echo "  venv already exists (Python ${VENV_PY_VER}), skipping"
    fi
else
    echo "  Creating virtual environment..."
    uv venv --python "${UV_PYTHON}" "${VENV_DIR}"
fi

# Activate environment variables for the script execution
export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:$PATH"
echo "Python: $(python --version)"

# ---- Prepare build environment ----
# Fix compiler (python-build-standalone sets CXX="clang++ -pthread", incompatible with CMake)
if command -v g++ &> /dev/null; then
    export CC=gcc CXX=g++
fi
# Ensure nvcc is in PATH.
# Recommended: a full CUDA Toolkit at /usr/local/cuda (ships complete headers,
# incl. CCCL/libcu++ `nv/target`) so runtime JIT compilation works out of the box.
# A pip-wheel CUDA (nvidia-cu* wheels) is also supported, e.g. when a newer GPU
# (Blackwell sm120) needs CUDA 13 but the system nvcc is too old. That path needs
# a one-time setup (install nvidia-cuda-nvcc/runtime, create the libcudart.so
# symlink, export LIBRARY_PATH/LD_LIBRARY_PATH in the server shell). See the
# "使用 pip-wheel CUDA 的额外步骤" section in README_zh.md. JIT also auto-falls
# back to flashinfer's bundled CCCL headers, so `nv/target` needs no manual fix.
if [ -z "${CUDA_HOME}" ]; then
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        export CUDA_HOME="/usr/local/cuda"
    fi
fi
if [ -n "${CUDA_HOME}" ]; then
    export PATH="${CUDA_HOME}/bin:$PATH"
    export CUDACXX="${CUDA_HOME}/bin/nvcc"
fi

# ---- Install Packages ----

# Some `python[all]` deps (e.g. sgl-router) compile from source and need Rust.
# Warn early with a clear hint instead of failing with a cryptic
# "can't find Rust compiler" / "edition2024" later.
if ! command -v cargo &> /dev/null; then
    echo "Warning: Rust toolchain (cargo) not found. Some dependencies (e.g. sgl-router)"
    echo "         compile from source and may fail. If you hit 'can't find Rust compiler'"
    echo "         or 'edition2024', install Rust:"
    echo "           curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
    echo "           source \"\$HOME/.cargo/env\""
fi

# Install sglang from the repo root
echo "[2/4] Installing sglang (current directory)..."
uv pip install "cmake>=3.26"
uv pip install --upgrade pip setuptools wheel
uv pip install -e "${REPO_ROOT}/python[all]"

# Build and install CUDA kernels
# InfLLM v2 kernels (max_pooling / bitmask / flash attention stage1) are now part
# of sgl-kernel and are built from source here. The sparse_kernel get_block_table
# ops live in sglang.jit_kernel and are JIT-compiled at runtime (no build step).
# Use `make build` (wheel install) rather than `make install` (editable): the
# editable layout puts the arch-specific common_ops .so under site-packages while
# `import sgl_kernel` resolves to the source tree, so the sm90/sm100 loader can't
# find them. A wheel install keeps the package and its .so files together.
echo "[3/4] Building sgl-kernel (includes InfLLM v2 kernels)..."
cd "${REPO_ROOT}/sgl-kernel"
make build

# Install additional libraries
echo "[4/4] Installing additional libraries..."
uv pip install tilelang flash-linear-attention
# flash-linear-attention pulls in kernels>=0.15, whose LayerRepository requires an
# explicit revision/version and breaks `import sglang` (via transformers hub_kernels).
# Pin back to a compatible release until upstream is fixed.
uv pip install "kernels<0.15"

# ---- Done ----
echo ""
echo "============================================"
echo " Installation complete!"
echo "============================================"
echo "To activate the environment, run:"
echo "source ${VENV_DIR}/bin/activate"
