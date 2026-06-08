<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://static.pepy.tech/badge/sglang?period=month)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

</div>

> [!WARNING]
> **MiniCPM-Sala 定制版本**
>
> 本仓库是 SGLang 针对 **OpenBMB MiniCPM-Sala** 模型优化的定制版本。
> 请 **仅** 使用此版本来运行 MiniCPM-Sala 模型。
>
> **环境依赖:**
> InfLLM V2 CUDA kernels（max_pooling / bitmask / flash attention stage1）已内置到 `sgl-kernel`，
> 随 `sgl-kernel` 一起从源码编译；sparse_kernel 的 `get_block_table` 算子已迁移到
> `sglang.jit_kernel`，在运行时按需 JIT 编译，无需单独安装。

--------------------------------------------------------------------------------

[English](./README.md) | [**中文**]

# MiniCPM-SALA 推理环境搭建流程

## 环境要求

- CUDA 12.x 或更高版本（**新架构 GPU（如 Blackwell sm120）需 CUDA 13**，
  详见下方 [使用 pip-wheel CUDA 的额外步骤](#使用-pip-wheel-cuda-的额外步骤)）
- `gcc` / `g++` 编译器
- Rust 工具链（`pip install -e python[all]` 的部分依赖如 `sgl-router` 需从源码编译；
  缺失时会报 `can't find Rust compiler` 或 `edition2024`，见 [Q&A](#qa)）
- `uv` 包管理器（脚本会自动检测）

> **推荐：使用完整的 CUDA Toolkit**（即包含完整头文件的 `/usr/local/cuda`，
> 绝大多数 NVIDIA 官方 docker 镜像自带）。这是最干净、最快的路径，运行时 JIT
> 编译可直接使用 toolkit 自带的 CCCL/libcu++ 头文件。
>
> 若你的 CUDA 来自 pip wheel（`nvidia-cu*` 系列）等环境，本仓库也**受支持**，
> 但需注意两点：
> - **缺头文件（`nv/target` 等 CCCL 头文件）**：JIT 编译会自动回退使用 `flashinfer`
>   内置的 CCCL 头文件，**无需手动处理**。
> - **缺 `libcudart.so` 软链导致链接报 `cannot find -lcudart`**：需按下方
>   [使用 pip-wheel CUDA 的额外步骤](#使用-pip-wheel-cuda-的额外步骤) 先配置一次环境
>   （建软链 + 导出库搜索路径），这部分无法在代码里自动兜底。

## 快速开始

### 安装

```bash
# 克隆仓库
git clone -b minicpm_sala https://github.com/OpenBMB/sglang.git
cd sglang

# 一键安装（自动创建虚拟环境并编译所有依赖）
bash install_minicpm_sala.sh

# 或指定 PyPI 镜像源
bash install_minicpm_sala.sh https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

安装脚本会自动完成以下步骤：

1. 创建 `sglang_minicpm_sala_env` 虚拟环境（Python 3.12）
2. 安装 MiniCPM-SALA (当前仓库)
3. 从源码编译安装 `sgl-kernel`（内置 InfLLM v2 kernels）
4. 安装 `tilelang` 和 `flash-linear-attention`

### 使用

```bash
# 激活环境
source sglang_minicpm_sala_env/bin/activate

# 启动推理服务（将 MODEL_PATH 替换为实际模型路径）
MODEL_PATH=/path/to/your/model

python3 -m sglang.launch_server \
    --model ${MODEL_PATH} \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size 8192 \
    --max-running-requests 32 \
    --skip-server-warmup \
    --port 31111 \
    --dense-as-sparse
```

| 参数 | 说明 |
|------|------|
| `--trust-remote-code` | 允许加载模型自带的自定义代码 |
| `--disable-radix-cache` | 禁用 RadixAttention 前缀缓存 |
| `--attention-backend minicpm_flashinfer` | 使用 MiniCPM FlashInfer 注意力后端 |
| `--chunked-prefill-size 8192` | chunked prefill 大小 |
| `--max-running-requests 32` | 最大并发推理请求数 |
| `--skip-server-warmup` | 跳过服务预热 |
| `--port 31111` | 服务端口 |
| `--dense-as-sparse` | 使用 dense-as-sparse 模式 |

> **提示：** 为获得最佳生成效果，建议在请求时设置 `temperature=0.9`。

### 工具调用（Tool Calling）

启动服务时添加 `--tool-call-parser minicpm4_xml` 参数即可启用工具调用：

```bash
# 激活环境
source sglang_minicpm_sala_env/bin/activate

# 启动推理服务（启用工具调用，将 MODEL_PATH 替换为实际模型路径）
MODEL_PATH=/path/to/your/model

python3 -m sglang.launch_server \
    --model ${MODEL_PATH} \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size 8192 \
    --max-running-requests 32 \
    --skip-server-warmup \
    --port 31111 \
    --dense-as-sparse \
    --tool-call-parser minicpm4_xml
```

**请求示例：**

```bash
curl -X POST "http://localhost:31111/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minicpm4.6-8b",
    "messages": [{"role": "user", "content": "北京天气怎么样"}],
    "chat_template_kwargs": {"enable_thinking": false},
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "查询天气",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"}
          }
        }
      }
    }]
  }'
```

## 目录结构

```
sglang/
├── README.md                       # 英文文档
├── README_zh.md                    # 本文件（中文文档）
├── install_minicpm_sala.sh         # 一键安装脚本
├── sgl-kernel/                     # CUDA kernels（含 InfLLM v2: max_pooling / bitmask / flash stage1）
├── python/                         # SGLang 源码（sparse_kernel get_block_table 位于 sglang/jit_kernel）
└── ...
```

## 手动安装

如果一键脚本不满足需求，可以分步执行：

```bash
# 0. 确保 uv 可用
pip install uv

# 1. 创建虚拟环境
uv venv --python 3.12 sglang_minicpm_sala_env
source sglang_minicpm_sala_env/bin/activate

# 2. 安装 SGLang
uv pip install --upgrade pip setuptools wheel
uv pip install -e ./python[all]

# 3. 从源码编译安装 sgl-kernel（内置 InfLLM v2 kernels）
# 用 `make build`（构建 wheel 后安装），不要用 `make install`（editable 模式下
# 架构相关的 common_ops .so 装到 site-packages，而 import 解析到源码树，sm90/sm100
# 加载器会找不到）。
cd sgl-kernel && make build && cd ..

# 4. 安装额外依赖
uv pip install tilelang flash-linear-attention
# flash-linear-attention 会把 kernels 升到 >=0.15，其 LayerRepository 要求显式指定
# revision/version，会导致 `import sglang`（经 transformers hub_kernels）报错。
# 暂时回退到兼容版本，待上游修复后可移除。
uv pip install "kernels<0.15"
```

> **提示（查看编译进度）：** `make build` 默认只显示一个 uv 转圈，看不到逐文件进度。
> 若想看到 `[当前/总数] 文件名` 形式的实时进度，可用下面这条命令替代上面的 `make build`
> （产物完全等价，只是日志更详细）：
>
> ```bash
> cd sgl-kernel
> rm -rf dist .skbuild
> CMAKE_POLICY_VERSION_MINIMUM=3.5 \
> CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
> stdbuf -oL -eL uv build --wheel -Cbuild-dir=.skbuild -Cbuild.verbose=true . --no-build-isolation 2>&1 \
>   | tee build.log \
>   | stdbuf -oL sed -uE '/\[[0-9]+\/[0-9]+\]/ s#(\[[0-9]+/[0-9]+\]).*/([^/ ]+)#\1 \2#'
> pip install dist/*.whl --force-reinstall --no-deps
> cd ..
> ```
>
> 说明：
> - `CMAKE_POLICY_VERSION_MINIMUM=3.5` 必不可少（`make build` 内部已带；手写 `uv build` 时
>   缺它会因 dlpack 等老依赖报 `Compatibility with CMake < 3.5 has been removed` 而失败）。
> - 上面用 `sed` 而非过滤式 `awk`：配置（configure）阶段会照常打印 CMake 输出，不会出现
>   「长时间空白像卡住」的情况；编译阶段的行则被缩短为 `[12/120] flash_fwd_hdim128_bf16_sm80.cu.o`。
> - 完整日志保留在 `sgl-kernel/build.log`，失败时查看 `tail -50 sgl-kernel/build.log`。
> - 这条带管道的命令仅适合**手动交互**执行，**不要**放进安装脚本——管道退出码取自最后的 `sed`，
>   会掩盖 `uv build` 的失败。

## 使用 pip-wheel CUDA 的额外步骤

> 仅当你的 CUDA 来自 pip wheel（如 `nvidia-cu13`）、而非完整 CUDA Toolkit
> （`/usr/local/cuda`）时才需要本节。用完整 toolkit 可整节跳过。

**什么时候会走到这条路？** 当 GPU 架构较新（如 Blackwell sm120），而系统自带的 nvcc
版本过旧（如 CUDA 12.8）不支持该架构时，需要换用与 PyTorch 一致的更高版本 CUDA 工具链
（如 CUDA 13），最省事的方式是用 pip wheel 安装。先确认 torch 的 CUDA 版本：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# 例如 2.11.0+cu130 / 13.0 —— 说明需用 CUDA 13 的 nvcc，不能用系统 12.8
```

### 1. 安装 cu13 工具链（nvcc + runtime + 头文件）

```bash
uv pip install nvidia-cuda-nvcc nvidia-cuda-cccl nvidia-cuda-runtime
```

- `nvidia-cuda-nvcc`：提供 CUDA 13 的 `nvcc` / `ptxas`（系统 12.8 不支持 sm120）。
- `nvidia-cuda-runtime`：提供 `libcudart.so.13`（否则后续链接期报 `cannot find -lcudart`）。
- `nvidia-cuda-cccl`：CCCL 头文件。本仓库 JIT 已能自动回退用 `flashinfer` 内置 CCCL，
  所以这个**装不装都行**（不装也不会报 `nv/target`）。

> ⚠️ **最容易踩的坑：所有 `nvidia-*` CUDA 组件必须对齐到同一个 CUDA 小版本，
> 且与 PyTorch 的 CUDA 版本一致。** 这些 wheel 各自独立版本号，`uv pip install`
> 不加约束时很容易把不同组件解析到不同小版本（如 nvcc 13.3 + runtime 13.0 +
> `nvidia-nvvm` 13.3），从而在编译期报错（见 [Q&A](#qa) 的两条版本不一致问题）。
> 装完后**务必核对版本一致**：
>
> ```bash
> python -c "import torch; print('torch cuda', torch.version.cuda)"  # 例如 13.0
> pip list | grep -iE "nvidia-cuda-(nvcc|crt|nvrtc|runtime|cupti)|nvidia-nvvm"
> ```
>
> 上面列出的所有组件（**尤其别漏了 `nvidia-nvvm`**，它提供决定 PTX 版本的
> `cicc`/`libnvvm`）小版本必须一致。若不一致，显式 pin 到与 torch 相同的小版本，例如：
>
> ```bash
> uv pip install \
>   "nvidia-cuda-nvcc==13.0.88" "nvidia-cuda-crt==13.0.88" \
>   "nvidia-cuda-nvrtc==13.0.88" "nvidia-nvvm==13.0.88"
> ```

### 2. 建 libcudart 软链（永久，做一次即可）

pip wheel 只装了带版本号的 `libcudart.so.13`，缺链接器 `ld` 需要的非版本号软链
`libcudart.so`，否则运行时 JIT 链接报 `cannot find -lcudart`：

```bash
export CUDA_HOME="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13"
ln -sf "$CUDA_HOME/lib/libcudart.so.13" "$CUDA_HOME/lib/libcudart.so"
```

### 3. 导出工具链与库搜索路径

⚠️ **必须在启动 server 的那个 shell 里设置**（这些变量不会从安装脚本自动带过来；
每次开新 shell 起服务前都要先 export，建议写进你的启动脚本或 venv 的 `activate`）：

```bash
export CUDA_HOME="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"                        # 用 cu13 的 nvcc
export LIBRARY_PATH="$CUDA_HOME/lib:$LIBRARY_PATH"        # 链接期：ld 找 libcudart.so
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"  # 运行期：加载 libcudart.so.13

nvcc --version   # 确认是 13.x，且路径在 nvidia/cu13 下，而非 /usr/local/cuda
```

### 4. 清掉可能存在的旧缓存

若之前用错过 CUDA 路径（例如指向并不存在的 `/usr/local/cuda-12.9`），或换过工具链版本，
旧状态会被各种缓存记住，后续即使改对了也仍报错。换工具链后建议把几类缓存一起清掉重来：

```bash
rm -rf ~/.cache/tvm-ffi ~/.cache/tvm_ffi ~/.cache/flashinfer ~/.triton/cache
```

（若设置了 `TVM_FFI_CACHE_DIR` / `TRITON_CACHE_DIR` 则删对应目录。）

配好后再启动 server。若能越过 `Capture cuda graph` 并打印出 server ready，说明
工具链接线正确。

## Q&A

**Q: CUDA 扩展编译失败怎么办？**

- 确保系统安装了 CUDA 12 以上（`nvcc --version` 检查）。
- 确保 `gcc` / `g++` 可用。
- 如果 `CXX` 环境变量被设为 `clang++ -pthread`，手动 `export CXX=g++`。
- 确保 **nvcc 版本与 PyTorch 的 CUDA 版本一致**（`python -c "import torch; print(torch.version.cuda)"`）。
  新架构 GPU（如 Blackwell sm120）需 CUDA 13，详见
  [使用 pip-wheel CUDA 的额外步骤](#使用-pip-wheel-cuda-的额外步骤)。

**Q: 安装报 `can't find Rust compiler` 或 `edition2024` 怎么办？**

- `pip install -e python[all]` 的部分依赖（如 `sgl-router`）需从源码编译，要求本机有
  Rust 工具链。安装：

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustc --version   # 建议 ≥ 1.85（旧版可能不认 edition2024）
```

**Q: 运行时 JIT 编译报 `fatal error: nv/target: No such file or directory` 怎么办？**

- 这是因为当前 CUDA 环境（常见于 pip wheel `nvidia-cu*` 安装）头文件不完整，
  缺少 CCCL/libcu++ 头文件（`nv/target` 等）。本仓库的 JIT 加载器会自动回退使用
  `flashinfer` 内置的 CCCL 头文件，因此正常情况下无需手动处理。
- 如果仍然报错，通常是**旧的 JIT 缓存**复用了之前的编译失败状态。清掉缓存后重试：
  `rm -rf ~/.cache/tvm-ffi ~/.cache/tvm_ffi ~/.cache/flashinfer`（若设置了 `TVM_FFI_CACHE_DIR` 则删对应目录）。
- 最干净的解法仍是使用完整 CUDA Toolkit（`/usr/local/cuda`），从根本上避免缺头文件。

**Q: 运行时 JIT 链接报 `cannot find -lcudart` 怎么办？**

- 常见于 pip wheel（`nvidia-cu*`）安装的 CUDA：只有 `libcudart.so.13`，缺少链接器需要的
  非版本号软链 `libcudart.so`。按
  [使用 pip-wheel CUDA 的额外步骤](#使用-pip-wheel-cuda-的额外步骤)
  建软链并在**启动 server 的 shell** 里导出 `LIBRARY_PATH` / `LD_LIBRARY_PATH` 即可。
- 完整 CUDA Toolkit（`/usr/local/cuda`）自带该软链，不会触发此问题。

**Q: 编译报 `CUDA compiler and CUDA toolkit headers are incompatible` 怎么办？**

- 这是 CCCL 的版本检查：**`nvcc` 编译器版本必须等于 CUDA runtime 头文件的 `CUDART_VERSION`**。
  报错说明 `nvidia-cuda-nvcc` 的版本和 `nvidia-cuda-runtime` 头文件版本不一致
  （如 nvcc 13.3 + cudart 13.0）。
- 由于 runtime 要与 torch 的 CUDA 版本一致，正确做法是**把 nvcc 对齐到 runtime 的小版本**
  （而非反过来）。参见 [pip-wheel 小节步骤 1](#1-安装-cu13-工具链nvcc--runtime--头文件) 的版本核对与 pin 方法。

**Q: 编译报 `ptxas ... Unsupported .version 9.3; current version is '9.0'` 怎么办？**

- PTX 由前端 `cicc`/`libnvvm`（来自 **`nvidia-nvvm`** 包）生成，再交给 `ptxas`（来自
  `nvidia-cuda-nvcc` 包）汇编。这两个包**版本不一致**时，前端产出的 PTX 版本高于 ptxas 支持的
  版本，就报此错（如 `nvidia-nvvm` 13.3 生成 9.3 PTX，但 13.0 的 ptxas 只支持 9.0）。
- 修复：把 `nvidia-nvvm` 对齐到与 `nvidia-cuda-nvcc` 相同的小版本，例如
  `uv pip install "nvidia-nvvm==13.0.88"`。这是降级 nvcc 时**最容易漏掉**的一个包。
- 改完版本后记得清缓存重编：`rm -rf ~/.cache/flashinfer ~/.cache/tvm-ffi ~/.cache/tvm_ffi`。

**Q: 运行时报 `json.decoder.JSONDecodeError: Expecting value: line 1 column 1` 怎么办？**

- 这是 **Triton 内核缓存损坏**（某个缓存的 metadata JSON 是空/截断的，多因之前的运行在写缓存时
  被 kill/OOM/Ctrl+C 中断）。和模型迁移无关。
- 清掉 Triton 缓存后重跑即可：`rm -rf ~/.triton/cache`（设了 `TRITON_CACHE_DIR` 则删对应目录）。
- 注意：稀疏路径（不带 `--force-dense-minicpm`，如 RULER）才会触发部分 Triton kernel，
  所以可能 dense 的 GSM8K 跑通、稀疏的 RULER 才暴露此问题。
