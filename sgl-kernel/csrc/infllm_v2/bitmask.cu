// InfLLM-V2 block-mask <-> uint64 bit-packing helpers, AOT build.
//
// Migrated from the following 3rdparty sources (device kernels kept faithful):
//   * 3rdparty/infllmv2_cuda_impl/csrc/topk_to_uint64.cuh
//   * 3rdparty/infllmv2_cuda_impl/csrc/uint64_to_bool.cuh
//   * 3rdparty/infllmv2_cuda_impl/csrc/blockmask_to_uint64.cuh
//
// Notes vs. the original implementation:
//   * Host launchers take pre-flattened 2D `at::Tensor`s and derive all integer
//     parameters from the tensor shapes; outputs are pre-allocated on the
//     Python side (as in the original wrappers).
//   * The uint64 representation is carried by torch.int64 tensors; boolean masks
//     are torch.bool tensors (1 byte, bit-identical to uint8 0/1) reinterpreted
//     as uint8 on the device side.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "utils.h"

namespace {

constexpr int kThreadsPerBlock = 256;

// One thread per output uint64 [row, col]; row in [0, batch), col in [0, n_uint64_per_row).
__global__ void
topk_to_uint64_kernel(const int* topk_idx, uint64_t* result, int batch_size, int k, int n_uint64_per_row) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= batch_size * n_uint64_per_row) return;
  const int row = linear / n_uint64_per_row;
  const int col = linear % n_uint64_per_row;
  const int bit_start = col * 64;

  uint64_t packed_value = 0;
  for (int i = 0; i < k; i++) {
    const int idx = topk_idx[row * k + i];
    if (idx == -1) continue;
    if (idx >= bit_start && idx < bit_start + 64) {
      packed_value |= (1ULL << (idx - bit_start));
    }
  }
  result[row * n_uint64_per_row + col] = packed_value;
}

// One thread per output bool [row, col]; row in [0, batch), col in [0, last_dim_size).
__global__ void
uint64_to_bool_kernel(const uint64_t* input, uint8_t* result, int batch_size, int last_dim_size, int n_uint64_per_row) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= batch_size * last_dim_size) return;
  const int row = linear / last_dim_size;
  const int col = linear % last_dim_size;
  const int uint64_idx = col / 64;
  const int bit_pos = col % 64;

  const uint64_t packed_value = input[row * n_uint64_per_row + uint64_idx];
  result[row * last_dim_size + col] = ((packed_value & (1ULL << bit_pos)) != 0) ? 1 : 0;
}

// One thread per output uint64 [row, col]; row in [0, batch), col in [0, n_uint64_per_row).
__global__ void blockmask_to_uint64_kernel(
    const uint8_t* blockmask, uint64_t* result, int batch_size, int last_dim_size, int n_uint64_per_row) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= batch_size * n_uint64_per_row) return;
  const int row = linear / n_uint64_per_row;
  const int col = linear % n_uint64_per_row;
  const int bit_start = col * 64;
  const int in_base = row * last_dim_size;

  uint64_t packed_value = 0;
  for (int bit = 0; bit < 64; bit++) {
    const int bit_pos = bit_start + bit;
    if (bit_pos < last_dim_size && blockmask[in_base + bit_pos] != 0) {
      packed_value |= (1ULL << bit);
    }
  }
  result[row * n_uint64_per_row + col] = packed_value;
}

}  // namespace

// topk_idx: [batch, k] int32 ; result: [batch, n_uint64_per_row] int64 (carries uint64).
void infllm_v2_topk_to_uint64(at::Tensor result, at::Tensor topk_idx) {
  TORCH_CHECK(topk_idx.dim() == 2 && result.dim() == 2, "topk_idx/result must be 2D (flattened)");
  TORCH_CHECK(topk_idx.scalar_type() == at::kInt, "topk_idx must be int32");
  TORCH_CHECK(result.scalar_type() == at::kLong, "result must be int64");

  const int batch_size = static_cast<int>(topk_idx.size(0));
  const int k = static_cast<int>(topk_idx.size(1));
  const int n_uint64_per_row = static_cast<int>(result.size(1));

  const int64_t total = static_cast<int64_t>(batch_size) * n_uint64_per_row;
  if (total == 0) return;
  const int64_t num_blocks = (total + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  topk_to_uint64_kernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(
      topk_idx.data_ptr<int>(),
      reinterpret_cast<uint64_t*>(result.data_ptr<int64_t>()),
      batch_size,
      k,
      n_uint64_per_row);
}

// input: [batch, n_uint64_per_row] int64 ; result: [batch, last_dim_size] bool.
void infllm_v2_uint64_to_bool(at::Tensor result, at::Tensor input) {
  TORCH_CHECK(input.dim() == 2 && result.dim() == 2, "input/result must be 2D (flattened)");
  TORCH_CHECK(input.scalar_type() == at::kLong, "input must be int64");
  TORCH_CHECK(result.scalar_type() == at::kBool, "result must be bool");

  const int batch_size = static_cast<int>(input.size(0));
  const int n_uint64_per_row = static_cast<int>(input.size(1));
  const int last_dim_size = static_cast<int>(result.size(1));

  const int64_t total = static_cast<int64_t>(batch_size) * last_dim_size;
  if (total == 0) return;
  const int64_t num_blocks = (total + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  uint64_to_bool_kernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const uint64_t*>(input.data_ptr<int64_t>()),
      reinterpret_cast<uint8_t*>(result.data_ptr<bool>()),
      batch_size,
      last_dim_size,
      n_uint64_per_row);
}

// blockmask: [batch, last_dim_size] bool ; result: [batch, n_uint64_per_row] int64.
void infllm_v2_blockmask_to_uint64(at::Tensor result, at::Tensor blockmask) {
  TORCH_CHECK(blockmask.dim() == 2 && result.dim() == 2, "blockmask/result must be 2D (flattened)");
  TORCH_CHECK(blockmask.scalar_type() == at::kBool, "blockmask must be bool");
  TORCH_CHECK(result.scalar_type() == at::kLong, "result must be int64");

  const int batch_size = static_cast<int>(blockmask.size(0));
  const int last_dim_size = static_cast<int>(blockmask.size(1));
  const int n_uint64_per_row = static_cast<int>(result.size(1));

  const int64_t total = static_cast<int64_t>(batch_size) * n_uint64_per_row;
  if (total == 0) return;
  const int64_t num_blocks = (total + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  blockmask_to_uint64_kernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(blockmask.data_ptr<bool>()),
      reinterpret_cast<uint64_t*>(result.data_ptr<int64_t>()),
      batch_size,
      last_dim_size,
      n_uint64_per_row);
}
