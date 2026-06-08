/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Pybind entry for the InfLLM-V2 FlashAttention backend (vendored from
// 3rdparty/infllmv2_cuda_impl). This builds as a standalone extension module
// `infllm_ops` so its `flash::` symbols stay isolated from sgl-kernel's own
// flash attention (`flash_ops` / `common_ops`).

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <vector>

// Forward declarations of the FlashAttention entry points implemented in
// flash_attn/flash_api.cpp. Signatures must match exactly.
std::vector<at::Tensor> mha_fwd(
    at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    c10::optional<at::Tensor>& out_,
    c10::optional<at::Tensor>& alibi_slopes_,
    const float p_dropout,
    const float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool return_softmax,
    c10::optional<at::Generator> gen_);

std::vector<at::Tensor> mha_varlen_fwd(
    at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    c10::optional<at::Tensor>& out_,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    c10::optional<at::Tensor>& seqused_k,
    c10::optional<const at::Tensor>& leftpad_k_,
    c10::optional<at::Tensor>& block_table_,
    c10::optional<at::Tensor>& alibi_slopes_,
    int max_seqlen_q,
    const int max_seqlen_k,
    const float p_dropout,
    const float softmax_scale,
    const bool zero_tensors,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool return_softmax,
    c10::optional<at::Generator> gen_,
    c10::optional<at::Tensor>& blockmask_);

std::vector<at::Tensor> mha_varlen_fwd_stage1(
    at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    c10::optional<at::Tensor>& out_,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    const at::Tensor& cu_seqlens_v,
    c10::optional<at::Tensor>& seqused_k,
    c10::optional<const at::Tensor>& leftpad_k_,
    c10::optional<at::Tensor>& block_table_,
    c10::optional<at::Tensor>& alibi_slopes_,
    int max_seqlen_q,
    const int max_seqlen_k,
    const float p_dropout,
    const float softmax_scale,
    const bool zero_tensors,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool return_softmax,
    c10::optional<at::Generator> gen_);

std::vector<at::Tensor> mha_bwd(
    const at::Tensor& dout,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& out,
    const at::Tensor& softmax_lse,
    c10::optional<at::Tensor>& dq_,
    c10::optional<at::Tensor>& dk_,
    c10::optional<at::Tensor>& dv_,
    c10::optional<at::Tensor>& alibi_slopes_,
    const float p_dropout,
    const float softmax_scale,
    const bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool deterministic,
    c10::optional<at::Generator> gen_,
    c10::optional<at::Tensor>& rng_state);

std::vector<at::Tensor> mha_varlen_bwd(
    const at::Tensor& dout,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& out,
    const at::Tensor& softmax_lse,
    c10::optional<at::Tensor>& dq_,
    c10::optional<at::Tensor>& dk_,
    c10::optional<at::Tensor>& dv_,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    c10::optional<at::Tensor>& alibi_slopes_,
    const int max_seqlen_q,
    const int max_seqlen_k,
    const float p_dropout,
    const float softmax_scale,
    const bool zero_tensors,
    const bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool deterministic,
    c10::optional<at::Tensor>& col_blockmask_,
    c10::optional<at::Generator> gen_,
    c10::optional<at::Tensor>& rng_state);

std::vector<at::Tensor> mha_fwd_kvcache(
    at::Tensor& q,
    const at::Tensor& kcache,
    const at::Tensor& vcache,
    c10::optional<const at::Tensor>& k_,
    c10::optional<const at::Tensor>& v_,
    c10::optional<const at::Tensor>& seqlens_k_,
    c10::optional<const at::Tensor>& rotary_cos_,
    c10::optional<const at::Tensor>& rotary_sin_,
    c10::optional<const at::Tensor>& cache_batch_idx_,
    c10::optional<const at::Tensor>& leftpad_k_,
    c10::optional<at::Tensor>& block_table_,
    c10::optional<at::Tensor>& alibi_slopes_,
    c10::optional<at::Tensor>& out_,
    const float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    bool is_rotary_interleaved,
    int num_splits,
    c10::optional<at::Tensor>& blockmask_);

PYBIND11_MODULE(infllm_ops, m) {
  m.doc() = "InfLLM V2 FlashAttention backend (vendored into sgl-kernel)";
  m.def("fwd", &mha_fwd, "Forward pass");
  m.def("varlen_fwd", &mha_varlen_fwd, "Forward pass (variable length)");
  m.def("varlen_fwd_stage1", &mha_varlen_fwd_stage1, "Forward pass (variable length) NSA stage 1");
  m.def("bwd", &mha_bwd, "Backward pass");
  m.def("varlen_bwd", &mha_varlen_bwd, "Backward pass (variable length)");
  m.def("fwd_kvcache", &mha_fwd_kvcache, "Forward pass, with KV-cache");
}
