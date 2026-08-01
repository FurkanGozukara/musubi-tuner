/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// Pybind entry point for the fused RMS-norm + interleaved RoPE kernel.

#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <vector>

at::Tensor rms_norm_rope(at::Tensor &x, c10::optional<at::Tensor> &weights_, at::Tensor &cos_freqs, at::Tensor &sin_freqs, bool out_16bit);
at::Tensor rms_norm_split_rope(at::Tensor &x, at::Tensor &sin_freqs, at::Tensor &cos_freqs, at::Tensor &weights, bool out_fp8);
std::vector<at::Tensor> rms_norm_split_rope_with_inv_rms(
    at::Tensor &x,
    at::Tensor &sin_freqs,
    at::Tensor &cos_freqs,
    at::Tensor &weights,
    bool out_fp8,
    double eps);
std::vector<at::Tensor> rms_norm_split_rope_backward_pair(
    at::Tensor grad_q,
    at::Tensor q,
    at::Tensor q_sin,
    at::Tensor q_cos,
    at::Tensor q_weight,
    at::Tensor q_inv_rms,
    at::Tensor grad_k,
    at::Tensor k,
    at::Tensor k_sin,
    at::Tensor k_cos,
    at::Tensor k_weight,
    at::Tensor k_inv_rms);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm_rope", &rms_norm_rope, "fused RMS norm + interleaved RoPE + cvt");
    m.def("rms_norm_split_rope", &rms_norm_split_rope, "fused RMS norm + split RoPE + cvt");
    m.def("rms_norm_split_rope_with_inv_rms", &rms_norm_split_rope_with_inv_rms);
    m.def("rms_norm_split_rope_backward_pair", &rms_norm_split_rope_backward_pair);
}
