/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// Pybind entry point for the fused RMS-norm + interleaved RoPE kernel.

#include <torch/extension.h>
#include <c10/util/Optional.h>

at::Tensor rms_norm_rope(at::Tensor &x, c10::optional<at::Tensor> &weights_, at::Tensor &cos_freqs, at::Tensor &sin_freqs, bool out_16bit);
at::Tensor rms_norm_split_rope(at::Tensor &x, at::Tensor &sin_freqs, at::Tensor &cos_freqs, at::Tensor &weights, bool out_fp8);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm_rope", &rms_norm_rope, "fused RMS norm + interleaved RoPE + cvt");
    m.def("rms_norm_split_rope", &rms_norm_split_rope, "fused RMS norm + split RoPE + cvt");
}
