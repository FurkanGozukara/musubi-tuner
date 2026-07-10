#include <torch/extension.h>

torch::Tensor cutlass_int4_linear_fused(
    torch::Tensor a,
    torch::Tensor b_t,
    torch::Tensor row_scale,
    c10::optional<torch::Tensor> col_scale,
    c10::optional<torch::Tensor> bias,
    int64_t output_dtype,
    int64_t k);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "linear_fused",
      &cutlass_int4_linear_fused,
      "Fused CUTLASS int4 GEMM with in-epilogue per-row/per-col dequant, optional bias, and typed output");
}
