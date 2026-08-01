#include <torch/extension.h>

torch::Tensor cutlass_int8_mm(torch::Tensor a, torch::Tensor b);
torch::Tensor cutlass_int8_mm_scaled(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor row_scale,
    c10::optional<torch::Tensor> col_scale,
    c10::optional<torch::Tensor> bias,
    int64_t output_dtype);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int8_mm", &cutlass_int8_mm, "CUTLASS int8 GEMM returning int32");
  m.def("int8_mm_scaled", &cutlass_int8_mm_scaled, "CUTLASS int8 GEMM with W8A8 scale/bias epilogue");
}
