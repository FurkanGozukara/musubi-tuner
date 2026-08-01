#include <torch/extension.h>

torch::Tensor cutlass_int4_mm(torch::Tensor a, torch::Tensor b_t, int64_t k);
torch::Tensor cutlass_int4_linear(
    torch::Tensor a,
    torch::Tensor b_t,
    torch::Tensor row_scale,
    c10::optional<torch::Tensor> col_scale,
    c10::optional<torch::Tensor> bias,
    int64_t output_dtype,
    int64_t k);
torch::Tensor cutlass_int4_linear_backward_input(
    torch::Tensor qg,
    torch::Tensor packed_weight,
    torch::Tensor row_scale,
    int64_t output_dtype,
    int64_t out_features,
    int64_t padded_features);
torch::Tensor transpose_packed_int4(torch::Tensor packed, int64_t logical_cols);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int4_mm", &cutlass_int4_mm, "CUTLASS native int4 GEMM returning int32");
  m.def("linear", &cutlass_int4_linear, "CUTLASS native int4 GEMM with scaled output epilogue");
  m.def(
      "linear_backward_input",
      &cutlass_int4_linear_backward_input,
      "CUTLASS native int4 input-gradient GEMM with packed transpose and scaled output epilogue");
  m.def("transpose_packed", &transpose_packed_int4, "Transpose a packed signed-int4 matrix");
}
