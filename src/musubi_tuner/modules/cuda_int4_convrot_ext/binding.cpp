#include <torch/extension.h>

#include <tuple>

std::tuple<torch::Tensor, torch::Tensor> quantize_rowwise(torch::Tensor x);
torch::Tensor unpack_to_int8(torch::Tensor packed, int64_t num_values);
torch::Tensor transpose_packed(torch::Tensor packed, int64_t logical_cols);
torch::Tensor wmma_int4_mm(torch::Tensor a, torch::Tensor b_t, int64_t logical_k);
torch::Tensor dequant_epilogue(
    torch::Tensor acc,
    torch::Tensor row_scale,
    c10::optional<torch::Tensor> col_scale,
    c10::optional<torch::Tensor> bias,
    int64_t output_dtype);
torch::Tensor linear_forward(
    torch::Tensor qx,
    torch::Tensor qw,
    torch::Tensor row_scale,
    torch::Tensor col_scale,
    c10::optional<torch::Tensor> bias,
    int64_t output_dtype);
torch::Tensor linear_backward_input(
    torch::Tensor qg,
    torch::Tensor qw,
    torch::Tensor row_scale,
    int64_t padded_features,
    int64_t output_dtype);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize_rowwise", &quantize_rowwise, "CUDA packed INT4 row-wise activation quantization");
  m.def("unpack_to_int8", &unpack_to_int8, "CUDA packed INT4 to INT8 unpack");
  m.def("transpose_packed", &transpose_packed, "CUDA packed signed-INT4 transpose");
  m.def("wmma_int4_mm", &wmma_int4_mm, "CUDA WMMA signed-INT4 tensor-core GEMM returning int32");
  m.def("dequant_epilogue", &dequant_epilogue, "CUDA INT4 tensor-core bridge dequantization epilogue");
  m.def("linear_forward", &linear_forward, "CUDA packed INT4 W4A4 linear forward");
  m.def("linear_backward_input", &linear_backward_input, "CUDA packed INT4 W4A4 input-gradient matmul");
}
