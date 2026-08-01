#include <torch/extension.h>

#include <tuple>

std::tuple<torch::Tensor, torch::Tensor> quantize_rowwise_convrot(torch::Tensor x, int64_t group_size);
std::tuple<torch::Tensor, torch::Tensor> quantize_rowwise(torch::Tensor x, c10::optional<torch::Tensor> col_scale);
torch::Tensor dequant_epilogue(
    torch::Tensor out_int32,
    torch::Tensor row_scale,
    c10::optional<torch::Tensor> col_scale,
    c10::optional<torch::Tensor> bias,
    int64_t output_dtype);
torch::Tensor dequant_epilogue_convrot(torch::Tensor out_int32, torch::Tensor row_scale, int64_t group_size, int64_t output_dtype);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize_rowwise_convrot", &quantize_rowwise_convrot, "CUDA ConvRot row-wise INT8 activation quantization");
  m.def("quantize_rowwise", &quantize_rowwise, "CUDA row-wise INT8 activation quantization");
  m.def("dequant_epilogue", &dequant_epilogue, "CUDA INT8 dequantization epilogue");
  m.def("dequant_epilogue_convrot", &dequant_epilogue_convrot, "CUDA INT8 ConvRot dequantization epilogue");
}
