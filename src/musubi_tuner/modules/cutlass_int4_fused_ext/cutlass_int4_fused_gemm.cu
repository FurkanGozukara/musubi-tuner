// Fused CUTLASS int4 GEMM + dequant epilogue for the W4A4 ConvRot path.
//
// The unfused path (cutlass_int4_ext) writes the int32 [M,N] accumulator to global
// memory and reads it back in a separate dequant kernel. This kernel folds the
// per-row (activation) scale, per-column (weight) scale, optional bias, and the
// int32->bf16 cast into an Sm80 Epilogue Visitor Tree so the int32 accumulator is
// never materialized in global memory. Layout mirrors cutlass_int4_ext: A row-major
// [M,K] packed int4, B column-major [N,K] packed int4, C row-major [M,N].
//
// Adapted from PyTorch's Sm89 rowwise-scaled MM (aten RowwiseScaledMM.cu), retargeted
// to Sm80 signed int4b_t operands with an int32 accumulator.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_runtime.h>
#include <limits>

#include <cutlass/cutlass.h>
// GemmUniversalBase pulls only the 2.x kernel tree; the 3.x GemmUniversalAdapter
// transitively includes unconditional SM100 (Blackwell) headers that do not compile
// under nvcc + MSVC (constexpr dim3), which is why PyTorch disables its EVT path on
// Windows. The visitor kernel here is a 2.x kernel, so Base is the correct launcher.
#include <cutlass/gemm/device/gemm_universal_base.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

namespace ltx2_cutlass_int4_fused {

using ElementInput = cutlass::int4b_t;
using ElementAccum = int32_t;
using ElementEpilogue = float;
using ElementScale = float;
using ElementBias = float;

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// int4 operands are 128-bit aligned (32 elements per 128-bit vector).
static constexpr int AlignmentInputA = 32;
static constexpr int AlignmentInputB = 32;

using ArchTag = cutlass::arch::Sm80;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 128>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using Operator = cutlass::arch::OpMultiplyAddSaturate;
static constexpr int NumStages = 3;
static constexpr int NumEVTEpilogueStages = 1;

template <typename ElementOutput>
struct FusedGemm {
  static constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
      ThreadblockShape, WarpShape, ElementOutput, AlignmentOutput, NumEVTEpilogueStages>;

  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

  // Per-row (M) activation scale.
  using RowScale = cutlass::epilogue::threadblock::VisitorColBroadcast<
      OutputTileThreadMap, ElementScale, cute::Stride<cute::_1, cute::_0, int64_t>>;
  // Per-column (N) weight scale.
  using ColScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap, ElementScale, cute::Stride<cute::_0, cute::_1, int64_t>>;
  // Per-column (N) bias.
  using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap, ElementBias, cute::Stride<cute::_0, cute::_1, int64_t>>;

  using ApplyRowScale = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, ElementEpilogue, ElementEpilogue, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTRowScale = cutlass::epilogue::threadblock::Sm80EVT<ApplyRowScale, Accum, RowScale>;

  using ApplyColScale = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, ElementEpilogue, ElementEpilogue, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTColScale = cutlass::epilogue::threadblock::Sm80EVT<ApplyColScale, EVTRowScale, ColScale>;

  using ApplyBias = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::plus, ElementEpilogue, ElementEpilogue, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTBias = cutlass::epilogue::threadblock::Sm80EVT<ApplyBias, EVTColScale, Bias>;

  using Output = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, ElementOutput, cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<int64_t, cute::_1, int64_t>>;
  using EVTOutput = cutlass::epilogue::threadblock::Sm80EVT<Output, EVTBias>;

  using EVTKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementInput, LayoutInputA, cutlass::ComplexTransform::kNone, AlignmentInputA,
      ElementInput, LayoutInputB, cutlass::ComplexTransform::kNone, AlignmentInputB,
      ElementOutput, LayoutOutput, AlignmentOutput,
      ElementAccum, ElementEpilogue, OperatorClass, ArchTag,
      ThreadblockShape, WarpShape, InstructionShape,
      EVTOutput, ThreadblockSwizzle, NumStages, Operator, NumEVTEpilogueStages>::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalBase<EVTKernel>;
};

template <typename ElementOutput>
void run_impl(
    torch::Tensor a,
    torch::Tensor b_t,
    torch::Tensor row_scale,
    const float* col_scale_ptr,
    const float* bias_ptr,
    torch::Tensor out,
    int M,
    int N,
    int K) {
  using GemmT = FusedGemm<ElementOutput>;
  using Gemm = typename GemmT::Gemm;
  using Output = typename GemmT::Output;
  using EVTOutput = typename GemmT::EVTOutput;

  cutlass::gemm::GemmCoord problem_size(M, N, K);
  constexpr int SplitKFactor = 1;

  typename GemmT::RowScale::Arguments row_scale_args{
      row_scale.data_ptr<float>(), ElementScale(1), {cute::_1{}, cute::_0{}, problem_size.m()}};
  typename GemmT::ColScale::Arguments col_scale_args{
      col_scale_ptr, ElementScale(1), {cute::_0{}, cute::_1{}, problem_size.n()}};
  typename GemmT::Bias::Arguments bias_args{
      bias_ptr, ElementBias(0), {cute::_0{}, cute::_1{}, problem_size.n()}};
  typename Output::Arguments output_args{
      reinterpret_cast<ElementOutput*>(out.data_ptr()),
      {problem_size.n(), cute::_1{}, problem_size.mn().product()}};

  typename EVTOutput::Arguments callback_args{
      {
          {
              {
                  {},              // Accum
                  row_scale_args,  // RowScale
                  {}               // ApplyRowScale
              },                   // EVTRowScale
              col_scale_args,      // ColScale
              {}                   // ApplyColScale
          },                       // EVTColScale
          bias_args,               // Bias
          {}                       // ApplyBias
      },                           // EVTBias
      output_args                  // Output
  };

  typename Gemm::Arguments arguments(
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      SplitKFactor,
      callback_args,
      reinterpret_cast<ElementInput const*>(a.data_ptr<uint8_t>()),
      reinterpret_cast<ElementInput const*>(b_t.data_ptr<uint8_t>()),
      nullptr,
      nullptr,
      problem_size.mk().product(),
      problem_size.nk().product(),
      0,
      0,
      problem_size.k(),
      problem_size.k(),
      0,
      0);

  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = a.new_empty({static_cast<int64_t>(workspace_size)}, at::TensorOptions().dtype(at::kByte));

  cutlass::Status status = gemm.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "fused CUTLASS int4 GEMM cannot implement this problem");
  status = gemm.initialize(arguments, workspace.data_ptr());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "fused CUTLASS int4 GEMM failed to initialize");
  status = gemm(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "fused CUTLASS int4 GEMM launch failed");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_uint8_matrix(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == torch::kUInt8, name, " must be torch.uint8");
  TORCH_CHECK(t.dim() == 2, name, " must be rank-2");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous row-major");
}

c10::ScalarType dtype_from_code(int64_t code) {
  if (code == 0) return torch::kFloat16;
  if (code == 1) return torch::kBFloat16;
  if (code == 2) return torch::kFloat32;
  TORCH_CHECK(false, "Unsupported fused CUTLASS int4 output dtype code: ", code);
}

torch::Tensor run_linear_fused(
    torch::Tensor a,
    torch::Tensor b_t,
    torch::Tensor row_scale,
    c10::optional<torch::Tensor> col_scale,
    c10::optional<torch::Tensor> bias,
    int64_t output_dtype,
    int64_t logical_k) {
  check_uint8_matrix(a, "a");
  check_uint8_matrix(b_t, "b_t");
  TORCH_CHECK(a.device() == b_t.device(), "a and b_t must be on the same device");
  TORCH_CHECK(logical_k > 0 && logical_k <= std::numeric_limits<int>::max(), "invalid logical K");
  TORCH_CHECK(logical_k % 64 == 0, "fused CUTLASS int4 GEMM requires K divisible by 64");
  TORCH_CHECK((logical_k + 1) / 2 == a.size(1), "a packed width does not match logical K");
  TORCH_CHECK((logical_k + 1) / 2 == b_t.size(1), "b_t packed width does not match logical K");

  const int64_t M = a.size(0);
  const int64_t N = b_t.size(0);
  TORCH_CHECK(M <= std::numeric_limits<int>::max() && N <= std::numeric_limits<int>::max(), "M/N too large");

  TORCH_CHECK(row_scale.is_cuda() && row_scale.scalar_type() == torch::kFloat32, "row_scale must be CUDA float32");
  auto row_scale_c = row_scale.reshape({-1}).contiguous();
  TORCH_CHECK(row_scale_c.numel() == M, "row_scale must have M elements");

  const float* col_scale_ptr = nullptr;
  torch::Tensor col_scale_c;
  if (col_scale.has_value() && col_scale->defined() && col_scale->numel() > 0) {
    col_scale_c = col_scale->reshape({-1}).contiguous().to(torch::kFloat32);
    TORCH_CHECK(col_scale_c.is_cuda() && col_scale_c.numel() == N, "col_scale must be CUDA with N elements");
    col_scale_ptr = col_scale_c.data_ptr<float>();
  }

  const float* bias_ptr = nullptr;
  torch::Tensor bias_c;
  if (bias.has_value() && bias->defined() && bias->numel() > 0) {
    bias_c = bias->reshape({-1}).contiguous().to(torch::kFloat32);
    TORCH_CHECK(bias_c.is_cuda() && bias_c.numel() == N, "bias must be CUDA with N elements");
    bias_ptr = bias_c.data_ptr<float>();
  }

  auto out = torch::empty({M, N}, a.options().dtype(dtype_from_code(output_dtype)));
  if (M == 0 || N == 0) {
    return out;
  }

  c10::cuda::CUDAGuard guard(a.device());
  if (output_dtype == 0) {
    run_impl<cutlass::half_t>(a, b_t, row_scale_c, col_scale_ptr, bias_ptr, out,
                              static_cast<int>(M), static_cast<int>(N), static_cast<int>(logical_k));
  } else if (output_dtype == 1) {
    run_impl<cutlass::bfloat16_t>(a, b_t, row_scale_c, col_scale_ptr, bias_ptr, out,
                                  static_cast<int>(M), static_cast<int>(N), static_cast<int>(logical_k));
  } else {
    run_impl<float>(a, b_t, row_scale_c, col_scale_ptr, bias_ptr, out,
                    static_cast<int>(M), static_cast<int>(N), static_cast<int>(logical_k));
  }
  return out;
}

}  // namespace ltx2_cutlass_int4_fused

torch::Tensor cutlass_int4_linear_fused(
    torch::Tensor a,
    torch::Tensor b_t,
    torch::Tensor row_scale,
    c10::optional<torch::Tensor> col_scale,
    c10::optional<torch::Tensor> bias,
    int64_t output_dtype,
    int64_t k) {
  return ltx2_cutlass_int4_fused::run_linear_fused(a, b_t, row_scale, col_scale, bias, output_dtype, k);
}
