#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <limits>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/matrix.h>

namespace ltx2_cutlass_int8 {

using CutlassInt8Gemm = cutlass::gemm::device::Gemm<
    int8_t,
    cutlass::layout::RowMajor,
    int8_t,
    cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 1, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;

void check_int8_matrix(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kInt8, name, " must be torch.int8");
  TORCH_CHECK(tensor.dim() == 2, name, " must be a rank-2 matrix");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous row-major");
}

void check_scale_vector(const torch::Tensor& tensor, const char* name, int64_t expected) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be torch.float32");
  TORCH_CHECK(tensor.numel() == expected, name, " must have ", expected, " elements");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

bool has_tensor(const c10::optional<torch::Tensor>& tensor) {
  return tensor.has_value() && tensor.value().defined() && tensor.value().numel() > 0;
}

template <typename scalar_t>
__device__ scalar_t cast_output(float value) {
  return static_cast<scalar_t>(value);
}

template <>
__device__ at::Half cast_output<at::Half>(float value) {
  return at::Half(__float2half_rn(value));
}

template <>
__device__ at::BFloat16 cast_output<at::BFloat16>(float value) {
  return at::BFloat16(__float2bfloat16_rn(value));
}

template <typename scalar_t>
__global__ void dequantize_epilogue_kernel(
    const int32_t* __restrict__ acc,
    const float* __restrict__ row_scale,
    const float* __restrict__ col_scale,
    const float* __restrict__ bias,
    scalar_t* __restrict__ out,
    int64_t total,
    int n,
    bool has_col_scale,
    bool has_bias) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < total; idx += stride) {
    int col = static_cast<int>(idx % n);
    int row = static_cast<int>(idx / n);
    float value = static_cast<float>(acc[idx]) * row_scale[row];
    if (has_col_scale) {
      value *= col_scale[col];
    }
    if (has_bias) {
      value += bias[col];
    }
    out[idx] = cast_output<scalar_t>(value);
  }
}

template <typename Gemm>
void launch_cutlass_gemm(torch::Tensor a, torch::Tensor b_t, torch::Tensor c, int m, int n, int k, cudaStream_t stream) {
  Gemm gemm_op;
  typename Gemm::Arguments args(
      {m, n, k},
      {reinterpret_cast<int8_t*>(a.data_ptr<int8_t>()), k},
      {reinterpret_cast<int8_t*>(b_t.data_ptr<int8_t>()), k},
      {reinterpret_cast<int32_t*>(c.data_ptr<int32_t>()), n},
      {reinterpret_cast<int32_t*>(c.data_ptr<int32_t>()), n},
      {1, 0});

  cutlass::Status status = gemm_op.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS int8 GEMM cannot implement this problem");
  status = gemm_op(args, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS int8 GEMM launch failed");
}

torch::Tensor run_cutlass_int8_mm(torch::Tensor a, torch::Tensor b_t) {
  check_int8_matrix(a, "a");
  check_int8_matrix(b_t, "b_t");
  TORCH_CHECK(a.device() == b_t.device(), "a and b_t must be on the same CUDA device");
  TORCH_CHECK(a.size(1) == b_t.size(1), "shape mismatch: expected a.size(1) == b_t.size(1)");
  TORCH_CHECK(a.size(1) % 32 == 0, "CUTLASS int8 GEMM requires K to be divisible by 32");

  const int64_t m64 = a.size(0);
  const int64_t k64 = a.size(1);
  const int64_t n64 = b_t.size(0);
  TORCH_CHECK(m64 <= std::numeric_limits<int>::max(), "M is too large for CUTLASS int GEMM");
  TORCH_CHECK(n64 <= std::numeric_limits<int>::max(), "N is too large for CUTLASS int GEMM");
  TORCH_CHECK(k64 <= std::numeric_limits<int>::max(), "K is too large for CUTLASS int GEMM");

  auto c = torch::empty({m64, n64}, a.options().dtype(torch::kInt32));
  if (m64 == 0 || n64 == 0 || k64 == 0) {
    return c;
  }

  const int m = static_cast<int>(m64);
  const int n = static_cast<int>(n64);
  const int k = static_cast<int>(k64);

  c10::cuda::CUDAGuard device_guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

  launch_cutlass_gemm<CutlassInt8Gemm>(a, b_t, c, m, n, k, stream.stream());
  return c;
}

}  // namespace ltx2_cutlass_int8

torch::Tensor cutlass_int8_mm(torch::Tensor a, torch::Tensor b_t) {
  return ltx2_cutlass_int8::run_cutlass_int8_mm(a, b_t);
}

torch::Tensor cutlass_int8_mm_scaled(
    torch::Tensor a,
    torch::Tensor b_t,
    torch::Tensor row_scale,
    c10::optional<torch::Tensor> col_scale,
    c10::optional<torch::Tensor> bias,
    int64_t output_dtype) {
  auto acc = ltx2_cutlass_int8::run_cutlass_int8_mm(a, b_t);
  const int64_t m = acc.size(0);
  const int64_t n = acc.size(1);

  ltx2_cutlass_int8::check_scale_vector(row_scale, "row_scale", m);
  bool use_col_scale = ltx2_cutlass_int8::has_tensor(col_scale);
  bool use_bias = ltx2_cutlass_int8::has_tensor(bias);
  if (use_col_scale) {
    ltx2_cutlass_int8::check_scale_vector(col_scale.value(), "col_scale", n);
    TORCH_CHECK(col_scale.value().device() == a.device(), "col_scale must be on the same CUDA device as a");
  }
  if (use_bias) {
    TORCH_CHECK(bias.value().is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias.value().numel() == n, "bias must have N elements");
    TORCH_CHECK(bias.value().is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias.value().device() == a.device(), "bias must be on the same CUDA device as a");
  }
  TORCH_CHECK(row_scale.device() == a.device(), "row_scale must be on the same CUDA device as a");

  auto out_options = a.options();
  if (output_dtype == 0) {
    out_options = out_options.dtype(torch::kFloat16);
  } else if (output_dtype == 1) {
    out_options = out_options.dtype(torch::kBFloat16);
  } else if (output_dtype == 2) {
    out_options = out_options.dtype(torch::kFloat32);
  } else {
    TORCH_CHECK(false, "Unsupported output dtype code for CUTLASS scaled int8 GEMM");
  }

  auto out = torch::empty({m, n}, out_options);
  if (m == 0 || n == 0) {
    return out;
  }

  c10::cuda::CUDAGuard device_guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
  const int threads = 256;
  const int64_t total = m * n;
  const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
  const float* col_scale_ptr = use_col_scale ? col_scale.value().data_ptr<float>() : nullptr;
  const float* bias_ptr = use_bias ? bias.value().data_ptr<float>() : nullptr;

  if (output_dtype == 0) {
    ltx2_cutlass_int8::dequantize_epilogue_kernel<at::Half><<<blocks, threads, 0, stream.stream()>>>(
        acc.data_ptr<int32_t>(),
        row_scale.data_ptr<float>(),
        col_scale_ptr,
        bias_ptr,
        out.data_ptr<at::Half>(),
        total,
        static_cast<int>(n),
        use_col_scale,
        use_bias);
  } else if (output_dtype == 1) {
    ltx2_cutlass_int8::dequantize_epilogue_kernel<at::BFloat16><<<blocks, threads, 0, stream.stream()>>>(
        acc.data_ptr<int32_t>(),
        row_scale.data_ptr<float>(),
        col_scale_ptr,
        bias_ptr,
        out.data_ptr<at::BFloat16>(),
        total,
        static_cast<int>(n),
        use_col_scale,
        use_bias);
  } else {
    ltx2_cutlass_int8::dequantize_epilogue_kernel<float><<<blocks, threads, 0, stream.stream()>>>(
        acc.data_ptr<int32_t>(),
        row_scale.data_ptr<float>(),
        col_scale_ptr,
        bias_ptr,
        out.data_ptr<float>(),
        total,
        static_cast<int>(n),
        use_col_scale,
        use_bias);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
