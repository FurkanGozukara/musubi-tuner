#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>
#include <type_traits>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

namespace ltx2_cutlass_int4 {

using ElementInput = cutlass::int4b_t;
using ElementOutput = int32_t;
using ElementAccumulator = int32_t;
using ElementCompute = int32_t;

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int WarpM, int WarpN, int Stages>
using CutlassInt4Gemm = cutlass::gemm::device::Gemm<
    ElementInput,
    cutlass::layout::RowMajor,
    ElementInput,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>,
    cutlass::gemm::GemmShape<WarpM, WarpN, ThreadblockK>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementCompute>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    Stages>;

using Gemm128x128 = CutlassInt4Gemm<128, 128, 128, 64, 64, 3>;
using Gemm128x256 = CutlassInt4Gemm<128, 256, 128, 64, 64, 3>;
using Gemm256x128 = CutlassInt4Gemm<256, 128, 128, 64, 64, 3>;
using Gemm128x64 = CutlassInt4Gemm<128, 64, 128, 64, 32, 4>;
using Gemm64x128 = CutlassInt4Gemm<64, 128, 128, 32, 64, 4>;
using Gemm64x256 = CutlassInt4Gemm<64, 256, 128, 32, 64, 4>;
using Gemm256x64 = CutlassInt4Gemm<256, 64, 128, 64, 32, 4>;
using Gemm64x64 = CutlassInt4Gemm<64, 64, 128, 32, 32, 6>;

template <typename T>
__device__ __forceinline__ float to_float(T v) {
  return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float to_float<at::Half>(at::Half v) {
  return __half2float(static_cast<half>(v));
}

template <>
__device__ __forceinline__ float to_float<at::BFloat16>(at::BFloat16 v) {
  return __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&v));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float v) {
  return static_cast<scalar_t>(v);
}

template <>
__device__ __forceinline__ at::Half from_float<at::Half>(float v) {
  return at::Half(__float2half_rn(v));
}

template <>
__device__ __forceinline__ at::BFloat16 from_float<at::BFloat16>(float v) {
  return at::BFloat16(__float2bfloat16_rn(v));
}

__device__ __forceinline__ int unpack_i4(uint8_t byte, int high) {
  const int nibble = high ? ((byte >> 4) & 0x0f) : (byte & 0x0f);
  return (nibble ^ 0x08) - 0x08;
}

__device__ __forceinline__ uint8_t pack_i4(int low, int high) {
  return static_cast<uint8_t>((low & 0x0f) | ((high & 0x0f) << 4));
}

__global__ void transpose_packed_i4_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int rows,
    int cols,
    int src_packed_cols,
    int dst_packed_cols,
    int64_t total_dst_bytes) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < total_dst_bytes; idx += stride) {
    const int dst_row = static_cast<int>(idx / dst_packed_cols);
    const int dst_byte_col = static_cast<int>(idx - static_cast<int64_t>(dst_row) * dst_packed_cols);
    const int src_row0 = dst_byte_col * 2;
    const int src_row1 = src_row0 + 1;
    const int src_byte_col = dst_row >> 1;
    const int src_high = dst_row & 1;
    const int low = src_row0 < rows ? unpack_i4(src[static_cast<int64_t>(src_row0) * src_packed_cols + src_byte_col], src_high) : 0;
    const int high = src_row1 < rows ? unpack_i4(src[static_cast<int64_t>(src_row1) * src_packed_cols + src_byte_col], src_high) : 0;
    dst[idx] = pack_i4(low, high);
  }
}

template <typename output_t, typename bias_t, bool HAS_COL_SCALE, bool HAS_BIAS>
__global__ void scaled_output_kernel(
    const int32_t* __restrict__ acc,
    const float* __restrict__ row_scale,
    const float* __restrict__ col_scale,
    const bias_t* __restrict__ bias,
    output_t* __restrict__ out,
    int64_t total,
    int N) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < total; idx += stride) {
    const int col = static_cast<int>(idx % N);
    const int row = static_cast<int>(idx / N);
    float v = static_cast<float>(acc[idx]) * row_scale[row];
    if constexpr (HAS_COL_SCALE) {
      v *= col_scale[col];
    }
    if constexpr (HAS_BIAS) {
      v += to_float(bias[col]);
    }
    out[idx] = from_float<output_t>(v);
  }
}

void check_uint8_matrix(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kUInt8, name, " must be torch.uint8");
  TORCH_CHECK(tensor.dim() == 2, name, " must be a rank-2 matrix");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous row-major");
}

void check_scale_vector(const torch::Tensor& tensor, const char* name, int64_t expected) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be torch.float32");
  TORCH_CHECK(tensor.dim() == 1, name, " must be a rank-1 vector");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.numel() == expected, name, " must have ", expected, " elements");
}

bool has_tensor(const c10::optional<torch::Tensor>& tensor) {
  return tensor.has_value() && tensor.value().defined() && tensor.value().numel() > 0;
}

c10::ScalarType dtype_from_code(int64_t code) {
  if (code == 0) {
    return torch::kFloat16;
  }
  if (code == 1) {
    return torch::kBFloat16;
  }
  if (code == 2) {
    return torch::kFloat32;
  }
  TORCH_CHECK(false, "Unsupported CUTLASS INT4 output dtype code: ", code);
}

template <typename Gemm>
void launch_cutlass_gemm(torch::Tensor a, torch::Tensor b_t, torch::Tensor c, int m, int n, int k, cudaStream_t stream) {
  Gemm gemm_op;
  typename Gemm::Arguments args(
      {m, n, k},
      {reinterpret_cast<ElementInput const*>(a.data_ptr<uint8_t>()), k},
      {reinterpret_cast<ElementInput const*>(b_t.data_ptr<uint8_t>()), k},
      {reinterpret_cast<ElementOutput*>(c.data_ptr<int32_t>()), n},
      {reinterpret_cast<ElementOutput*>(c.data_ptr<int32_t>()), n},
      {1, 0});

  cutlass::Status status = gemm_op.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS int4 GEMM cannot implement this problem");
  status = gemm_op(args, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS int4 GEMM launch failed");
}

void launch_best_gemm(torch::Tensor a, torch::Tensor b_t, torch::Tensor c, int m, int n, int k, cudaStream_t stream) {
  if (m <= 64 && n <= 64) {
    launch_cutlass_gemm<Gemm64x64>(a, b_t, c, m, n, k, stream);
  } else if (m <= 64 && n >= 192) {
    launch_cutlass_gemm<Gemm64x256>(a, b_t, c, m, n, k, stream);
  } else if (m <= 64) {
    launch_cutlass_gemm<Gemm64x128>(a, b_t, c, m, n, k, stream);
  } else if (n <= 64 && m >= 192) {
    launch_cutlass_gemm<Gemm256x64>(a, b_t, c, m, n, k, stream);
  } else if (n <= 64) {
    launch_cutlass_gemm<Gemm128x64>(a, b_t, c, m, n, k, stream);
  } else if (n >= 2 * m && n >= 256) {
    launch_cutlass_gemm<Gemm128x256>(a, b_t, c, m, n, k, stream);
  } else if (m >= 2 * n && m >= 256) {
    launch_cutlass_gemm<Gemm256x128>(a, b_t, c, m, n, k, stream);
  } else {
    launch_cutlass_gemm<Gemm128x128>(a, b_t, c, m, n, k, stream);
  }
}

void check_cutlass_int4_gemm_inputs(torch::Tensor a, torch::Tensor b_t, int64_t logical_k) {
  check_uint8_matrix(a, "a");
  check_uint8_matrix(b_t, "b_t");
  TORCH_CHECK(a.device() == b_t.device(), "a and b_t must be on the same CUDA device");
  TORCH_CHECK(logical_k >= 0, "logical K must be non-negative");
  TORCH_CHECK(logical_k <= std::numeric_limits<int>::max(), "K is too large for CUTLASS int4 GEMM");
  TORCH_CHECK((logical_k + 1) / 2 == a.size(1), "a packed width does not match logical K");
  TORCH_CHECK((logical_k + 1) / 2 == b_t.size(1), "b_t packed width does not match logical K");
  TORCH_CHECK(logical_k % 64 == 0, "CUTLASS int4 tensor-core GEMM requires K to be divisible by 64");
}

torch::Tensor run_cutlass_int4_mm(torch::Tensor a, torch::Tensor b_t, int64_t logical_k) {
  check_cutlass_int4_gemm_inputs(a, b_t, logical_k);

  const int64_t m64 = a.size(0);
  const int64_t n64 = b_t.size(0);
  TORCH_CHECK(m64 <= std::numeric_limits<int>::max(), "M is too large for CUTLASS int4 GEMM");
  TORCH_CHECK(n64 <= std::numeric_limits<int>::max(), "N is too large for CUTLASS int4 GEMM");

  auto c = torch::empty({m64, n64}, a.options().dtype(torch::kInt32));
  if (m64 == 0 || n64 == 0 || logical_k == 0) {
    return c;
  }

  c10::cuda::CUDAGuard device_guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
  launch_best_gemm(
      a,
      b_t,
      c,
      static_cast<int>(m64),
      static_cast<int>(n64),
      static_cast<int>(logical_k),
      stream.stream());
  return c;
}

template <typename output_t, typename bias_t, bool HAS_COL_SCALE, bool HAS_BIAS>
void launch_scaled_output(
    torch::Tensor acc,
    torch::Tensor row_scale,
    const float* col_scale,
    const bias_t* bias,
    torch::Tensor out,
    int64_t total,
    int N,
    cudaStream_t stream) {
  constexpr int threads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
  scaled_output_kernel<output_t, bias_t, HAS_COL_SCALE, HAS_BIAS><<<blocks, threads, 0, stream>>>(
      acc.data_ptr<int32_t>(),
      row_scale.data_ptr<float>(),
      col_scale,
      bias,
      out.data_ptr<output_t>(),
      total,
      N);
}

torch::Tensor apply_scaled_output(
    torch::Tensor acc,
    torch::Tensor row_scale,
    c10::optional<torch::Tensor> col_scale,
    c10::optional<torch::Tensor> bias,
    int64_t output_dtype,
    cudaStream_t stream) {
  const int64_t M = acc.size(0);
  const int64_t N = acc.size(1);
  check_scale_vector(row_scale, "row_scale", M);
  TORCH_CHECK(row_scale.device() == acc.device(), "row_scale must be on acc device");
  const bool use_col_scale = has_tensor(col_scale);
  const bool use_bias = has_tensor(bias);
  if (use_col_scale) {
    check_scale_vector(col_scale.value(), "col_scale", N);
    TORCH_CHECK(col_scale.value().device() == acc.device(), "col_scale must be on acc device");
  }
  if (use_bias) {
    TORCH_CHECK(bias.value().is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias.value().is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias.value().numel() == N, "bias must have N elements");
    TORCH_CHECK(bias.value().device() == acc.device(), "bias must be on acc device");
    TORCH_CHECK(
        bias.value().scalar_type() == torch::kFloat32 || bias.value().scalar_type() == torch::kFloat16 ||
            bias.value().scalar_type() == torch::kBFloat16,
        "bias must be float32, float16, or bfloat16");
  }

  auto out = torch::empty({M, N}, acc.options().dtype(dtype_from_code(output_dtype)));
  if (M == 0 || N == 0) {
    return out;
  }

  const int64_t total = M * N;
  const float* col_scale_ptr = use_col_scale ? col_scale.value().data_ptr<float>() : nullptr;
  auto launch_for_output = [&](auto* out_typed) {
    using output_t = std::remove_pointer_t<decltype(out_typed)>;
    if (!use_bias) {
      if (use_col_scale) {
        launch_scaled_output<output_t, float, true, false>(
            acc, row_scale, col_scale_ptr, nullptr, out, total, static_cast<int>(N), stream);
      } else {
        launch_scaled_output<output_t, float, false, false>(
            acc, row_scale, nullptr, nullptr, out, total, static_cast<int>(N), stream);
      }
      return;
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, bias.value().scalar_type(), "cutlass_int4_scaled_output_bias", [&] {
      if (use_col_scale) {
        launch_scaled_output<output_t, scalar_t, true, true>(
            acc,
            row_scale,
            col_scale_ptr,
            bias.value().data_ptr<scalar_t>(),
            out,
            total,
            static_cast<int>(N),
            stream);
      } else {
        launch_scaled_output<output_t, scalar_t, false, true>(
            acc,
            row_scale,
            nullptr,
            bias.value().data_ptr<scalar_t>(),
            out,
            total,
            static_cast<int>(N),
            stream);
      }
    });
  };

  if (out.scalar_type() == torch::kFloat16) {
    launch_for_output(static_cast<at::Half*>(nullptr));
  } else if (out.scalar_type() == torch::kBFloat16) {
    launch_for_output(static_cast<at::BFloat16*>(nullptr));
  } else if (out.scalar_type() == torch::kFloat32) {
    launch_for_output(static_cast<float*>(nullptr));
  } else {
    TORCH_CHECK(false, "unsupported output dtype");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor run_cutlass_int4_linear(
    torch::Tensor a,
    torch::Tensor b_t,
    torch::Tensor row_scale,
    c10::optional<torch::Tensor> col_scale,
    c10::optional<torch::Tensor> bias,
    int64_t output_dtype,
    int64_t logical_k) {
  auto acc = run_cutlass_int4_mm(a, b_t, logical_k);
  if (acc.size(0) == 0 || acc.size(1) == 0) {
    return torch::empty({acc.size(0), acc.size(1)}, acc.options().dtype(dtype_from_code(output_dtype)));
  }
  c10::cuda::CUDAGuard device_guard(acc.device());
  auto stream = at::cuda::getCurrentCUDAStream(acc.get_device());
  return apply_scaled_output(acc, row_scale, col_scale, bias, output_dtype, stream.stream());
}

torch::Tensor run_transpose_packed_int4(torch::Tensor packed, int64_t logical_cols) {
  check_uint8_matrix(packed, "packed");
  TORCH_CHECK(logical_cols >= 0, "logical_cols must be non-negative");
  TORCH_CHECK(logical_cols <= std::numeric_limits<int>::max(), "logical_cols is too large");
  TORCH_CHECK((logical_cols + 1) / 2 <= packed.size(1), "logical_cols exceeds packed tensor width");

  const int64_t rows64 = packed.size(0);
  const int64_t dst_packed_cols64 = (rows64 + 1) / 2;
  TORCH_CHECK(rows64 <= std::numeric_limits<int>::max(), "row count is too large");
  TORCH_CHECK(dst_packed_cols64 <= std::numeric_limits<int>::max(), "transposed packed width is too large");
  auto out = torch::empty({logical_cols, dst_packed_cols64}, packed.options());
  if (rows64 == 0 || logical_cols == 0) {
    return out;
  }

  c10::cuda::CUDAGuard device_guard(packed.device());
  auto stream = at::cuda::getCurrentCUDAStream(packed.get_device());
  constexpr int threads = 256;
  const int64_t total = logical_cols * dst_packed_cols64;
  const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
  transpose_packed_i4_kernel<<<blocks, threads, 0, stream.stream()>>>(
      packed.data_ptr<uint8_t>(),
      out.data_ptr<uint8_t>(),
      static_cast<int>(rows64),
      static_cast<int>(logical_cols),
      static_cast<int>(packed.size(1)),
      static_cast<int>(dst_packed_cols64),
      total);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor run_cutlass_int4_linear_backward_input(
    torch::Tensor qg,
    torch::Tensor packed_weight,
    torch::Tensor row_scale,
    int64_t output_dtype,
    int64_t out_features,
    int64_t padded_features) {
  auto weight_t = run_transpose_packed_int4(packed_weight, padded_features);
  return run_cutlass_int4_linear(qg, weight_t, row_scale, c10::nullopt, c10::nullopt, output_dtype, out_features);
}

}  // namespace ltx2_cutlass_int4

torch::Tensor cutlass_int4_mm(torch::Tensor a, torch::Tensor b_t, int64_t k) {
  return ltx2_cutlass_int4::run_cutlass_int4_mm(a, b_t, k);
}

torch::Tensor cutlass_int4_linear(
    torch::Tensor a,
    torch::Tensor b_t,
    torch::Tensor row_scale,
    c10::optional<torch::Tensor> col_scale,
    c10::optional<torch::Tensor> bias,
    int64_t output_dtype,
    int64_t k) {
  return ltx2_cutlass_int4::run_cutlass_int4_linear(a, b_t, row_scale, col_scale, bias, output_dtype, k);
}

torch::Tensor cutlass_int4_linear_backward_input(
    torch::Tensor qg,
    torch::Tensor packed_weight,
    torch::Tensor row_scale,
    int64_t output_dtype,
    int64_t out_features,
    int64_t padded_features) {
  return ltx2_cutlass_int4::run_cutlass_int4_linear_backward_input(
      qg, packed_weight, row_scale, output_dtype, out_features, padded_features);
}

torch::Tensor transpose_packed_int4(torch::Tensor packed, int64_t logical_cols) {
  return ltx2_cutlass_int4::run_transpose_packed_int4(packed, logical_cols);
}
