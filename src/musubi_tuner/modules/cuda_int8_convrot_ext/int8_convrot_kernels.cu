#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace ltx2_cuda_int8_convrot {

constexpr int kWarp = 32;

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

__device__ __forceinline__ float warp_reduce_max(float v) {
  for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
  }
  return v;
}

template <int NUM_WARPS>
__device__ __forceinline__ float block_reduce_max(float v, float* warp_smem, float* block_smem) {
  const int lane = threadIdx.x & (kWarp - 1);
  const int wid = threadIdx.x >> 5;
  v = warp_reduce_max(v);
  if (lane == 0) {
    warp_smem[wid] = v;
  }
  __syncthreads();
  if (wid == 0) {
    float total = lane < NUM_WARPS ? warp_smem[lane] : 0.0f;
    total = warp_reduce_max(total);
    if (lane == 0) {
      *block_smem = total;
    }
  }
  __syncthreads();
  return *block_smem;
}

__device__ __forceinline__ float h4_row_dot(int d, float x0, float x1, float x2, float x3) {
  switch (d) {
    case 0:
      return x0 + x1 + x2 - x3;
    case 1:
      return x0 + x1 - x2 + x3;
    case 2:
      return x0 - x1 + x2 + x3;
    default:
      return -x0 + x1 + x2 + x3;
  }
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

template <typename scalar_t, int BLOCK_THREADS, bool HAS_COL_SCALE>
__global__ void quantize_rowwise_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ col_scale,
    int8_t* __restrict__ q,
    float* __restrict__ scales,
    int K) {
  constexpr int kWarps = BLOCK_THREADS / kWarp;
  __shared__ float warp_smem[kWarps];
  __shared__ float block_smem;

  const int row = static_cast<int>(blockIdx.x);
  const int tid = threadIdx.x;
  const int64_t row_offset = static_cast<int64_t>(row) * K;

  float abs_max = 0.0f;
  for (int col = tid; col < K; col += BLOCK_THREADS) {
    float v = to_float(x[row_offset + col]);
    if constexpr (HAS_COL_SCALE) {
      v *= col_scale[col];
    }
    abs_max = fmaxf(abs_max, fabsf(v));
  }
  abs_max = block_reduce_max<kWarps>(abs_max, warp_smem, &block_smem);
  const float scale = fmaxf(abs_max * (1.0f / 127.0f), 1.0e-30f);
  if (tid == 0) {
    scales[row] = scale;
  }

  for (int col = tid; col < K; col += BLOCK_THREADS) {
    const int64_t idx = row_offset + col;
    float v = to_float(x[idx]);
    if constexpr (HAS_COL_SCALE) {
      v *= col_scale[col];
    }
    float quantized = nearbyintf(v / scale);
    quantized = fminf(127.0f, fmaxf(-127.0f, quantized));
    q[idx] = static_cast<int8_t>(quantized);
  }
}

template <typename output_t, typename bias_t, bool HAS_COL_SCALE, bool HAS_BIAS>
__global__ void dequant_epilogue_kernel(
    const int32_t* __restrict__ acc,
    const float* __restrict__ row_scale,
    const float* __restrict__ col_scale,
    const bias_t* __restrict__ bias,
    output_t* __restrict__ out,
    int64_t total,
    int N) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
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

template <typename output_t, bool HAS_COL_SCALE, bool HAS_BIAS>
__global__ void dequant_epilogue_vec4_kernel(
    const int32_t* __restrict__ acc,
    const float* __restrict__ row_scale,
    const float* __restrict__ col_scale,
    const float* __restrict__ bias,
    output_t* __restrict__ out,
    int64_t total_vec4,
    int N) {
  const int64_t idx4 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx4 >= total_vec4) {
    return;
  }
  const int64_t idx = idx4 * 4;
  const int col = static_cast<int>(idx % N);
  const int row = static_cast<int>(idx / N);
  const int4 acc4 = reinterpret_cast<const int4*>(acc)[idx4];
  const float xs = row_scale[row];
  float v0 = static_cast<float>(acc4.x) * xs;
  float v1 = static_cast<float>(acc4.y) * xs;
  float v2 = static_cast<float>(acc4.z) * xs;
  float v3 = static_cast<float>(acc4.w) * xs;
  if constexpr (HAS_COL_SCALE) {
    const float4 ws = reinterpret_cast<const float4*>(col_scale)[col / 4];
    v0 *= ws.x;
    v1 *= ws.y;
    v2 *= ws.z;
    v3 *= ws.w;
  }
  if constexpr (HAS_BIAS) {
    const float4 b = reinterpret_cast<const float4*>(bias)[col / 4];
    v0 += b.x;
    v1 += b.y;
    v2 += b.z;
    v3 += b.w;
  }
  out[idx] = from_float<output_t>(v0);
  out[idx + 1] = from_float<output_t>(v1);
  out[idx + 2] = from_float<output_t>(v2);
  out[idx + 3] = from_float<output_t>(v3);
}

template <typename output_t, int GROUP, int BLOCK_THREADS>
__global__ void dequant_epilogue_convrot_kernel(
    const int32_t* __restrict__ acc,
    const float* __restrict__ row_scale,
    output_t* __restrict__ out,
    int K) {
  static_assert(GROUP == 16 || GROUP == 64 || GROUP == 256, "unsupported ConvRot group size");
  static_assert(BLOCK_THREADS % GROUP == 0, "BLOCK_THREADS must be divisible by GROUP");
  constexpr int kGroupsInFlight = BLOCK_THREADS / GROUP;
  extern __shared__ float smem[];
  float* row_buf = smem;
  float* tmp = smem + K;

  const int row = static_cast<int>(blockIdx.x);
  const int tid = threadIdx.x;
  const int64_t row_offset = static_cast<int64_t>(row) * K;
  const float scale = row_scale[row];

  for (int col = tid; col < K; col += BLOCK_THREADS) {
    row_buf[col] = static_cast<float>(acc[row_offset + col]) * scale;
  }
  __syncthreads();

  const int sub = tid / GROUP;
  const int i = tid % GROUP;
  float* buf0 = tmp + sub * (2 * GROUP);
  float* buf1 = buf0 + GROUP;
  const int n_groups = K / GROUP;
  const int iters = (n_groups + kGroupsInFlight - 1) / kGroupsInFlight;

  for (int it = 0; it < iters; ++it) {
    const int g = it * kGroupsInFlight + sub;
    const bool active = g < n_groups;
    float* src = active ? (row_buf + g * GROUP) : buf0;
    float* dst = active ? buf0 : buf1;

    #pragma unroll
    for (int stage = 0; stage < 4; ++stage) {
      if ((stage == 0 && GROUP >= 4) ||
          (stage == 1 && GROUP >= 16) ||
          (stage == 2 && GROUP >= 64) ||
          (stage == 3 && GROUP >= 256)) {
        const int s = (stage == 0) ? 1 : (stage == 1) ? 4 : (stage == 2) ? 16 : 64;
        const int d = (i / s) & 3;
        const int base = i - d * s;
        dst[i] = 0.5f * h4_row_dot(d, src[base], src[base + s], src[base + 2 * s], src[base + 3 * s]);
        __syncthreads();
        float* t = src;
        src = dst;
        dst = t;
      }
    }
    if constexpr (GROUP == 64) {
      if (active) {
        row_buf[g * GROUP + i] = src[i];
      }
      __syncthreads();
    }
  }

  for (int col = tid; col < K; col += BLOCK_THREADS) {
    out[row_offset + col] = from_float<output_t>(row_buf[col]);
  }
}

template <typename scalar_t, int GROUP, int BLOCK_THREADS>
__global__ void quantize_rowwise_convrot_kernel(
    const scalar_t* __restrict__ x,
    int8_t* __restrict__ q,
    float* __restrict__ scales,
    int K) {
  static_assert(GROUP == 16 || GROUP == 64 || GROUP == 256, "unsupported ConvRot group size");
  static_assert(BLOCK_THREADS % GROUP == 0, "BLOCK_THREADS must be divisible by GROUP");
  constexpr int kGroupsInFlight = BLOCK_THREADS / GROUP;
  constexpr int kWarps = BLOCK_THREADS / kWarp;

  extern __shared__ float smem[];
  float* row_buf = smem;
  float* tmp = smem + K;

  __shared__ float warp_smem[kWarps];
  __shared__ float block_smem;

  const int row = static_cast<int>(blockIdx.x);
  const int tid = threadIdx.x;
  const int64_t row_offset = static_cast<int64_t>(row) * K;

  for (int col = tid; col < K; col += BLOCK_THREADS) {
    row_buf[col] = to_float(x[row_offset + col]);
  }
  __syncthreads();

  const int sub = tid / GROUP;
  const int i = tid % GROUP;
  float* buf0 = tmp + sub * (2 * GROUP);
  float* buf1 = buf0 + GROUP;
  const int n_groups = K / GROUP;
  const int iters = (n_groups + kGroupsInFlight - 1) / kGroupsInFlight;

  for (int it = 0; it < iters; ++it) {
    const int g = it * kGroupsInFlight + sub;
    const bool active = g < n_groups;
    float* src = active ? (row_buf + g * GROUP) : buf0;
    float* dst = active ? buf0 : buf1;

    #pragma unroll
    for (int stage = 0; stage < 4; ++stage) {
      if ((stage == 0 && GROUP >= 4) ||
          (stage == 1 && GROUP >= 16) ||
          (stage == 2 && GROUP >= 64) ||
          (stage == 3 && GROUP >= 256)) {
        const int s = (stage == 0) ? 1 : (stage == 1) ? 4 : (stage == 2) ? 16 : 64;
        const int d = (i / s) & 3;
        const int base = i - d * s;
        dst[i] = 0.5f * h4_row_dot(d, src[base], src[base + s], src[base + 2 * s], src[base + 3 * s]);
        __syncthreads();
        float* t = src;
        src = dst;
        dst = t;
      }
    }
    if constexpr (GROUP == 64) {
      if (active) {
        row_buf[g * GROUP + i] = src[i];
      }
      __syncthreads();
    }
  }

  float abs_max = 0.0f;
  for (int col = tid; col < K; col += BLOCK_THREADS) {
    abs_max = fmaxf(abs_max, fabsf(row_buf[col]));
  }
  abs_max = block_reduce_max<kWarps>(abs_max, warp_smem, &block_smem);
  const float scale = fmaxf(abs_max * (1.0f / 127.0f), 1.0e-30f);
  if (tid == 0) {
    scales[row] = scale;
  }

  for (int col = tid; col < K; col += BLOCK_THREADS) {
    const int64_t idx = row_offset + col;
    float quantized = nearbyintf(row_buf[col] / scale);
    quantized = fminf(127.0f, fmaxf(-127.0f, quantized));
    q[idx] = static_cast<int8_t>(quantized);
  }
}

void check_input(const torch::Tensor& x, int64_t group_size) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.dim() == 2, "x must be a rank-2 matrix");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(
      x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 || x.scalar_type() == torch::kBFloat16,
      "x must be float32, float16, or bfloat16");
  TORCH_CHECK(group_size == 16 || group_size == 64 || group_size == 256, "group_size must be 16, 64, or 256");
  TORCH_CHECK(x.size(1) % group_size == 0, "x feature dimension must be divisible by group_size");
  TORCH_CHECK(x.size(1) <= std::numeric_limits<int>::max(), "K is too large");
}

void check_float_matrix(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(x.dim() == 2, name, " must be a rank-2 matrix");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(
      x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 || x.scalar_type() == torch::kBFloat16,
      name, " must be float32, float16, or bfloat16");
}

void check_int32_matrix(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(x.dim() == 2, name, " must be a rank-2 matrix");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kInt32, name, " must be torch.int32");
}

void check_scale_vector(const torch::Tensor& x, const char* name, int64_t expected) {
  TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, name, " must be torch.float32");
  TORCH_CHECK(x.numel() == expected, name, " must have ", expected, " elements");
}

bool has_tensor(const c10::optional<torch::Tensor>& tensor) {
  return tensor.has_value() && tensor.value().defined() && tensor.value().numel() > 0;
}

template <typename scalar_t, int GROUP, int BLOCK_THREADS>
void launch_typed(torch::Tensor x, torch::Tensor q, torch::Tensor scales, int K, cudaStream_t stream) {
  using Kernel = void (*)(const scalar_t*, int8_t*, float*, int);
  Kernel kernel = quantize_rowwise_convrot_kernel<scalar_t, GROUP, BLOCK_THREADS>;
  constexpr int groups_in_flight = BLOCK_THREADS / GROUP;
  const size_t smem_bytes = (static_cast<size_t>(K) + groups_in_flight * 2 * GROUP) * sizeof(float);
  cudaError_t attr_err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
  TORCH_CHECK(
      attr_err == cudaSuccess,
      "CUDA ConvRot quantization shared-memory request failed: ",
      cudaGetErrorString(attr_err));
  kernel<<<static_cast<unsigned int>(x.size(0)), BLOCK_THREADS, smem_bytes, stream>>>(
      x.data_ptr<scalar_t>(),
      q.data_ptr<int8_t>(),
      scales.data_ptr<float>(),
      K);
}

template <typename scalar_t, int GROUP>
void launch_group(torch::Tensor x, torch::Tensor q, torch::Tensor scales, int K, cudaStream_t stream) {
  if constexpr (GROUP == 16) {
    if (K >= 4096) {
      launch_typed<scalar_t, GROUP, 512>(x, q, scales, K, stream);
    } else {
      launch_typed<scalar_t, GROUP, 256>(x, q, scales, K, stream);
    }
  } else if constexpr (GROUP == 64) {
    if (K >= 4096) {
      launch_typed<scalar_t, GROUP, 512>(x, q, scales, K, stream);
    } else {
      launch_typed<scalar_t, GROUP, 256>(x, q, scales, K, stream);
    }
  } else {
    if (K >= 4096) {
      launch_typed<scalar_t, GROUP, 1024>(x, q, scales, K, stream);
    } else if (K >= 1024) {
      launch_typed<scalar_t, GROUP, 512>(x, q, scales, K, stream);
    } else {
      launch_typed<scalar_t, GROUP, 256>(x, q, scales, K, stream);
    }
  }
}

template <typename scalar_t, bool HAS_COL_SCALE>
void launch_quantize_rowwise(torch::Tensor x, const float* col_scale, torch::Tensor q, torch::Tensor scales, int K, cudaStream_t stream) {
  if (K >= 4096) {
    quantize_rowwise_kernel<scalar_t, 512, HAS_COL_SCALE><<<static_cast<unsigned int>(x.size(0)), 512, 0, stream>>>(
        x.data_ptr<scalar_t>(), col_scale, q.data_ptr<int8_t>(), scales.data_ptr<float>(), K);
  } else {
    quantize_rowwise_kernel<scalar_t, 256, HAS_COL_SCALE><<<static_cast<unsigned int>(x.size(0)), 256, 0, stream>>>(
        x.data_ptr<scalar_t>(), col_scale, q.data_ptr<int8_t>(), scales.data_ptr<float>(), K);
  }
}

template <typename output_t, bool HAS_COL_SCALE>
void launch_dequant_no_bias(
    torch::Tensor acc,
    torch::Tensor row_scale,
    const float* col_scale,
    torch::Tensor out,
    int64_t total,
    int N,
    cudaStream_t stream) {
  constexpr int threads = 256;
  if ((total % 4) == 0 && (N % 4) == 0) {
    const int64_t total_vec4 = total / 4;
    const int blocks = static_cast<int>((total_vec4 + threads - 1) / threads);
    dequant_epilogue_vec4_kernel<output_t, HAS_COL_SCALE, false><<<blocks, threads, 0, stream>>>(
        acc.data_ptr<int32_t>(),
        row_scale.data_ptr<float>(),
        col_scale,
        nullptr,
        out.data_ptr<output_t>(),
        total_vec4,
        N);
  } else {
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    dequant_epilogue_kernel<output_t, float, HAS_COL_SCALE, false><<<blocks, threads, 0, stream>>>(
        acc.data_ptr<int32_t>(),
        row_scale.data_ptr<float>(),
        col_scale,
        nullptr,
        out.data_ptr<output_t>(),
        total,
        N);
  }
}

template <typename output_t, bool HAS_COL_SCALE>
void launch_dequant_bias_vec4(
    torch::Tensor acc,
    torch::Tensor row_scale,
    const float* col_scale,
    const float* bias,
    torch::Tensor out,
    int64_t total,
    int N,
    cudaStream_t stream) {
  constexpr int threads = 256;
  const int64_t total_vec4 = total / 4;
  const int blocks = static_cast<int>((total_vec4 + threads - 1) / threads);
  dequant_epilogue_vec4_kernel<output_t, HAS_COL_SCALE, true><<<blocks, threads, 0, stream>>>(
      acc.data_ptr<int32_t>(),
      row_scale.data_ptr<float>(),
      col_scale,
      bias,
      out.data_ptr<output_t>(),
      total_vec4,
      N);
}

template <typename output_t, typename bias_t, bool HAS_COL_SCALE>
void launch_dequant_bias(
    torch::Tensor acc,
    torch::Tensor row_scale,
    const float* col_scale,
    torch::Tensor bias,
    torch::Tensor out,
    int64_t total,
    int N,
    cudaStream_t stream) {
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  dequant_epilogue_kernel<output_t, bias_t, HAS_COL_SCALE, true><<<blocks, threads, 0, stream>>>(
      acc.data_ptr<int32_t>(),
      row_scale.data_ptr<float>(),
      col_scale,
      bias.data_ptr<bias_t>(),
      out.data_ptr<output_t>(),
      total,
      N);
}

template <typename output_t, int GROUP, int BLOCK_THREADS>
void launch_convrot_epilogue_typed(torch::Tensor acc, torch::Tensor row_scale, torch::Tensor out, int K, cudaStream_t stream) {
  using Kernel = void (*)(const int32_t*, const float*, output_t*, int);
  Kernel kernel = dequant_epilogue_convrot_kernel<output_t, GROUP, BLOCK_THREADS>;
  constexpr int groups_in_flight = BLOCK_THREADS / GROUP;
  const size_t smem_bytes = (static_cast<size_t>(K) + groups_in_flight * 2 * GROUP) * sizeof(float);
  cudaError_t attr_err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
  TORCH_CHECK(
      attr_err == cudaSuccess,
      "CUDA ConvRot epilogue shared-memory request failed: ",
      cudaGetErrorString(attr_err));
  kernel<<<static_cast<unsigned int>(acc.size(0)), BLOCK_THREADS, smem_bytes, stream>>>(
      acc.data_ptr<int32_t>(),
      row_scale.data_ptr<float>(),
      out.data_ptr<output_t>(),
      K);
}

template <typename output_t, int GROUP>
void launch_convrot_epilogue_group(torch::Tensor acc, torch::Tensor row_scale, torch::Tensor out, int K, cudaStream_t stream) {
  if constexpr (GROUP == 256) {
    if (K >= 4096) {
      launch_convrot_epilogue_typed<output_t, GROUP, 1024>(acc, row_scale, out, K, stream);
    } else if (K >= 1024) {
      launch_convrot_epilogue_typed<output_t, GROUP, 512>(acc, row_scale, out, K, stream);
    } else {
      launch_convrot_epilogue_typed<output_t, GROUP, 256>(acc, row_scale, out, K, stream);
    }
  } else if constexpr (GROUP == 64) {
    if (K >= 4096) {
      launch_convrot_epilogue_typed<output_t, GROUP, 512>(acc, row_scale, out, K, stream);
    } else {
      launch_convrot_epilogue_typed<output_t, GROUP, 256>(acc, row_scale, out, K, stream);
    }
  } else {
    if (K >= 4096) {
      launch_convrot_epilogue_typed<output_t, GROUP, 512>(acc, row_scale, out, K, stream);
    } else {
      launch_convrot_epilogue_typed<output_t, GROUP, 256>(acc, row_scale, out, K, stream);
    }
  }
}

}  // namespace ltx2_cuda_int8_convrot

std::tuple<torch::Tensor, torch::Tensor> quantize_rowwise_convrot(torch::Tensor x, int64_t group_size) {
  ltx2_cuda_int8_convrot::check_input(x, group_size);
  const auto M = x.size(0);
  const auto K64 = x.size(1);
  const int K = static_cast<int>(K64);
  auto q = torch::empty({M, K64}, x.options().dtype(torch::kInt8));
  auto scales = torch::empty({M, 1}, x.options().dtype(torch::kFloat32));
  if (M == 0 || K64 == 0) {
    return {q, scales};
  }

  c10::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "quantize_rowwise_convrot", [&] {
    if (group_size == 16) {
      ltx2_cuda_int8_convrot::launch_group<scalar_t, 16>(x, q, scales, K, stream.stream());
    } else if (group_size == 64) {
      ltx2_cuda_int8_convrot::launch_group<scalar_t, 64>(x, q, scales, K, stream.stream());
    } else {
      ltx2_cuda_int8_convrot::launch_group<scalar_t, 256>(x, q, scales, K, stream.stream());
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, scales};
}

std::tuple<torch::Tensor, torch::Tensor> quantize_rowwise(torch::Tensor x, c10::optional<torch::Tensor> col_scale) {
  ltx2_cuda_int8_convrot::check_float_matrix(x, "x");
  const auto M = x.size(0);
  const auto K64 = x.size(1);
  TORCH_CHECK(K64 <= std::numeric_limits<int>::max(), "K is too large");
  const int K = static_cast<int>(K64);
  const bool use_col_scale = ltx2_cuda_int8_convrot::has_tensor(col_scale);
  if (use_col_scale) {
    ltx2_cuda_int8_convrot::check_scale_vector(col_scale.value(), "col_scale", K64);
    TORCH_CHECK(col_scale.value().device() == x.device(), "col_scale must be on the same CUDA device as x");
  }

  auto q = torch::empty({M, K64}, x.options().dtype(torch::kInt8));
  auto scales = torch::empty({M, 1}, x.options().dtype(torch::kFloat32));
  if (M == 0 || K64 == 0) {
    return {q, scales};
  }

  c10::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  const float* col_scale_ptr = use_col_scale ? col_scale.value().data_ptr<float>() : nullptr;

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "quantize_rowwise", [&] {
    if (use_col_scale) {
      ltx2_cuda_int8_convrot::launch_quantize_rowwise<scalar_t, true>(x, col_scale_ptr, q, scales, K, stream.stream());
    } else {
      ltx2_cuda_int8_convrot::launch_quantize_rowwise<scalar_t, false>(x, col_scale_ptr, q, scales, K, stream.stream());
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, scales};
}

torch::Tensor dequant_epilogue(
    torch::Tensor out_int32,
    torch::Tensor row_scale,
    c10::optional<torch::Tensor> col_scale,
    c10::optional<torch::Tensor> bias,
    int64_t output_dtype) {
  ltx2_cuda_int8_convrot::check_int32_matrix(out_int32, "out_int32");
  const auto M = out_int32.size(0);
  const auto N64 = out_int32.size(1);
  TORCH_CHECK(N64 <= std::numeric_limits<int>::max(), "N is too large");
  const int N = static_cast<int>(N64);
  ltx2_cuda_int8_convrot::check_scale_vector(row_scale, "row_scale", M);
  TORCH_CHECK(row_scale.device() == out_int32.device(), "row_scale must be on the same CUDA device as out_int32");
  const bool use_col_scale = ltx2_cuda_int8_convrot::has_tensor(col_scale);
  const bool use_bias = ltx2_cuda_int8_convrot::has_tensor(bias);
  if (use_col_scale) {
    ltx2_cuda_int8_convrot::check_scale_vector(col_scale.value(), "col_scale", N64);
    TORCH_CHECK(col_scale.value().device() == out_int32.device(), "col_scale must be on the same CUDA device as out_int32");
  }
  if (use_bias) {
    TORCH_CHECK(bias.value().is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias.value().is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias.value().numel() == N64, "bias must have N elements");
    TORCH_CHECK(bias.value().device() == out_int32.device(), "bias must be on the same CUDA device as out_int32");
    TORCH_CHECK(
        bias.value().scalar_type() == torch::kFloat32 || bias.value().scalar_type() == torch::kFloat16 ||
            bias.value().scalar_type() == torch::kBFloat16,
        "bias must be float32, float16, or bfloat16");
  }

  c10::ScalarType dtype;
  if (output_dtype == 0) {
    dtype = torch::kFloat16;
  } else if (output_dtype == 1) {
    dtype = torch::kBFloat16;
  } else if (output_dtype == 2) {
    dtype = torch::kFloat32;
  } else {
    TORCH_CHECK(false, "Unsupported output dtype code");
  }
  auto out = torch::empty({M, N64}, out_int32.options().dtype(dtype));
  if (M == 0 || N64 == 0) {
    return out;
  }

  c10::cuda::CUDAGuard device_guard(out_int32.device());
  auto stream = at::cuda::getCurrentCUDAStream(out_int32.get_device());
  const int64_t total = M * N64;
  const float* col_scale_ptr = use_col_scale ? col_scale.value().data_ptr<float>() : nullptr;

  auto launch_for_output = [&](auto* out_typed) {
    using output_t = std::remove_pointer_t<decltype(out_typed)>;
    if (!use_bias) {
      if (use_col_scale) {
        ltx2_cuda_int8_convrot::launch_dequant_no_bias<output_t, true>(
            out_int32, row_scale, col_scale_ptr, out, total, N, stream.stream());
      } else {
        ltx2_cuda_int8_convrot::launch_dequant_no_bias<output_t, false>(
            out_int32, row_scale, col_scale_ptr, out, total, N, stream.stream());
      }
      return;
    }
    if (bias.value().scalar_type() == torch::kFloat32 && (total % 4) == 0 && (N % 4) == 0) {
      // fp32 bias + vec4-eligible: keep the vectorized epilogue instead of the scalar fallback.
      const float* bias_ptr = bias.value().data_ptr<float>();
      if (use_col_scale) {
        ltx2_cuda_int8_convrot::launch_dequant_bias_vec4<output_t, true>(
            out_int32, row_scale, col_scale_ptr, bias_ptr, out, total, N, stream.stream());
      } else {
        ltx2_cuda_int8_convrot::launch_dequant_bias_vec4<output_t, false>(
            out_int32, row_scale, col_scale_ptr, bias_ptr, out, total, N, stream.stream());
      }
      return;
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, bias.value().scalar_type(), "dequant_epilogue_bias", [&] {
      if (use_col_scale) {
        ltx2_cuda_int8_convrot::launch_dequant_bias<output_t, scalar_t, true>(
            out_int32, row_scale, col_scale_ptr, bias.value(), out, total, N, stream.stream());
      } else {
        ltx2_cuda_int8_convrot::launch_dequant_bias<output_t, scalar_t, false>(
            out_int32, row_scale, col_scale_ptr, bias.value(), out, total, N, stream.stream());
      }
    });
  };

  if (output_dtype == 0) {
    launch_for_output(out.data_ptr<at::Half>());
  } else if (output_dtype == 1) {
    launch_for_output(out.data_ptr<at::BFloat16>());
  } else {
    launch_for_output(out.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor dequant_epilogue_convrot(torch::Tensor out_int32, torch::Tensor row_scale, int64_t group_size, int64_t output_dtype) {
  ltx2_cuda_int8_convrot::check_int32_matrix(out_int32, "out_int32");
  const auto M = out_int32.size(0);
  const auto K64 = out_int32.size(1);
  TORCH_CHECK(K64 <= std::numeric_limits<int>::max(), "K is too large");
  const int K = static_cast<int>(K64);
  TORCH_CHECK(group_size == 16 || group_size == 64 || group_size == 256, "group_size must be 16, 64, or 256");
  TORCH_CHECK(K64 % group_size == 0, "K must be divisible by group_size");
  ltx2_cuda_int8_convrot::check_scale_vector(row_scale, "row_scale", M);
  TORCH_CHECK(row_scale.device() == out_int32.device(), "row_scale must be on the same CUDA device as out_int32");

  c10::ScalarType dtype;
  if (output_dtype == 0) {
    dtype = torch::kFloat16;
  } else if (output_dtype == 1) {
    dtype = torch::kBFloat16;
  } else if (output_dtype == 2) {
    dtype = torch::kFloat32;
  } else {
    TORCH_CHECK(false, "Unsupported output dtype code");
  }
  auto out = torch::empty({M, K64}, out_int32.options().dtype(dtype));
  if (M == 0 || K64 == 0) {
    return out;
  }

  c10::cuda::CUDAGuard device_guard(out_int32.device());
  auto stream = at::cuda::getCurrentCUDAStream(out_int32.get_device());
  auto launch_for_output = [&](auto* out_typed) {
    using output_t = std::remove_pointer_t<decltype(out_typed)>;
    if (group_size == 16) {
      ltx2_cuda_int8_convrot::launch_convrot_epilogue_group<output_t, 16>(out_int32, row_scale, out, K, stream.stream());
    } else if (group_size == 64) {
      ltx2_cuda_int8_convrot::launch_convrot_epilogue_group<output_t, 64>(out_int32, row_scale, out, K, stream.stream());
    } else {
      ltx2_cuda_int8_convrot::launch_convrot_epilogue_group<output_t, 256>(out_int32, row_scale, out, K, stream.stream());
    }
  };
  if (output_dtype == 0) {
    launch_for_output(out.data_ptr<at::Half>());
  } else if (output_dtype == 1) {
    launch_for_output(out.data_ptr<at::BFloat16>());
  } else {
    launch_for_output(out.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
