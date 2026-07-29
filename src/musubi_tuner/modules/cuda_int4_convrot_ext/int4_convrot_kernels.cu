#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace ltx2_cuda_int4_convrot {

constexpr int kWarp = 32;
constexpr int kWmmaTileM = 8;
constexpr int kWmmaTileN = 8;
constexpr int kWmmaTileK = 32;

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

__device__ __forceinline__ int unpack_i4(uint8_t byte, int high) {
  const int nibble = high ? ((byte >> 4) & 0x0f) : (byte & 0x0f);
  return (nibble ^ 0x08) - 0x08;
}

__device__ __forceinline__ uint8_t pack_i4(int low, int high) {
  return static_cast<uint8_t>((low & 0x0f) | ((high & 0x0f) << 4));
}

__device__ __forceinline__ int quant_i4(float v, float scale) {
  float q = nearbyintf(v / scale);
  q = fminf(7.0f, fmaxf(-7.0f, q));
  return static_cast<int>(q);
}

template <typename scalar_t, int BLOCK_THREADS>
__global__ void quantize_rowwise_kernel(
    const scalar_t* __restrict__ x,
    uint8_t* __restrict__ q,
    float* __restrict__ scales,
    int K,
    int packed_K) {
  constexpr int kWarps = BLOCK_THREADS / kWarp;
  __shared__ float warp_smem[kWarps];
  __shared__ float block_smem;

  const int row = static_cast<int>(blockIdx.x);
  const int tid = threadIdx.x;
  const int64_t row_offset = static_cast<int64_t>(row) * K;
  const int64_t packed_offset = static_cast<int64_t>(row) * packed_K;

  float abs_max = 0.0f;
  for (int col = tid; col < K; col += BLOCK_THREADS) {
    abs_max = fmaxf(abs_max, fabsf(to_float(x[row_offset + col])));
  }
  abs_max = block_reduce_max<kWarps>(abs_max, warp_smem, &block_smem);
  const float scale = fmaxf(abs_max * (1.0f / 7.0f), 1.0e-30f);
  if (tid == 0) {
    scales[row] = scale;
  }

  for (int byte_idx = tid; byte_idx < packed_K; byte_idx += BLOCK_THREADS) {
    const int col0 = byte_idx * 2;
    const int low = col0 < K ? quant_i4(to_float(x[row_offset + col0]), scale) : 0;
    const int high = (col0 + 1) < K ? quant_i4(to_float(x[row_offset + col0 + 1]), scale) : 0;
    q[packed_offset + byte_idx] = pack_i4(low, high);
  }
}

// Fused ConvRot Hadamard rotation + row-wise INT4 quantization in one pass.
// The regular group-wise Hadamard is block-diagonal over ``group`` contiguous
// channels; we apply it in-place in shared memory with a staged radix-4 butterfly
// (the same combine as the Triton _convrot_hadamard4 kernel, and algebraically equal
// to build_hadamard(group)), then take the row absmax scale and pack signed int4.
// One block per row; dynamic shared memory holds the padded row as float.
template <typename scalar_t, int BLOCK_THREADS>
__global__ void quantize_rowwise_convrot_kernel(
    const scalar_t* __restrict__ x,
    uint8_t* __restrict__ q,
    float* __restrict__ scales,
    int K,
    int padded_K,
    int packed_K,
    int group) {
  extern __shared__ float srow[];
  constexpr int kWarps = BLOCK_THREADS / kWarp;
  __shared__ float warp_smem[kWarps];
  __shared__ float block_smem;

  const int row = static_cast<int>(blockIdx.x);
  const int tid = threadIdx.x;
  const int64_t row_offset = static_cast<int64_t>(row) * K;
  const int64_t packed_offset = static_cast<int64_t>(row) * packed_K;

  for (int col = tid; col < padded_K; col += BLOCK_THREADS) {
    srow[col] = col < K ? to_float(x[row_offset + col]) : 0.0f;
  }
  __syncthreads();

  const int quads = padded_K / 4;
  for (int step = 1; step < group; step *= 4) {
    const int quads_per_group = group / 4;
    const int subblocks = group / (4 * step);  // sub-blocks per group
    for (int qd = tid; qd < quads; qd += BLOCK_THREADS) {
      const int gi = qd / quads_per_group;         // group index
      const int wq = qd - gi * quads_per_group;    // quad within group
      const int sub = wq / step;                   // sub-block within group
      const int s = wq - sub * step;               // lane within sub-block
      const int base = gi * group + sub * (4 * step) + s;
      const int i0 = base;
      const int i1 = base + step;
      const int i2 = base + 2 * step;
      const int i3 = base + 3 * step;
      const float a = srow[i0];
      const float b = srow[i1];
      const float c = srow[i2];
      const float d = srow[i3];
      srow[i0] = a + b + c - d;
      srow[i1] = a + b - c + d;
      srow[i2] = a - b + c + d;
      srow[i3] = -a + b + c + d;
      (void)subblocks;
    }
    __syncthreads();
  }

  const float inv_norm = rsqrtf(static_cast<float>(group));
  float abs_max = 0.0f;
  for (int col = tid; col < padded_K; col += BLOCK_THREADS) {
    srow[col] *= inv_norm;
    abs_max = fmaxf(abs_max, fabsf(srow[col]));
  }
  abs_max = block_reduce_max<kWarps>(abs_max, warp_smem, &block_smem);
  const float scale = fmaxf(abs_max * (1.0f / 7.0f), 1.0e-30f);
  if (tid == 0) {
    scales[row] = scale;
  }

  for (int byte_idx = tid; byte_idx < packed_K; byte_idx += BLOCK_THREADS) {
    const int col0 = byte_idx * 2;
    const int low = col0 < padded_K ? quant_i4(srow[col0], scale) : 0;
    const int high = (col0 + 1) < padded_K ? quant_i4(srow[col0 + 1], scale) : 0;
    q[packed_offset + byte_idx] = pack_i4(low, high);
  }
}

template <typename output_t, typename bias_t, bool HAS_BIAS>
__global__ void linear_forward_kernel(
    const uint8_t* __restrict__ qx,
    const uint8_t* __restrict__ qw,
    const float* __restrict__ row_scale,
    const float* __restrict__ col_scale,
    const bias_t* __restrict__ bias,
    output_t* __restrict__ out,
    int M,
    int N,
    int K,
    int packed_K) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) {
    return;
  }

  int acc = 0;
  const int64_t x_offset = static_cast<int64_t>(row) * packed_K;
  const int64_t w_offset = static_cast<int64_t>(col) * packed_K;
  for (int k = 0; k < K; ++k) {
    const int packed_idx = k >> 1;
    const int high = k & 1;
    const int xv = unpack_i4(qx[x_offset + packed_idx], high);
    const int wv = unpack_i4(qw[w_offset + packed_idx], high);
    acc += xv * wv;
  }

  float v = static_cast<float>(acc) * row_scale[row] * col_scale[col];
  if constexpr (HAS_BIAS) {
    v += to_float(bias[col]);
  }
  out[static_cast<int64_t>(row) * N + col] = from_float<output_t>(v);
}

__global__ void unpack_to_int8_kernel(
    const uint8_t* __restrict__ packed,
    int8_t* __restrict__ out,
    int64_t total,
    int num_values,
    int packed_values) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < total; idx += stride) {
    const int col = static_cast<int>(idx % num_values);
    const int row = static_cast<int>(idx / num_values);
    const uint8_t byte = packed[static_cast<int64_t>(row) * packed_values + (col >> 1)];
    out[idx] = static_cast<int8_t>(unpack_i4(byte, col & 1));
  }
}

template <typename ratio_t>
__device__ __forceinline__ float group_ratio_to_float(ratio_t value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float group_ratio_to_float<int16_t>(int16_t value) {
  return static_cast<float>(value) * (1.0f / 256.0f);
}

template <typename ratio_t>
__global__ void unpack_group_scaled_to_int8_kernel(
    const uint8_t* __restrict__ packed,
    const ratio_t* __restrict__ ratio,
    int8_t* __restrict__ out,
    int64_t total_words,
    int num_values,
    int packed_values,
    int group_size,
    int groups_per_row) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  const int words_per_row = num_values / 8;
  for (; idx < total_words; idx += stride) {
    const int word_col = static_cast<int>(idx % words_per_row);
    const int row = static_cast<int>(idx / words_per_row);
    const int col = word_col * 8;
    const auto* packed_word_ptr =
        reinterpret_cast<const uint32_t*>(packed + static_cast<int64_t>(row) * packed_values + word_col * 4);
    const uint32_t word = *packed_word_ptr;
    const float group_ratio =
        group_ratio_to_float(ratio[static_cast<int64_t>(row) * groups_per_row + col / group_size]);
    uint64_t mapped_word = 0;
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
      const int nibble = static_cast<int>((word >> (lane * 4)) & 0x0f);
      const int code = (nibble ^ 0x08) - 0x08;
      const int mapped = __float2int_rn(static_cast<float>(code) * group_ratio);
      const int clipped = mapped < -127 ? -127 : (mapped > 127 ? 127 : mapped);
      mapped_word |= static_cast<uint64_t>(static_cast<uint8_t>(clipped)) << (lane * 8);
    }
    reinterpret_cast<uint64_t*>(out)[idx] = mapped_word;
  }
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
    const int low =
        src_row0 < rows ? unpack_i4(src[static_cast<int64_t>(src_row0) * src_packed_cols + src_byte_col], src_high) : 0;
    const int high =
        src_row1 < rows ? unpack_i4(src[static_cast<int64_t>(src_row1) * src_packed_cols + src_byte_col], src_high) : 0;
    dst[idx] = pack_i4(low, high);
  }
}

__global__ void wmma_int4_mm_kernel(
    const uint8_t* __restrict__ a,
    const uint8_t* __restrict__ b_t,
    int32_t* __restrict__ c,
    int M,
    int N,
    int K) {
  const int tile_m = blockIdx.y * kWmmaTileM;
  const int tile_n = blockIdx.x * kWmmaTileN;

  using namespace nvcuda;
  using int4_t = wmma::experimental::precision::s4;
  wmma::fragment<wmma::matrix_a, kWmmaTileM, kWmmaTileN, kWmmaTileK, int4_t, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, kWmmaTileM, kWmmaTileN, kWmmaTileK, int4_t, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, kWmmaTileM, kWmmaTileN, kWmmaTileK, int32_t> c_frag;
  wmma::fill_fragment(c_frag, 0);

  const int* a_words = reinterpret_cast<const int*>(a);
  const int* b_words = reinterpret_cast<const int*>(b_t);
  for (int k0 = 0; k0 < K; k0 += kWmmaTileK) {
    const int a_word_offset = (tile_m * K + k0) / 8;
    const int b_word_offset = (tile_n * K + k0) / 8;
    wmma::load_matrix_sync(a_frag, a_words + a_word_offset, K);
    wmma::load_matrix_sync(b_frag, b_words + b_word_offset, K);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  wmma::store_matrix_sync(c + static_cast<int64_t>(tile_m) * N + tile_n, c_frag, N, wmma::mem_row_major);
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

template <typename output_t>
__global__ void linear_backward_input_kernel(
    const uint8_t* __restrict__ qg,
    const uint8_t* __restrict__ qw,
    const float* __restrict__ row_scale,
    output_t* __restrict__ out,
    int M,
    int N,
    int K,
    int grad_packed_N,
    int weight_packed_K) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= K) {
    return;
  }

  int acc = 0;
  const int64_t g_offset = static_cast<int64_t>(row) * grad_packed_N;
  for (int n = 0; n < N; ++n) {
    const int gv = unpack_i4(qg[g_offset + (n >> 1)], n & 1);
    const int wv = unpack_i4(qw[static_cast<int64_t>(n) * weight_packed_K + (col >> 1)], col & 1);
    acc += gv * wv;
  }

  const float v = static_cast<float>(acc) * row_scale[row];
  out[static_cast<int64_t>(row) * K + col] = from_float<output_t>(v);
}

void check_float_matrix(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(x.dim() == 2, name, " must be a rank-2 matrix");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(
      x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 || x.scalar_type() == torch::kBFloat16,
      name, " must be float32, float16, or bfloat16");
}

void check_uint8_matrix(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(x.dim() == 2, name, " must be a rank-2 matrix");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kUInt8, name, " must be torch.uint8");
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

void check_int32_matrix(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(x.dim() == 2, name, " must be a rank-2 matrix");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kInt32, name, " must be torch.int32");
}

void check_wmma_int4_gemm_inputs(torch::Tensor a, torch::Tensor b_t, int64_t logical_k) {
  check_uint8_matrix(a, "a");
  check_uint8_matrix(b_t, "b_t");
  TORCH_CHECK(a.device() == b_t.device(), "a and b_t must be on the same CUDA device");
  TORCH_CHECK(logical_k >= 0, "logical K must be non-negative");
  TORCH_CHECK(logical_k <= std::numeric_limits<int>::max(), "K is too large for WMMA int4 GEMM");
  TORCH_CHECK((logical_k + 1) / 2 == a.size(1), "a packed width does not match logical K");
  TORCH_CHECK((logical_k + 1) / 2 == b_t.size(1), "b_t packed width does not match logical K");
  TORCH_CHECK(logical_k % kWmmaTileK == 0, "WMMA int4 GEMM requires K to be divisible by 32");
  TORCH_CHECK(a.size(0) % kWmmaTileM == 0, "WMMA int4 GEMM requires M to be divisible by 8");
  TORCH_CHECK(b_t.size(0) % kWmmaTileN == 0, "WMMA int4 GEMM requires N to be divisible by 8");
}


c10::ScalarType dtype_from_code(int64_t output_dtype) {
  if (output_dtype == 0) {
    return torch::kFloat16;
  }
  if (output_dtype == 1) {
    return torch::kBFloat16;
  }
  if (output_dtype == 2) {
    return torch::kFloat32;
  }
  TORCH_CHECK(false, "Unsupported output dtype code");
}

template <typename output_t, typename bias_t, bool HAS_COL_SCALE, bool HAS_BIAS>
void launch_dequant_epilogue(
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
  dequant_epilogue_kernel<output_t, bias_t, HAS_COL_SCALE, HAS_BIAS><<<blocks, threads, 0, stream>>>(
      acc.data_ptr<int32_t>(),
      row_scale.data_ptr<float>(),
      col_scale,
      bias,
      out.data_ptr<output_t>(),
      total,
      N);
}

template <typename scalar_t>
void launch_quantize(torch::Tensor x, torch::Tensor q, torch::Tensor scales, int K, int packed_K, cudaStream_t stream) {
  if (K >= 4096) {
    quantize_rowwise_kernel<scalar_t, 512><<<static_cast<unsigned int>(x.size(0)), 512, 0, stream>>>(
        x.data_ptr<scalar_t>(), q.data_ptr<uint8_t>(), scales.data_ptr<float>(), K, packed_K);
  } else {
    quantize_rowwise_kernel<scalar_t, 256><<<static_cast<unsigned int>(x.size(0)), 256, 0, stream>>>(
        x.data_ptr<scalar_t>(), q.data_ptr<uint8_t>(), scales.data_ptr<float>(), K, packed_K);
  }
}

template <typename output_t, typename bias_t, bool HAS_BIAS>
void launch_forward(
    torch::Tensor qx,
    torch::Tensor qw,
    torch::Tensor row_scale,
    torch::Tensor col_scale,
    const bias_t* bias,
    torch::Tensor out,
    int M,
    int N,
    int K,
    int packed_K,
    cudaStream_t stream) {
  const dim3 threads(16, 16);
  const dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
  linear_forward_kernel<output_t, bias_t, HAS_BIAS><<<blocks, threads, 0, stream>>>(
      qx.data_ptr<uint8_t>(),
      qw.data_ptr<uint8_t>(),
      row_scale.data_ptr<float>(),
      col_scale.data_ptr<float>(),
      bias,
      out.data_ptr<output_t>(),
      M,
      N,
      K,
      packed_K);
}

template <typename output_t>
void launch_backward_input(
    torch::Tensor qg,
    torch::Tensor qw,
    torch::Tensor row_scale,
    torch::Tensor out,
    int M,
    int N,
    int K,
    int grad_packed_N,
    int weight_packed_K,
    cudaStream_t stream) {
  const dim3 threads(16, 16);
  const dim3 blocks((K + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
  linear_backward_input_kernel<output_t><<<blocks, threads, 0, stream>>>(
      qg.data_ptr<uint8_t>(),
      qw.data_ptr<uint8_t>(),
      row_scale.data_ptr<float>(),
      out.data_ptr<output_t>(),
      M,
      N,
      K,
      grad_packed_N,
      weight_packed_K);
}

}  // namespace ltx2_cuda_int4_convrot

std::tuple<torch::Tensor, torch::Tensor> quantize_rowwise(torch::Tensor x) {
  ltx2_cuda_int4_convrot::check_float_matrix(x, "x");
  const auto M = x.size(0);
  const auto K64 = x.size(1);
  TORCH_CHECK(K64 <= std::numeric_limits<int>::max(), "K is too large");
  const int K = static_cast<int>(K64);
  const int64_t packed_K64 = (K64 + 1) / 2;
  const int packed_K = static_cast<int>(packed_K64);
  auto q = torch::empty({M, packed_K64}, x.options().dtype(torch::kUInt8));
  auto scales = torch::empty({M}, x.options().dtype(torch::kFloat32));
  if (M == 0 || K64 == 0) {
    return {q, scales};
  }

  c10::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "int4_quantize_rowwise", [&] {
    ltx2_cuda_int4_convrot::launch_quantize<scalar_t>(x, q, scales, K, packed_K, stream.stream());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, scales};
}

std::tuple<torch::Tensor, torch::Tensor> quantize_rowwise_convrot(torch::Tensor x, int64_t padded_features, int64_t group) {
  ltx2_cuda_int4_convrot::check_float_matrix(x, "x");
  const auto M = x.size(0);
  const auto K64 = x.size(1);
  TORCH_CHECK(padded_features >= K64, "padded_features must be >= K");
  TORCH_CHECK(padded_features <= std::numeric_limits<int>::max(), "padded_features too large");
  TORCH_CHECK(group >= 4 && (padded_features % group == 0), "group must be >= 4 and divide padded_features");
  // group must be a power of four (the radix-4 butterfly assumes it).
  for (int64_t g = group; g > 1; g /= 4) {
    TORCH_CHECK(g % 4 == 0, "ConvRot group size must be a power of four");
  }
  const int K = static_cast<int>(K64);
  const int padded_K = static_cast<int>(padded_features);
  const int packed_K = (padded_K + 1) / 2;
  auto q = torch::empty({M, (padded_K + 1) / 2}, x.options().dtype(torch::kUInt8));
  auto scales = torch::empty({M}, x.options().dtype(torch::kFloat32));
  if (M == 0 || padded_K == 0) {
    return {q, scales};
  }

  const size_t smem_bytes = static_cast<size_t>(padded_K) * sizeof(float);
  int max_smem = 0;
  cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, x.get_device());
  TORCH_CHECK(smem_bytes <= static_cast<size_t>(max_smem),
              "fused ConvRot quant needs ", smem_bytes, " bytes of shared memory but device allows ", max_smem);

  c10::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  constexpr int kThreads = 512;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "int4_quantize_rowwise_convrot", [&] {
    auto kernel = ltx2_cuda_int4_convrot::quantize_rowwise_convrot_kernel<scalar_t, kThreads>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
    kernel<<<static_cast<unsigned int>(M), kThreads, smem_bytes, stream.stream()>>>(
        x.data_ptr<scalar_t>(), q.data_ptr<uint8_t>(), scales.data_ptr<float>(), K, padded_K, packed_K, static_cast<int>(group));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, scales};
}

torch::Tensor unpack_to_int8(torch::Tensor packed, int64_t num_values) {
  ltx2_cuda_int4_convrot::check_uint8_matrix(packed, "packed");
  TORCH_CHECK(num_values >= 0, "num_values must be non-negative");
  TORCH_CHECK((num_values + 1) / 2 <= packed.size(1), "num_values exceeds packed tensor width");
  TORCH_CHECK(num_values <= std::numeric_limits<int>::max(), "num_values is too large");
  const auto M = packed.size(0);
  auto out = torch::empty({M, num_values}, packed.options().dtype(torch::kInt8));
  if (M == 0 || num_values == 0) {
    return out;
  }

  c10::cuda::CUDAGuard device_guard(packed.device());
  auto stream = at::cuda::getCurrentCUDAStream(packed.get_device());
  constexpr int threads = 256;
  const int64_t total = M * num_values;
  const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
  ltx2_cuda_int4_convrot::unpack_to_int8_kernel<<<blocks, threads, 0, stream.stream()>>>(
      packed.data_ptr<uint8_t>(),
      out.data_ptr<int8_t>(),
      total,
      static_cast<int>(num_values),
      static_cast<int>(packed.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor unpack_group_scaled_to_int8(
    torch::Tensor packed,
    torch::Tensor ratio,
    int64_t group_size,
    int64_t num_values) {
  ltx2_cuda_int4_convrot::check_uint8_matrix(packed, "packed");
  TORCH_CHECK(ratio.is_cuda(), "ratio must be a CUDA tensor");
  TORCH_CHECK(
      ratio.scalar_type() == torch::kFloat32 || ratio.scalar_type() == torch::kInt16,
      "ratio must be float32 or int16 Q8.8");
  TORCH_CHECK(ratio.dim() == 2, "ratio must be a 2D tensor");
  TORCH_CHECK(ratio.is_contiguous(), "ratio must be contiguous");
  TORCH_CHECK(ratio.device() == packed.device(), "ratio and packed must be on the same CUDA device");
  TORCH_CHECK(num_values >= 0, "num_values must be non-negative");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(num_values % group_size == 0, "num_values must be divisible by group_size");
  TORCH_CHECK(num_values % 8 == 0, "num_values must be divisible by 8");
  TORCH_CHECK(group_size % 8 == 0, "group_size must be divisible by 8");
  TORCH_CHECK((num_values + 1) / 2 <= packed.size(1), "num_values exceeds packed tensor width");
  TORCH_CHECK(num_values <= std::numeric_limits<int>::max(), "num_values is too large");
  TORCH_CHECK(group_size <= std::numeric_limits<int>::max(), "group_size is too large");
  const auto M = packed.size(0);
  const auto groups_per_row = num_values / group_size;
  TORCH_CHECK(ratio.size(0) == M && ratio.size(1) == groups_per_row, "ratio shape does not match packed rows/groups");
  auto out = torch::empty({M, num_values}, packed.options().dtype(torch::kInt8));
  if (M == 0 || num_values == 0) {
    return out;
  }

  c10::cuda::CUDAGuard device_guard(packed.device());
  auto stream = at::cuda::getCurrentCUDAStream(packed.get_device());
  constexpr int threads = 256;
  const int64_t total_words = M * (num_values / 8);
  const int blocks = static_cast<int>(std::min<int64_t>((total_words + threads - 1) / threads, 65535));
  if (ratio.scalar_type() == torch::kInt16) {
    ltx2_cuda_int4_convrot::unpack_group_scaled_to_int8_kernel<int16_t><<<blocks, threads, 0, stream.stream()>>>(
        packed.data_ptr<uint8_t>(),
        ratio.data_ptr<int16_t>(),
        out.data_ptr<int8_t>(),
        total_words,
        static_cast<int>(num_values),
        static_cast<int>(packed.size(1)),
        static_cast<int>(group_size),
        static_cast<int>(groups_per_row));
  } else {
    ltx2_cuda_int4_convrot::unpack_group_scaled_to_int8_kernel<float><<<blocks, threads, 0, stream.stream()>>>(
        packed.data_ptr<uint8_t>(),
        ratio.data_ptr<float>(),
        out.data_ptr<int8_t>(),
        total_words,
        static_cast<int>(num_values),
        static_cast<int>(packed.size(1)),
        static_cast<int>(group_size),
        static_cast<int>(groups_per_row));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor transpose_packed(torch::Tensor packed, int64_t logical_cols) {
  ltx2_cuda_int4_convrot::check_uint8_matrix(packed, "packed");
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
  ltx2_cuda_int4_convrot::transpose_packed_i4_kernel<<<blocks, threads, 0, stream.stream()>>>(
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

torch::Tensor wmma_int4_mm(torch::Tensor a, torch::Tensor b_t, int64_t logical_k) {
  ltx2_cuda_int4_convrot::check_wmma_int4_gemm_inputs(a, b_t, logical_k);
  const int64_t M = a.size(0);
  const int64_t N = b_t.size(0);
  auto out = torch::empty({M, N}, a.options().dtype(torch::kInt32));
  if (M == 0 || N == 0 || logical_k == 0) {
    return out;
  }

  c10::cuda::CUDAGuard device_guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
  const dim3 blocks(
      static_cast<unsigned int>(N / ltx2_cuda_int4_convrot::kWmmaTileN),
      static_cast<unsigned int>(M / ltx2_cuda_int4_convrot::kWmmaTileM));
  ltx2_cuda_int4_convrot::wmma_int4_mm_kernel<<<blocks, ltx2_cuda_int4_convrot::kWarp, 0, stream.stream()>>>(
      a.data_ptr<uint8_t>(),
      b_t.data_ptr<uint8_t>(),
      out.data_ptr<int32_t>(),
      static_cast<int>(M),
      static_cast<int>(N),
      static_cast<int>(logical_k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor dequant_epilogue(
    torch::Tensor acc,
    torch::Tensor row_scale,
    c10::optional<torch::Tensor> col_scale,
    c10::optional<torch::Tensor> bias,
    int64_t output_dtype) {
  ltx2_cuda_int4_convrot::check_int32_matrix(acc, "acc");
  const auto M = acc.size(0);
  const auto N = acc.size(1);
  TORCH_CHECK(N <= std::numeric_limits<int>::max(), "N is too large");
  ltx2_cuda_int4_convrot::check_scale_vector(row_scale, "row_scale", M);
  TORCH_CHECK(row_scale.device() == acc.device(), "row_scale must be on acc device");
  const bool use_col_scale = ltx2_cuda_int4_convrot::has_tensor(col_scale);
  const bool use_bias = ltx2_cuda_int4_convrot::has_tensor(bias);
  if (use_col_scale) {
    ltx2_cuda_int4_convrot::check_scale_vector(col_scale.value(), "col_scale", N);
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

  auto out = torch::empty({M, N}, acc.options().dtype(ltx2_cuda_int4_convrot::dtype_from_code(output_dtype)));
  if (M == 0 || N == 0) {
    return out;
  }

  c10::cuda::CUDAGuard device_guard(acc.device());
  auto stream = at::cuda::getCurrentCUDAStream(acc.get_device());
  const int64_t total = M * N;
  const float* col_scale_ptr = use_col_scale ? col_scale.value().data_ptr<float>() : nullptr;
  auto launch_for_output = [&](auto* out_typed) {
    using output_t = std::remove_pointer_t<decltype(out_typed)>;
    if (!use_bias) {
      if (use_col_scale) {
        ltx2_cuda_int4_convrot::launch_dequant_epilogue<output_t, float, true, false>(
            acc, row_scale, col_scale_ptr, nullptr, out, total, static_cast<int>(N), stream.stream());
      } else {
        ltx2_cuda_int4_convrot::launch_dequant_epilogue<output_t, float, false, false>(
            acc, row_scale, nullptr, nullptr, out, total, static_cast<int>(N), stream.stream());
      }
      return;
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, bias.value().scalar_type(), "int4_dequant_epilogue_bias", [&] {
      if (use_col_scale) {
        ltx2_cuda_int4_convrot::launch_dequant_epilogue<output_t, scalar_t, true, true>(
            acc,
            row_scale,
            col_scale_ptr,
            bias.value().data_ptr<scalar_t>(),
            out,
            total,
            static_cast<int>(N),
            stream.stream());
      } else {
        ltx2_cuda_int4_convrot::launch_dequant_epilogue<output_t, scalar_t, false, true>(
            acc,
            row_scale,
            nullptr,
            bias.value().data_ptr<scalar_t>(),
            out,
            total,
            static_cast<int>(N),
            stream.stream());
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

torch::Tensor linear_forward(
    torch::Tensor qx,
    torch::Tensor qw,
    torch::Tensor row_scale,
    torch::Tensor col_scale,
    c10::optional<torch::Tensor> bias,
    int64_t output_dtype) {
  ltx2_cuda_int4_convrot::check_uint8_matrix(qx, "qx");
  ltx2_cuda_int4_convrot::check_uint8_matrix(qw, "qw");
  TORCH_CHECK(qx.device() == qw.device(), "qx and qw must be on the same CUDA device");
  const auto M = qx.size(0);
  const auto packed_K64 = qx.size(1);
  const auto N = qw.size(0);
  TORCH_CHECK(qw.size(1) == packed_K64, "qw packed feature dimension must match qx");
  TORCH_CHECK(packed_K64 <= std::numeric_limits<int>::max() / 2, "K is too large");
  TORCH_CHECK(M <= std::numeric_limits<int>::max() && N <= std::numeric_limits<int>::max(), "M/N are too large");
  const int K = static_cast<int>(packed_K64 * 2);
  const int packed_K = static_cast<int>(packed_K64);
  ltx2_cuda_int4_convrot::check_scale_vector(row_scale, "row_scale", M);
  ltx2_cuda_int4_convrot::check_scale_vector(col_scale, "col_scale", N);
  TORCH_CHECK(row_scale.device() == qx.device(), "row_scale must be on qx device");
  TORCH_CHECK(col_scale.device() == qx.device(), "col_scale must be on qx device");

  const bool use_bias = ltx2_cuda_int4_convrot::has_tensor(bias);
  if (use_bias) {
    TORCH_CHECK(bias.value().is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias.value().is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias.value().numel() == N, "bias must have N elements");
    TORCH_CHECK(bias.value().device() == qx.device(), "bias must be on qx device");
    TORCH_CHECK(
        bias.value().scalar_type() == torch::kFloat32 || bias.value().scalar_type() == torch::kFloat16 ||
            bias.value().scalar_type() == torch::kBFloat16,
        "bias must be float32, float16, or bfloat16");
  }

  auto out = torch::empty({M, N}, qx.options().dtype(ltx2_cuda_int4_convrot::dtype_from_code(output_dtype)));
  if (M == 0 || N == 0 || packed_K64 == 0) {
    return out;
  }

  c10::cuda::CUDAGuard device_guard(qx.device());
  auto stream = at::cuda::getCurrentCUDAStream(qx.get_device());
  auto launch_for_output = [&](auto* out_typed) {
    using output_t = std::remove_pointer_t<decltype(out_typed)>;
    if (!use_bias) {
      ltx2_cuda_int4_convrot::launch_forward<output_t, float, false>(
          qx, qw, row_scale, col_scale, nullptr, out, static_cast<int>(M), static_cast<int>(N), K, packed_K, stream.stream());
      return;
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, bias.value().scalar_type(), "int4_linear_forward_bias", [&] {
      ltx2_cuda_int4_convrot::launch_forward<output_t, scalar_t, true>(
          qx,
          qw,
          row_scale,
          col_scale,
          bias.value().data_ptr<scalar_t>(),
          out,
          static_cast<int>(M),
          static_cast<int>(N),
          K,
          packed_K,
          stream.stream());
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

torch::Tensor linear_backward_input(
    torch::Tensor qg,
    torch::Tensor qw,
    torch::Tensor row_scale,
    int64_t padded_features,
    int64_t output_dtype) {
  ltx2_cuda_int4_convrot::check_uint8_matrix(qg, "qg");
  ltx2_cuda_int4_convrot::check_uint8_matrix(qw, "qw");
  TORCH_CHECK(qg.device() == qw.device(), "qg and qw must be on the same CUDA device");
  const auto M = qg.size(0);
  const auto grad_packed_N64 = qg.size(1);
  const auto N = qw.size(0);
  const auto weight_packed_K64 = qw.size(1);
  TORCH_CHECK(padded_features > 0, "padded_features must be positive");
  TORCH_CHECK((padded_features + 1) / 2 == weight_packed_K64, "padded_features does not match packed weight width");
  TORCH_CHECK((N + 1) / 2 == grad_packed_N64, "qg packed width does not match weight rows");
  TORCH_CHECK(
      M <= std::numeric_limits<int>::max() && N <= std::numeric_limits<int>::max() &&
          padded_features <= std::numeric_limits<int>::max(),
      "M/N/K are too large");
  ltx2_cuda_int4_convrot::check_scale_vector(row_scale, "row_scale", M);
  TORCH_CHECK(row_scale.device() == qg.device(), "row_scale must be on qg device");

  auto out = torch::empty({M, padded_features}, qg.options().dtype(ltx2_cuda_int4_convrot::dtype_from_code(output_dtype)));
  if (M == 0 || N == 0 || padded_features == 0) {
    return out;
  }

  c10::cuda::CUDAGuard device_guard(qg.device());
  auto stream = at::cuda::getCurrentCUDAStream(qg.get_device());
  auto launch_for_output = [&](auto* out_typed) {
    using output_t = std::remove_pointer_t<decltype(out_typed)>;
    ltx2_cuda_int4_convrot::launch_backward_input<output_t>(
        qg,
        qw,
        row_scale,
        out,
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(padded_features),
        static_cast<int>(grad_packed_N64),
        static_cast<int>(weight_packed_K64),
        stream.stream());
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
