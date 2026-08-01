#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// CUDA kernel template for RMS norm + split RoPE
// out_t can be at::Float8_e4m3fn or at::BFloat16

using bf16 = __nv_bfloat16;
using fp8 = __nv_fp8_e4m3;
__device__ __forceinline__ void _load_row(const bf16* x, float x_vals[8], int h, int row){
    bf16 x_tmp[8];
    *reinterpret_cast<int4*>(x_tmp) = reinterpret_cast<const int4*>(x + row * h)[threadIdx.x];
    #pragma unroll
    for(int i = 0; i < 8; i++){
        x_vals[i] = float(x_tmp[i]);
    }
}
// Load 8 freq values for this thread from a table laid out logically as
// [b, n, s, d/2] (what apply_split_rotary_emb produces -- a swapaxes view whose
// physical layout is [b, s, n, d/2]). The strides (sb, sn, ss; inner d/2 stride
// is 1) are forwarded from the host so the read is correct for both that
// non-contiguous view and a genuinely contiguous [b, n, s, d/2] tensor. Both
// head-halves map to the same freq element (mirrors the eager cos.unsqueeze(-2)).
__device__ __forceinline__ void _load_freqs(
    const bf16* freqs, float x_vals[8], int row, int s, int d, long sb, long sn, long ss
){
    bf16 x_tmp[8];
    int threads_per_head = d / 8;
    int head_idx = threadIdx.x / threads_per_head;
    int lane = threadIdx.x % (threads_per_head / 2);
    int b_idx = row / s;
    int t_idx = row % s;
    long off = b_idx * sb + head_idx * sn + t_idx * ss + (long)lane * 8;
    *reinterpret_cast<int4*>(x_tmp) = *reinterpret_cast<const int4*>(freqs + off);
    #pragma unroll
    for(int i = 0; i < 8; i++){
        x_vals[i] = float(x_tmp[i]);
    }
}

template<typename out_t>
__global__ void _rms_norm_split_rope_kernel(
    bf16* x, bf16* sin_freqs, bf16* cos_freqs, void* out, bf16* weights, int b, int s, int n, int h,
    long cos_sb, long cos_sn, long cos_ss, long sin_sb, long sin_sn, long sin_ss, float eps, float* inv_rms_out){
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    // freqs have shape [b, s, h/2]
    // each thread block calculate one row
    // there are h/8 threads in thread block, each thread processes 8 values
    // num_of_rows = b * s
    // freqs have h/2 dim
    // gridDim is (num_of_rows, 1, 1)
    // calculate rms norm x_normed = x/x_norm * weights. x_norm is calculated across row, it means thread block wide sum reduction

    extern __shared__ float smem[];

    // Step 1: Load input values (8 per thread)
    float x_vals[8];
    _load_row(x, x_vals, h, token_idx);

    float sum_sq = 0.0f;
    #pragma unroll
    for(int i = 0; i < 8; i++){
        sum_sq += x_vals[i] * x_vals[i];
    }

    // Warp-level reduction
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1){
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }

    if(tid % 32 == 0){
        smem[tid / 32] = sum_sq;
    }
    __syncthreads();

    // Final reduction across warps
    if(tid == 0){
        float total_sum = 0.0f;
        int num_warps = blockDim.x / 32;
        for(int i = 0; i < num_warps; i++){
            total_sum += smem[i];
        }
        // RMS: sqrt(mean(x^2))
        float rms = rsqrtf(total_sum / h + eps);
        smem[0] = rms;
        if (inv_rms_out != nullptr) inv_rms_out[token_idx] = rms;
    }
    __syncthreads();

    float inv_rms = smem[0];

    // Step 3: Apply RMS normalization (and weights if provided)
    #pragma unroll
    for(int i = 0; i < 8; i++){
        x_vals[i] *= inv_rms;
        // Apply the RMSNorm affine weight (indexed over the full hidden dim).
        if(weights != nullptr) x_vals[i] *= float(weights[tid * 8 + i]);
    }

    // Step 4: Calculate dimensions for split RoPE
    // Conceptually: [b, s, h] -> [b, s, n, 2*d] -> [b, s, n, 2, d]
    // where h = n * 2 * d
    int d = h / n;
    float x_other_vals[8];

    int threads_per_head = d / 8;
    int head_idx = tid / threads_per_head;
    int idx_in_head = tid % threads_per_head;
    bool is_first_half = idx_in_head < (threads_per_head / 2);

    // Full-warp mask: every lane in the warp participates. The XOR exchange stays
    // within each power-of-two head group (laneMask < threads_per_head), so a
    // full-warp mask is the correct participation set for every lane. A partial
    // mask covering only one head's lanes would leave other lanes' shuffle results
    // undefined and corrupt the RoPE mixing.
    const unsigned mask = 0xffffffffu;
    const int laneMask = threads_per_head / 2;           // 4, 8, or 16
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        x_other_vals[i] = __shfl_xor_sync(mask, x_vals[i], laneMask);
    }

    float cos_vals[8], sin_vals[8];
    _load_freqs(cos_freqs, cos_vals, token_idx, s, d, cos_sb, cos_sn, cos_ss);
    _load_freqs(sin_freqs, sin_vals, token_idx, s, d, sin_sb, sin_sn, sin_ss);
    #pragma unroll
    for(int i = 0; i < 8; i++){
        x_vals[i] = cos_vals[i]*x_vals[i];
    }


    float sign = is_first_half ? -1.0f : 1.0f;
    for(int i = 0; i < 8; i++){
        x_vals[i] += sign*sin_vals[i]*x_other_vals[i];
    }

    // Step 6: Convert and store output
    if constexpr (std::is_same_v<out_t, at::Float8_e4m3fn>){
        fp8 out_tmp[8];
        #pragma unroll
        for(int i = 0; i < 8; i++){
            out_tmp[i] = fp8(x_vals[i]);
        }
        *reinterpret_cast<int64_t*>((fp8*)out + token_idx * h + tid * 8) = *reinterpret_cast<int64_t*>(out_tmp);
    } else {
        bf16 out_tmp[8];
        #pragma unroll
        for(int i = 0; i < 8; i++){
            out_tmp[i] = __float2bfloat16(x_vals[i]);
        }
        *reinterpret_cast<int4*>((bf16*)out + token_idx * h + tid * 8) = *reinterpret_cast<int4*>(out_tmp);
    }
}

template<typename out_t>
void rms_norm_split_rope_cuda(
    void* x,           // Input: [b, s, h]
    void* sin_freqs,   // Sin frequencies: [b, n, s, d]
    void* cos_freqs,   // Cos frequencies: [b, n, s, d]
    void* weights,
    int b,                      // Batch size
    int s,                      // Sequence length
    int n,                      // Number of heads (32)
    int h,                      // Hidden dimension (2048, 4096, or 8192)
    long cos_sb, long cos_sn, long cos_ss,  // cos_freqs strides (b, n, s)
    long sin_sb, long sin_sn, long sin_ss,  // sin_freqs strides (b, n, s)
    float eps,
    void* inv_rms,
    void* out,                // Output: [b, s, h]
    cudaStream_t stream
) {
    int num_tokens = b * s;
    int num_threads = h / 8;  // Each thread processes 8 elements
    int smem_size = (num_threads / 32 + 1) * sizeof(float);  // Shared memory for reductions

    dim3 grid(num_tokens);
    dim3 block(num_threads);

    _rms_norm_split_rope_kernel<out_t><<<grid, block, smem_size, stream>>>(
        reinterpret_cast<bf16*>(x),
        reinterpret_cast<bf16*>(sin_freqs),
        reinterpret_cast<bf16*>(cos_freqs),
        out,
        reinterpret_cast<bf16*>(weights),
        b, s, n, h,
        cos_sb, cos_sn, cos_ss,
        sin_sb, sin_sn, sin_ss,
        eps,
        reinterpret_cast<float*>(inv_rms)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void _rms_norm_split_rope_backward_pair_kernel(
    const bf16* grad_q, const bf16* q, const bf16* q_sin, const bf16* q_cos, const bf16* q_weight,
    const float* q_inv_rms, int q_rows, int qs, int qn,
    long q_cos_sb, long q_cos_sn, long q_cos_ss, long q_sin_sb, long q_sin_sn, long q_sin_ss,
    const bf16* grad_k, const bf16* k, const bf16* k_sin, const bf16* k_cos, const bf16* k_weight,
    const float* k_inv_rms, int k_rows, int ks, int kn,
    long k_cos_sb, long k_cos_sn, long k_cos_ss, long k_sin_sb, long k_sin_sn, long k_sin_ss,
    bf16* grad_x_q, bf16* grad_x_k, int h){
    int global_row = blockIdx.x;
    bool is_q = global_row < q_rows;
    int row = is_q ? global_row : global_row - q_rows;
    int s = is_q ? qs : ks;
    int n = is_q ? qn : kn;
    const bf16* grad = is_q ? grad_q : grad_k;
    const bf16* x = is_q ? q : k;
    const bf16* sin_freqs = is_q ? q_sin : k_sin;
    const bf16* cos_freqs = is_q ? q_cos : k_cos;
    const bf16* weights = is_q ? q_weight : k_weight;
    const float* inv_rms = is_q ? q_inv_rms : k_inv_rms;
    bf16* grad_x = is_q ? grad_x_q : grad_x_k;
    long cos_sb = is_q ? q_cos_sb : k_cos_sb;
    long cos_sn = is_q ? q_cos_sn : k_cos_sn;
    long cos_ss = is_q ? q_cos_ss : k_cos_ss;
    long sin_sb = is_q ? q_sin_sb : k_sin_sb;
    long sin_sn = is_q ? q_sin_sn : k_sin_sn;
    long sin_ss = is_q ? q_sin_ss : k_sin_ss;

    int tid = threadIdx.x;
    int d = h / n;
    int threads_per_head = d / 8;
    int idx_in_head = tid % threads_per_head;
    bool first_half = idx_in_head < threads_per_head / 2;

    float grad_vals[8], other_vals[8], x_vals[8], cos_vals[8], sin_vals[8];
    _load_row(grad, grad_vals, h, row);
    _load_row(x, x_vals, h, row);
    _load_freqs(cos_freqs, cos_vals, row, s, d, cos_sb, cos_sn, cos_ss);
    _load_freqs(sin_freqs, sin_vals, row, s, d, sin_sb, sin_sn, sin_ss);

    int lane_mask = threads_per_head / 2;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        other_vals[i] = __shfl_xor_sync(0xffffffffu, grad_vals[i], lane_mask);
    }

    float g_vals[8];
    float row_dot = 0.0f;
    float rope_sign = first_half ? 1.0f : -1.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        float da = cos_vals[i] * grad_vals[i] + rope_sign * sin_vals[i] * other_vals[i];
        g_vals[i] = da * float(weights[tid * 8 + i]);
        row_dot += g_vals[i] * x_vals[i];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        row_dot += __shfl_xor_sync(0xffffffffu, row_dot, offset);
    }

    extern __shared__ float smem[];
    if (tid % 32 == 0) smem[tid / 32] = row_dot;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < blockDim.x / 32; i++) total += smem[i];
        smem[0] = total;
    }
    __syncthreads();

    float r = inv_rms[row];
    float correction = r * r * r * smem[0] / h;
    bf16 out_vals[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out_vals[i] = __float2bfloat16(r * g_vals[i] - correction * x_vals[i]);
    }
    *reinterpret_cast<int4*>(grad_x + row * h + tid * 8) = *reinterpret_cast<int4*>(out_vals);
}

void rms_norm_split_rope_backward_pair_cuda(
    void* grad_q, void* q, void* q_sin, void* q_cos, void* q_weight, void* q_inv_rms,
    int qb, int qs, int qn, int h,
    long q_cos_sb, long q_cos_sn, long q_cos_ss,
    long q_sin_sb, long q_sin_sn, long q_sin_ss,
    void* grad_k, void* k, void* k_sin, void* k_cos, void* k_weight, void* k_inv_rms,
    int kb, int ks, int kn,
    long k_cos_sb, long k_cos_sn, long k_cos_ss,
    long k_sin_sb, long k_sin_sn, long k_sin_ss,
    void* grad_x_q, void* grad_x_k, cudaStream_t stream){
    int q_rows = qb * qs;
    int k_rows = kb * ks;
    int num_threads = h / 8;
    int smem_size = (num_threads / 32 + 1) * sizeof(float);
    _rms_norm_split_rope_backward_pair_kernel<<<q_rows + k_rows, num_threads, smem_size, stream>>>(
        reinterpret_cast<const bf16*>(grad_q), reinterpret_cast<const bf16*>(q),
        reinterpret_cast<const bf16*>(q_sin), reinterpret_cast<const bf16*>(q_cos),
        reinterpret_cast<const bf16*>(q_weight), reinterpret_cast<const float*>(q_inv_rms),
        q_rows, qs, qn, q_cos_sb, q_cos_sn, q_cos_ss, q_sin_sb, q_sin_sn, q_sin_ss,
        reinterpret_cast<const bf16*>(grad_k), reinterpret_cast<const bf16*>(k),
        reinterpret_cast<const bf16*>(k_sin), reinterpret_cast<const bf16*>(k_cos),
        reinterpret_cast<const bf16*>(k_weight), reinterpret_cast<const float*>(k_inv_rms),
        k_rows, ks, kn, k_cos_sb, k_cos_sn, k_cos_ss, k_sin_sb, k_sin_sn, k_sin_ss,
        reinterpret_cast<bf16*>(grad_x_q), reinterpret_cast<bf16*>(grad_x_k), h);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Explicit template instantiations
template void rms_norm_split_rope_cuda<at::BFloat16>(
    void*, void*, void*, void*, int, int, int, int, long, long, long, long, long, long, float, void*, void*, cudaStream_t
);

template void rms_norm_split_rope_cuda<at::Float8_e4m3fn>(
    void*, void*, void*, void*, int, int, int, int, long, long, long, long, long, long, float, void*, void*, cudaStream_t
);
