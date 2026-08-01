#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/python.h>

#include <vector>

// Forward declaration of CUDA kernel template
template<typename out_t>
void rms_norm_split_rope_cuda(
    void* x,
    void* sin_freqs,
    void* cos_freqs,
    void* weights,
    int b,
    int s,
    int n,
    int h,
    long cos_sb, long cos_sn, long cos_ss,
    long sin_sb, long sin_sn, long sin_ss,
    float eps,
    void* inv_rms,
    void* out,
    cudaStream_t stream
);

void rms_norm_split_rope_backward_pair_cuda(
    void* grad_q, void* q, void* q_sin, void* q_cos, void* q_weight, void* q_inv_rms,
    int qb, int qs, int qn, int h,
    long q_cos_sb, long q_cos_sn, long q_cos_ss,
    long q_sin_sb, long q_sin_sn, long q_sin_ss,
    void* grad_k, void* k, void* k_sin, void* k_cos, void* k_weight, void* k_inv_rms,
    int kb, int ks, int kn,
    long k_cos_sb, long k_cos_sn, long k_cos_ss,
    long k_sin_sb, long k_sin_sn, long k_sin_ss,
    void* grad_x_q, void* grad_x_k, cudaStream_t stream);

static std::vector<at::Tensor> rms_norm_split_rope_impl(
    at::Tensor &x,
    at::Tensor &sin_freqs,
    at::Tensor &cos_freqs,
    at::Tensor &weights,
    bool out_fp8,
    bool save_inv_rms,
    float eps
) {
    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, "Input must be BFloat16");
    TORCH_CHECK(sin_freqs.scalar_type() == at::ScalarType::BFloat16, "sin_freqs must be BFloat16");
    TORCH_CHECK(cos_freqs.scalar_type() == at::ScalarType::BFloat16, "cos_freqs must be BFloat16");
    TORCH_CHECK(weights.scalar_type() == at::ScalarType::BFloat16, "weights must be BFloat16");
    TORCH_CHECK(x.is_cuda() && sin_freqs.is_cuda() && cos_freqs.is_cuda() && weights.is_cuda(), "Inputs must be on CUDA");

    int b = x.size(0);
    int s = x.size(1);
    int h = x.size(2);
    TORCH_CHECK(cos_freqs.dim() == 4 && sin_freqs.dim() == 4, "Frequencies must be 4D");
    int n = cos_freqs.size(1);

    if (x.stride(-1) != 1) { x = x.contiguous(); }
    if (cos_freqs.stride(-1) != 1) { cos_freqs = cos_freqs.contiguous(); }
    if (sin_freqs.stride(-1) != 1) { sin_freqs = sin_freqs.contiguous(); }

    long cos_sb = cos_freqs.stride(0), cos_sn = cos_freqs.stride(1), cos_ss = cos_freqs.stride(2);
    long sin_sb = sin_freqs.stride(0), sin_sn = sin_freqs.stride(1), sin_ss = sin_freqs.stride(2);

    at::Tensor out = out_fp8
        ? torch::empty(x.sizes(), x.options().dtype(torch::kFloat8_e4m3fn))
        : torch::empty(x.sizes(), x.options().dtype(torch::kBFloat16));
    at::Tensor inv_rms;
    void* inv_rms_ptr = nullptr;
    if (save_inv_rms) {
        inv_rms = torch::empty({b, s}, x.options().dtype(torch::kFloat32));
        inv_rms_ptr = inv_rms.data_ptr();
    }

    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (out_fp8) {
        rms_norm_split_rope_cuda<at::Float8_e4m3fn>(
            x.data_ptr(), sin_freqs.data_ptr(), cos_freqs.data_ptr(), weights.data_ptr(),
            b, s, n, h, cos_sb, cos_sn, cos_ss, sin_sb, sin_sn, sin_ss,
            eps, inv_rms_ptr, out.data_ptr(), stream);
    } else {
        rms_norm_split_rope_cuda<at::BFloat16>(
            x.data_ptr(), sin_freqs.data_ptr(), cos_freqs.data_ptr(), weights.data_ptr(),
            b, s, n, h, cos_sb, cos_sn, cos_ss, sin_sb, sin_sn, sin_ss,
            eps, inv_rms_ptr, out.data_ptr(), stream);
    }
    return save_inv_rms ? std::vector<at::Tensor>{out, inv_rms} : std::vector<at::Tensor>{out};
}

at::Tensor rms_norm_split_rope(
    at::Tensor &x,
    at::Tensor &sin_freqs,
    at::Tensor &cos_freqs,
    at::Tensor &weights,
    bool out_fp8
) {
    return rms_norm_split_rope_impl(x, sin_freqs, cos_freqs, weights, out_fp8, false, 1e-6f)[0];
}

std::vector<at::Tensor> rms_norm_split_rope_with_inv_rms(
    at::Tensor &x,
    at::Tensor &sin_freqs,
    at::Tensor &cos_freqs,
    at::Tensor &weights,
    bool out_fp8,
    double eps
) {
    return rms_norm_split_rope_impl(x, sin_freqs, cos_freqs, weights, out_fp8, true, static_cast<float>(eps));
}

std::vector<at::Tensor> rms_norm_split_rope_backward_pair(
    at::Tensor grad_q,
    at::Tensor q,
    at::Tensor q_sin,
    at::Tensor q_cos,
    at::Tensor q_weight,
    at::Tensor q_inv_rms,
    at::Tensor grad_k,
    at::Tensor k,
    at::Tensor k_sin,
    at::Tensor k_cos,
    at::Tensor k_weight,
    at::Tensor k_inv_rms
) {
    TORCH_CHECK(
        grad_q.scalar_type() == at::ScalarType::BFloat16 && grad_k.scalar_type() == at::ScalarType::BFloat16 &&
        q.scalar_type() == at::ScalarType::BFloat16 && k.scalar_type() == at::ScalarType::BFloat16 &&
        q_weight.scalar_type() == at::ScalarType::BFloat16 && k_weight.scalar_type() == at::ScalarType::BFloat16,
        "Backward inputs must be BFloat16");
    TORCH_CHECK(q_inv_rms.scalar_type() == at::ScalarType::Float && k_inv_rms.scalar_type() == at::ScalarType::Float,
        "inv_rms must be Float32");
    TORCH_CHECK(q.size(2) == k.size(2), "Q and K hidden dimensions must match");
    TORCH_CHECK(q.device() == k.device(), "Q and K must be on the same device");

    grad_q = grad_q.contiguous();
    grad_k = grad_k.contiguous();
    q = q.contiguous();
    k = k.contiguous();
    q_weight = q_weight.contiguous();
    k_weight = k_weight.contiguous();
    if (q_cos.stride(-1) != 1) { q_cos = q_cos.contiguous(); }
    if (q_sin.stride(-1) != 1) { q_sin = q_sin.contiguous(); }
    if (k_cos.stride(-1) != 1) { k_cos = k_cos.contiguous(); }
    if (k_sin.stride(-1) != 1) { k_sin = k_sin.contiguous(); }

    auto grad_x_q = torch::empty_like(q);
    auto grad_x_k = torch::empty_like(k);
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    rms_norm_split_rope_backward_pair_cuda(
        grad_q.data_ptr(), q.data_ptr(), q_sin.data_ptr(), q_cos.data_ptr(), q_weight.data_ptr(), q_inv_rms.data_ptr(),
        q.size(0), q.size(1), q_cos.size(1), q.size(2),
        q_cos.stride(0), q_cos.stride(1), q_cos.stride(2), q_sin.stride(0), q_sin.stride(1), q_sin.stride(2),
        grad_k.data_ptr(), k.data_ptr(), k_sin.data_ptr(), k_cos.data_ptr(), k_weight.data_ptr(), k_inv_rms.data_ptr(),
        k.size(0), k.size(1), k_cos.size(1),
        k_cos.stride(0), k_cos.stride(1), k_cos.stride(2), k_sin.stride(0), k_sin.stride(1), k_sin.stride(2),
        grad_x_q.data_ptr(), grad_x_k.data_ptr(), stream);
    return {grad_x_q, grad_x_k};
}
