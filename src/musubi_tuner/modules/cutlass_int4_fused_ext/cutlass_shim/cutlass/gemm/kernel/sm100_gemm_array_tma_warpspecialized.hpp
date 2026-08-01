// Intentionally empty MSVC shim: the real CUTLASS 3.8 SM100 kernel headers do not
// compile under nvcc+MSVC (constexpr dim3). This W4A4 fused GEMM only ever instantiates
// a 2.x visitor kernel, and cutlass/gemm/kernel/gemm_universal.hpp never names the SM100
// types it includes, so shadowing these four headers with empty stubs is safe on Windows.
#pragma once
