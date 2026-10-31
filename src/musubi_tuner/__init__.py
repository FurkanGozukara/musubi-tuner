import os

# torch 2.13 puts cuDNN attention first for SDPA on compute capability 9.x / 10.x (H100, H200, B200) with cuDNN > 9.15,
# and cuDNN SDPA can return NaN gradients from finite inputs (cudnn-frontend #1072 / #160; SimpleTuner disabled it on
# 2026-10-26). torch reads this once, at the first SDPA call: every script imports this package before that. An explicit
# TORCH_CUDNN_SDPA_DEPRIORITIZED=0 keeps torch's own order.
os.environ.setdefault("TORCH_CUDNN_SDPA_DEPRIORITIZED", "1")
