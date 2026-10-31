"""Anima DiT (circlestone-labs/Anima, 2B): NVIDIA Cosmos-Predict2's MiniTrainDiT with an LLM adapter.

28 blocks of AdaLN-LoRA modulated self-attention (3D RoPE, QK RMSNorm), cross-attention to the text context and a GELU
MLP, a patch embedder that takes the latent plus a zero padding-mask channel, and a final AdaLN layer. The text context
comes from the LLM adapter, which also lives in the DiT file (``llm_adapter.*``): the caption's T5 token ids (the T5
vocabulary only, no T5 weights) are embedded and run through six blocks of self-attention, cross-attention to the
Qwen3-0.6B last hidden state and an MLP; the output is zero-padded to 512 tokens (ComfyUI ``preprocess_text_embeds``,
diffusers ``min_sequence_length``). The DiT cross-attends to all 512 tokens without a mask, so the pads (zero keys and
values) are part of the condition.

Ported from kohya-ss sd-scripts' ``library/anima_models.py`` (0.12.3) and checked against it, ComfyUI's
``comfy/ldm/anima`` + ``comfy/ldm/cosmos/predict2.py`` and diffusers' ``CosmosTransformer3DModel`` /
``AnimaTextConditioner`` (ai-toolkit's Anima) in fp32 on random weights (``dont_upload/vram_lab/anima_ref_check.py``).
The dtype handling is ComfyUI's: the timestep sinusoid and the RoPE rotation in fp32, everything else in the weights'
dtype, and with FP16 weights the residual stream in fp32 (the stream outgrows FP16, ComfyUI's note). Differences that do
not change the outputs: images only (one latent frame), so the blocks work on (B, L, D) token sequences; the 3D RoPE
is built per call (no maximum size); the blocks swap to the CPU and checkpoint their activations.
"""

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from musubi_tuner.modules.attention import AttentionParams, attention
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig, create_offloader
from musubi_tuner.utils.model_utils import create_cpu_offloading_wrapper

# Linear weights of the 28 blocks (FP8 scaled); the modulations, norms, embedders, final layer and the LLM adapter keep
# the checkpoint's precision
FP8_OPTIMIZATION_TARGET_KEYS = ["blocks"]
FP8_OPTIMIZATION_EXCLUDE_KEYS = ["norm", "adaln_modulation", "_embedder", "final_layer", "llm_adapter"]

CONTEXT_TOKENS = 512  # the adapter's output is zero-padded to at least this many tokens
T5_VOCAB_SIZE = 32128
T5_EOS_TOKEN_ID = 1
T5_PAD_TOKEN_ID = 0


@dataclass
class AnimaParams:
    model_channels: int = 2048
    num_blocks: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    in_channels: int = 16
    out_channels: int = 16
    patch_spatial: int = 2
    patch_temporal: int = 1
    crossattn_emb_channels: int = 1024
    adaln_lora_dim: int = 256
    rope_h_extrapolation_ratio: float = 4.0
    rope_w_extrapolation_ratio: float = 4.0
    rope_t_extrapolation_ratio: float = 1.0
    # LLM adapter
    adapter_dim: int = 1024
    adapter_layers: int = 6
    adapter_heads: int = 16
    adapter_vocab_size: int = T5_VOCAB_SIZE


# ------------------------------------------------------------------------------------------------------------- basics
class RMSNorm(nn.Module):
    """RMSNorm in the input's dtype with the weight cast to it (ComfyUI's ``comfy.rmsnorm``)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), self.eps)


def layer_norm(x: Tensor) -> Tensor:
    """LayerNorm without affine parameters (eps 1e-6), as every norm of the DiT blocks and the final layer."""
    return F.layer_norm(x, (x.shape[-1],), eps=1e-6)


def apply_rope_split_half(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x (B, L, H, D); cos / sin (L, D / 2) fp32. Rotates dims i and i + D / 2 by the i-th angle in fp32 (ComfyUI's
    ``rms_rope_split_half``; kohya's ``_rotate_half`` non-interleaved) and returns x's dtype."""
    x32 = x.float()
    a, b = x32.chunk(2, dim=-1)
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    return torch.cat((a * c - b * s, b * c + a * s), dim=-1).type_as(x)


def rope_3d(frames: int, height: int, width: int, head_dim: int, ratios: tuple[float, float, float], device) -> tuple[Tensor, Tensor]:
    """Cosmos' 3D RoPE angles of a (frames, height, width) patch grid: (L, head_dim / 2) cos and sin in fp32, per token
    [t angles | h angles | w angles], NTK-scaled thetas (Anima: h / w extrapolation 4, t 1, no fps modulation)."""
    dim_h = head_dim // 6 * 2
    dim_w = dim_h
    dim_t = head_dim - 2 * dim_h
    spatial_range = torch.arange(0, dim_h, 2, device=device)[: dim_h // 2].float() / dim_h
    temporal_range = torch.arange(0, dim_t, 2, device=device)[: dim_t // 2].float() / dim_t
    ratio_t, ratio_h, ratio_w = ratios
    h_theta = 10000.0 * ratio_h ** (dim_h / (dim_h - 2))
    w_theta = 10000.0 * ratio_w ** (dim_w / (dim_w - 2))
    t_theta = 10000.0 * ratio_t ** (dim_t / (dim_t - 2))
    h_freqs = 1.0 / (h_theta**spatial_range)
    w_freqs = 1.0 / (w_theta**spatial_range)
    t_freqs = 1.0 / (t_theta**temporal_range)
    seq = torch.arange(max(frames, height, width), dtype=torch.float, device=device)
    half_t = torch.outer(seq[:frames], t_freqs)
    half_h = torch.outer(seq[:height], h_freqs)
    half_w = torch.outer(seq[:width], w_freqs)
    angles = torch.cat(
        (
            half_t[:, None, None, :].expand(frames, height, width, -1),
            half_h[None, :, None, :].expand(frames, height, width, -1),
            half_w[None, None, :, :].expand(frames, height, width, -1),
        ),
        dim=-1,
    ).reshape(frames * height * width, -1)
    return angles.cos(), angles.sin()


def timestep_sinusoid(timesteps: Tensor, dim: int) -> Tensor:
    """(B,) -> (B, dim) fp32 [cos | sin], Cosmos' ``Timesteps`` (Anima's timestep is sigma in 0..1)."""
    half = dim // 2
    exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    emb = timesteps.reshape(-1).float()[:, None] * torch.exp(exponent)[None, :]
    return torch.cat((emb.cos(), emb.sin()), dim=-1)


class Timesteps(nn.Module):
    """Parameter-free first stage of ``t_embedder`` (kept so the checkpoint's ``t_embedder.1.*`` keys match)."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps: Tensor) -> Tensor:
        return timestep_sinusoid(timesteps, self.num_channels)


class TimestepEmbedding(nn.Module):
    """AdaLN-LoRA timestep embedding: returns the sinusoid itself as the embedding and the 3 * D AdaLN-LoRA term."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim, bias=False)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, sample: Tensor) -> tuple[Tensor, Tensor]:
        return sample, self.linear_2(self.activation(self.linear_1(sample)))


class _Patchify(nn.Module):
    """``b c (t r) (h m) (w n) -> b t h w (c r m n)``."""

    def __init__(self, temporal: int, spatial: int):
        super().__init__()
        self.temporal = temporal
        self.spatial = spatial

    def forward(self, x: Tensor) -> Tensor:
        b, c, t, h, w = x.shape
        r, m = self.temporal, self.spatial
        x = x.view(b, c, t // r, r, h // m, m, w // m, m)
        return x.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(b, t // r, h // m, w // m, c * r * m * m)


class PatchEmbed(nn.Module):
    def __init__(self, spatial: int, temporal: int, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Sequential(_Patchify(temporal, spatial), nn.Linear(in_channels * spatial * spatial * temporal, out_channels, bias=False))

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class GPT2FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.layer1 = nn.Linear(dim, hidden, bias=False)
        self.activation = nn.GELU()
        self.layer2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer2(self.activation(self.layer1(x)))


class Attention(nn.Module):
    """Multi-head attention with QK RMSNorm; RoPE on q and k for self-attention."""

    def __init__(self, query_dim: int, context_dim: int, heads: int, head_dim: int):
        super().__init__()
        inner = heads * head_dim
        self.heads = heads
        self.q_proj = nn.Linear(query_dim, inner, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_proj = nn.Linear(context_dim, inner, bias=False)
        self.k_norm = RMSNorm(head_dim)
        self.v_proj = nn.Linear(context_dim, inner, bias=False)
        self.output_proj = nn.Linear(inner, query_dim, bias=False)

    def forward(
        self,
        x: Tensor,
        attn_params: AttentionParams,
        context: Optional[Tensor] = None,
        rope_cos: Optional[Tensor] = None,
        rope_sin: Optional[Tensor] = None,
    ) -> Tensor:
        source = x if context is None else context
        q = self.q_norm(self.q_proj(x).unflatten(-1, (self.heads, -1)))
        k = self.k_norm(self.k_proj(source).unflatten(-1, (self.heads, -1)))
        v = self.v_proj(source).unflatten(-1, (self.heads, -1))
        if rope_cos is not None:
            q = apply_rope_split_half(q, rope_cos, rope_sin)
            k = apply_rope_split_half(k, rope_cos, rope_sin)
        return self.output_proj(attention([q, k, v], attn_params=attn_params))  # (B, L, H * D)


class _CheckpointMixin:
    gradient_checkpointing = False
    activation_cpu_offloading = False
    # set by the trainable block-swap ring (full fine-tuning): a checkpointed block's recompute waits for its weights
    recompute_wait: Optional[Callable[[], None]] = None

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def _run(self, fn, device: torch.device, *args):
        if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
            if self.activation_cpu_offloading:
                fn = create_cpu_offloading_wrapper(fn, device)
            kwargs = {"use_reentrant": False}
            if self.recompute_wait is not None:
                wait = self.recompute_wait

                @contextmanager
                def before_recompute():
                    wait()
                    yield

                kwargs["context_fn"] = lambda: (nullcontext(), before_recompute())
            return checkpoint(fn, *args, **kwargs)
        return fn(*args)


class AnimaBlock(_CheckpointMixin, nn.Module):
    """One of the 28 DiT blocks (the LoRA target): x = x + gate * f(norm(x) * (1 + scale) + shift) for self-attention,
    cross-attention and the MLP, each with its own AdaLN-LoRA modulation."""

    def __init__(self, dim: int, context_dim: int, heads: int, mlp_ratio: float, adaln_lora_dim: int):
        super().__init__()
        head_dim = dim // heads
        self.self_attn = Attention(dim, dim, heads, head_dim)
        self.cross_attn = Attention(dim, context_dim, heads, head_dim)
        self.mlp = GPT2FeedForward(dim, int(dim * mlp_ratio))
        self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(dim, adaln_lora_dim, bias=False), nn.Linear(adaln_lora_dim, 3 * dim, bias=False))
        self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(dim, adaln_lora_dim, bias=False), nn.Linear(adaln_lora_dim, 3 * dim, bias=False))
        self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, adaln_lora_dim, bias=False), nn.Linear(adaln_lora_dim, 3 * dim, bias=False))

    def _forward(
        self,
        x: Tensor,
        emb: Tensor,
        context: Tensor,
        rope_cos: Tensor,
        rope_sin: Tensor,
        adaln_lora: Tensor,
        attn_params: AttentionParams,
    ) -> Tensor:
        """x (B, L, D) in the residual dtype; emb (B, D) and adaln_lora (B, 3D) in the compute dtype."""
        residual_dtype = x.dtype
        compute_dtype = emb.dtype
        shift_sa, scale_sa, gate_sa = (self.adaln_modulation_self_attn(emb) + adaln_lora).unsqueeze(1).chunk(3, dim=-1)
        shift_ca, scale_ca, gate_ca = (self.adaln_modulation_cross_attn(emb) + adaln_lora).unsqueeze(1).chunk(3, dim=-1)
        shift_mlp, scale_mlp, gate_mlp = (self.adaln_modulation_mlp(emb) + adaln_lora).unsqueeze(1).chunk(3, dim=-1)

        h = (layer_norm(x) * (1 + scale_sa) + shift_sa).to(compute_dtype)
        out = self.self_attn(h, attn_params, None, rope_cos, rope_sin)
        x = torch.addcmul(x, gate_sa.to(residual_dtype), out.to(residual_dtype))

        h = (layer_norm(x) * (1 + scale_ca) + shift_ca).to(compute_dtype)
        out = self.cross_attn(h, attn_params, context)
        x = torch.addcmul(x, gate_ca.to(residual_dtype), out.to(residual_dtype))

        h = (layer_norm(x) * (1 + scale_mlp) + shift_mlp).to(compute_dtype)
        out = self.mlp(h)
        return torch.addcmul(x, gate_mlp.to(residual_dtype), out.to(residual_dtype))

    def forward(self, x, emb, context, rope_cos, rope_sin, adaln_lora, attn_params):
        return self._run(self._forward, emb.device, x, emb, context, rope_cos, rope_sin, adaln_lora, attn_params)


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_features: int, adaln_lora_dim: int):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, out_features, bias=False)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, adaln_lora_dim, bias=False), nn.Linear(adaln_lora_dim, 2 * dim, bias=False))

    def forward(self, x: Tensor, emb: Tensor, adaln_lora: Tensor) -> Tensor:
        shift, scale = (self.adaln_modulation(emb) + adaln_lora[:, : 2 * self.dim]).unsqueeze(1).chunk(2, dim=-1)
        return self.linear(layer_norm(x) * (1 + scale) + shift)


# -------------------------------------------------------------------------------------------------------- LLM adapter
class AdapterRotaryEmbedding(nn.Module):
    """1D RoPE of the adapter (theta 10000): cos / sin in fp32, returned in x's dtype like ComfyUI and kohya."""

    def __init__(self, head_dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, length: int) -> tuple[Tensor, Tensor]:
        positions = torch.arange(length, device=x.device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(x.device).float())
        emb = torch.cat((freqs, freqs), dim=-1)  # (L, head_dim)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def _adapter_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x (B, H, L, D), cos / sin (L, D) in x's dtype (rotate_half)."""
    a, b = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-b, a), dim=-1) * sin


class AdapterAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, heads: int, head_dim: int):
        super().__init__()
        inner = heads * head_dim
        self.heads = heads
        self.q_proj = nn.Linear(query_dim, inner, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_proj = nn.Linear(context_dim, inner, bias=False)
        self.k_norm = RMSNorm(head_dim)
        self.v_proj = nn.Linear(context_dim, inner, bias=False)
        self.o_proj = nn.Linear(inner, query_dim, bias=False)

    def forward(self, x: Tensor, mask: Optional[Tensor], context: Tensor, rope_q, rope_k) -> Tensor:
        q = self.q_norm(self.q_proj(x).unflatten(-1, (self.heads, -1))).transpose(1, 2)
        k = self.k_norm(self.k_proj(context).unflatten(-1, (self.heads, -1))).transpose(1, 2)
        v = self.v_proj(context).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        q = _adapter_rope(q, *rope_q)
        k = _adapter_rope(k, *rope_k)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(out.transpose(1, 2).flatten(2))


class AnimaAdapterBlock(nn.Module):
    """One of the adapter's six blocks (a LoRA target only with ``train_llm_adapter=True``)."""

    def __init__(self, source_dim: int, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        head_dim = dim // heads
        self.norm_self_attn = RMSNorm(dim)
        self.self_attn = AdapterAttention(dim, dim, heads, head_dim)
        self.norm_cross_attn = RMSNorm(dim)
        self.cross_attn = AdapterAttention(dim, source_dim, heads, head_dim)
        self.norm_mlp = RMSNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x, context, target_mask, source_mask, rope_target, rope_source) -> Tensor:
        h = self.norm_self_attn(x)
        x = x + self.self_attn(h, target_mask, h, rope_target, rope_target)
        x = x + self.cross_attn(self.norm_cross_attn(x), source_mask, context, rope_target, rope_source)
        return x + self.mlp(self.norm_mlp(x))


class LLMAdapter(nn.Module):
    """Qwen3-0.6B last hidden state (source) + T5 token ids (target) -> the DiT's (B, T, 1024) text context."""

    def __init__(self, source_dim: int = 1024, dim: int = 1024, layers: int = 6, heads: int = 16, vocab_size: int = T5_VOCAB_SIZE):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.rotary_emb = AdapterRotaryEmbedding(dim // heads)
        self.blocks = nn.ModuleList([AnimaAdapterBlock(source_dim, dim, heads) for _ in range(layers)])
        self.out_proj = nn.Linear(dim, dim)
        self.norm = RMSNorm(dim)

    def forward(
        self,
        source: Tensor,
        target_ids: Tensor,
        target_mask: Optional[Tensor] = None,
        source_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """source (B, S, 1024) Qwen3 states, target_ids (B, T) T5 ids; masks (B, T) / (B, S), True = real token. Without
        masks every token is real (one caption, as ComfyUI runs it)."""
        target_mask = None if target_mask is None else target_mask.to(torch.bool)[:, None, None, :]
        source_mask = None if source_mask is None else source_mask.to(torch.bool)[:, None, None, :]
        x = self.embed(target_ids).to(source.dtype)
        rope_target = self.rotary_emb(x, x.shape[1])
        rope_source = self.rotary_emb(x, source.shape[1])
        for block in self.blocks:
            x = block(x, source, target_mask, source_mask, rope_target, rope_source)
        return self.norm(self.out_proj(x))


# ---------------------------------------------------------------------------------------------------------------- DiT
class Anima(nn.Module):
    def __init__(self, params: Optional[AnimaParams] = None, attn_mode: str = "torch", split_attn: bool = False):
        super().__init__()
        params = params or AnimaParams()
        self.params = params
        # every mode of modules.attention, the fork's "flash_auto" (resolve_sdpa_backend) included (Guardrail 3)
        self.attn_mode = "torch" if attn_mode == "sdpa" else attn_mode
        self.split_attn = split_attn
        dim = params.model_channels
        self.x_embedder = PatchEmbed(params.patch_spatial, params.patch_temporal, params.in_channels + 1, dim)
        self.t_embedder = nn.Sequential(Timesteps(dim), TimestepEmbedding(dim))
        self.blocks = nn.ModuleList(
            [AnimaBlock(dim, params.crossattn_emb_channels, params.num_heads, params.mlp_ratio, params.adaln_lora_dim) for _ in range(params.num_blocks)]
        )
        out_features = params.patch_spatial * params.patch_spatial * params.patch_temporal * params.out_channels
        self.final_layer = FinalLayer(dim, out_features, params.adaln_lora_dim)
        self.t_embedding_norm = RMSNorm(dim)
        self.llm_adapter = LLMAdapter(params.crossattn_emb_channels, params.adapter_dim, params.adapter_layers, params.adapter_heads, params.adapter_vocab_size)

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        self.blocks_to_swap = None
        self.offloader = None
        self.num_blocks = len(self.blocks)

    @property
    def device(self):
        return self.t_embedding_norm.weight.device

    @property
    def dtype(self):
        # the compute dtype: the norms keep the checkpoint's dtype under FP8 scaled weights
        return self.t_embedding_norm.weight.dtype

    # ------------------------------------------------------------------ memory
    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading
        for block in self.blocks:
            block.enable_gradient_checkpointing(activation_cpu_offloading)
        print(f"Anima: Gradient checkpointing enabled. Activation CPU offloading: {activation_cpu_offloading}")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        for block in self.blocks:
            block.disable_gradient_checkpointing()
        print("Anima: Gradient checkpointing disabled.")

    def enable_block_swap(self, num_blocks: int, config: BlockSwapConfig):
        """Swap ``num_blocks`` of the 28 blocks (two stay resident for the backward pass, Guardrail 9)."""
        assert 0 < num_blocks <= self.num_blocks - 2, f"Anima: blocks_to_swap must be 1 to {self.num_blocks - 2}, got {num_blocks}"
        self.blocks_to_swap = num_blocks
        self.offloader = create_offloader("anima", self.blocks, self.num_blocks, num_blocks, config)
        if getattr(self.offloader, "recompute_requires_wait", False):  # the trainable ring (full fine-tuning)
            for index, block in enumerate(self.blocks):
                block.recompute_wait = partial(self.offloader.wait_for_block, index)
        print(f"Anima: Block swap enabled. Swapping {num_blocks} of {self.num_blocks} blocks.")

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()
            print("Anima: Block swap set to forward only.")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()
            print("Anima: Block swap set to forward and backward.")

    def move_to_device_except_swap_blocks(self, device: torch.device):
        # the model is on the CPU: leave the swapped blocks there
        if self.blocks_to_swap:
            saved = self.blocks
            self.blocks = nn.ModuleList()
        self.to(device)
        if self.blocks_to_swap:
            self.blocks = saved

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.prepare_block_devices_before_forward(self.blocks)

    # ------------------------------------------------------------------ text
    def encode_context(
        self,
        source: Tensor,
        t5_ids: Tensor,
        t5_mask: Optional[Tensor] = None,
        source_mask: Optional[Tensor] = None,
        min_tokens: int = CONTEXT_TOKENS,
    ) -> Tensor:
        """The DiT's text context: the adapter's output with its padded T5 positions zeroed, zero-padded at the end to
        ``min_tokens`` (ComfyUI pads one caption's output to 512, kohya and diffusers tokenize to 512)."""
        dtype = self.dtype
        out = self.llm_adapter(source.to(dtype), t5_ids, t5_mask, source_mask)
        if t5_mask is not None:
            out = out * t5_mask.to(device=out.device, dtype=out.dtype)[..., None]
        if out.shape[1] < min_tokens:
            out = F.pad(out, (0, 0, 0, min_tokens - out.shape[1]))
        return out

    # ------------------------------------------------------------------ forward
    def unpatchify(self, x: Tensor, frames: int, height: int, width: int) -> Tensor:
        """(B, L, p * p * t * C) -> (B, C, frames * t, height * p, width * p)."""
        p, r = self.params.patch_spatial, self.params.patch_temporal
        b = x.shape[0]
        x = x.view(b, frames, height, width, p, p, r, -1)
        x = x.permute(0, 7, 1, 6, 2, 4, 3, 5)
        return x.reshape(b, x.shape[1], frames * r, height * p, width * p)

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        context: Tensor,
        t5_ids: Optional[Tensor] = None,
        t5_mask: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Velocity (noise - clean) of the latents ``x`` (B, 16, H, W) or (B, 16, 1, H, W) at ``timesteps`` (B,) = sigma
        in 0..1. ``context`` is the text context (B, N, 1024), or with ``t5_ids`` the Qwen3 states the adapter turns into
        one (see ``encode_context``)."""
        if t5_ids is not None:
            context = self.encode_context(context, t5_ids, t5_mask, context_mask)
        squeeze = x.ndim == 4
        if squeeze:
            x = x.unsqueeze(2)
        batch = x.shape[0]
        device = x.device
        dtype = self.dtype

        # the padding-mask channel: zeros (an image fills its whole latent)
        x = x.to(dtype)
        x = torch.cat((x, torch.zeros_like(x[:, :1])), dim=1)
        tokens = self.x_embedder(x)  # (B, T, H, W, D)
        _, frames, height, width, _ = tokens.shape
        tokens = tokens.flatten(1, 3)
        rope_cos, rope_sin = rope_3d(
            frames,
            height,
            width,
            self.params.model_channels // self.params.num_heads,
            (self.params.rope_t_extrapolation_ratio, self.params.rope_h_extrapolation_ratio, self.params.rope_w_extrapolation_ratio),
            device,
        )

        timesteps = torch.as_tensor(timesteps, device=device).reshape(-1).expand(batch)
        emb, adaln_lora = self.t_embedder[1](self.t_embedder[0](timesteps).to(tokens.dtype))
        emb = self.t_embedding_norm(emb)
        context = context.to(dtype)

        if tokens.dtype == torch.float16:
            tokens = tokens.float()  # the residual stream outgrows FP16 (ComfyUI keeps it in fp32 as well)

        attn_params = AttentionParams.create_attention_params(self.attn_mode, self.split_attn)
        for index, block in enumerate(self.blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(index)
            tokens = block(tokens, emb, context, rope_cos, rope_sin, adaln_lora, attn_params)
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.blocks, index)

        # activation CPU offloading returns each checkpointed block's output on the CPU
        tokens = tokens.to(device=device, dtype=context.dtype)
        out = self.final_layer(tokens, emb, adaln_lora)
        out = self.unpatchify(out, frames, height, width)
        return out.squeeze(2) if squeeze else out


def strip_checkpoint_prefix(state_dict: dict) -> dict:
    """The released files' ``net.`` (base, preview) or ``model.diffusion_model.`` (aesthetic, turbo) prefix -> bare
    keys. Derived buffers that some exports carry are dropped. In place; returns the dict."""
    for key in list(state_dict.keys()):
        new_key = key
        for prefix in ("model.diffusion_model.", "net."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
                break
        value = state_dict.pop(key)
        if new_key.startswith("pos_embedder.") or new_key.endswith(".inv_freq"):
            continue
        state_dict[new_key] = value
    return state_dict


def comfy_state_dict(state_dict: dict) -> dict:
    """Bare keys -> the released layout (``net.`` prefix) that ComfyUI, SwarmUI and the trainers load. New dict; the
    tensors are not copied."""
    return {f"net.{key}": value for key, value in state_dict.items()}
