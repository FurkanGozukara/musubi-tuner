"""FIBO 2.0 DiT (Bria, briaai/Fibo-2.0-dev and -edit-turbo, 8.1B parameters).

44 Z-Image blocks of width 3584 in one stream of ``[gist | image | references]``: the gist tokens come from a two-layer
Perceiver over the first Qwen3-VL bundle of the text (256 to 512 of them, by text length), five injection blocks
(6, 14, 22, 30, 38) also cross-attend to their own bundle, and two noise-refiner blocks see the image (and references)
first. The gist and reference tokens are clean (t = 0 modulation), the image tokens noisy. RoPE over three axes
(32, 48, 48) of theta 256: gist token i at (i + 1, 0, 0), the image on plane 513, reference k on plane 514 + k.

Ported from ai-toolkit's fibo2 transformer (MIT, ostris; a port of diffusers' ``BriaFibo2Transformer2DModel``) with the
same math, in the Comfy-Org/Fibo-2.0 key layout: self-attention q / k / v fused into ``attention.qkv``, cross-attention
and Perceiver k / v into ``to_kv``, SwiGLU w1 / w3 into ``feed_forward.w13`` (hidden width zero-padded to 9728), so the
ComfyUI files load as they are and every LoRA / DoRA module name is one ComfyUI's loader maps
(``lora_unet_layers_0_attention_qkv``). Additions that do not change the outputs: the attention runs through this fork's
dispatcher (every ``--attn_mode``, ``flash_auto`` included) when nothing is masked, the 44 blocks swap to the CPU, and
every block checkpoints (optionally offloading) its activations. A batch whose captions differ in length masks the text
padding, and when their gist counts differ the gist tokens go last with their padding masked, exactly as diffusers does
(identical to one forward per item); masked batches use PyTorch SDPA.
"""

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from musubi_tuner.modules.attention import AttentionParams, attention as dispatch_attention
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig, create_offloader
from musubi_tuner.utils.model_utils import create_cpu_offloading_wrapper

ADALN_DIM = 256
NUM_TRAIN_TIMESTEPS = 1000

# Linear weights of the blocks (the 44 layers, the noise refiner and the Perceiver's layers). The embedders, the
# modulations, the final layer and the norms stay in full precision, as in Comfy-Org's INT8 ConvRot file.
FP8_OPTIMIZATION_TARGET_KEYS = ["layers.", "noise_refiner."]
FP8_OPTIMIZATION_EXCLUDE_KEYS = ["norm", "adaLN_modulation", "x_embedder", "t_embedder", "final_layer", "latent_queries"]


@dataclass
class Fibo2Params:
    dim: int = 3584
    n_heads: int = 28
    n_layers: int = 44
    n_refiner_layers: int = 2
    injection_layers: Tuple[int, ...] = (6, 14, 22, 30, 38)
    cap_feat_dim: int = 7680
    perceiver_layers: int = 2
    max_gist_tokens: int = 512
    ffn_dim: int = 9728
    in_channels: int = 128
    norm_eps: float = 1e-5
    # fixed by the release (diffusers config defaults)
    min_gist_tokens: int = 256
    gist_step: int = 64
    gist_min_text_len: int = 256
    gist_max_text_len: int = 2048
    rope_theta: float = 256.0
    axes_dims: Tuple[int, ...] = (32, 48, 48)

    @property
    def context_dim(self) -> int:
        """Per-token width of the text context: one bundle for the Perceiver and one per injection block."""
        return self.cap_feat_dim * (1 + len(self.injection_layers))


def gist_counts(lengths: Sequence[int], p: Fibo2Params) -> List[int]:
    """Gist tokens per prompt: min_gist up to min_text_len text tokens, rising linearly to max_gist at max_text_len,
    rounded up to gist_step."""
    span = p.gist_max_text_len - p.gist_min_text_len
    counts = []
    for n in lengths:
        ratio = min(1.0, max(0.0, (float(n) - p.gist_min_text_len) / span))
        raw = p.min_gist_tokens + ratio * (p.max_gist_tokens - p.min_gist_tokens)
        quantized = int(math.ceil(raw / p.gist_step)) * p.gist_step
        counts.append(max(p.min_gist_tokens, min(p.max_gist_tokens, quantized)))
    return counts


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        y = F.rms_norm(x.float(), (x.shape[-1],), eps=self.eps)
        return y.to(x.dtype) * self.weight


def apply_rope(x: Tensor, rope: tuple[Tensor, Tensor]) -> Tensor:
    """Interleaved (real, imag) pairs rotated in fp32, as diffusers' complex form."""
    cos, sin = rope
    re, im = x.float().unflatten(-1, (-1, 2)).unbind(-1)
    out = torch.stack((re * cos - im * sin, re * sin + im * cos), -1).flatten(-2)
    return out.type_as(x)


def attend(q: Tensor, k: Tensor, v: Tensor, attn_mode: str, mask: Optional[Tensor]) -> Tensor:
    """(B, L, H, D) q and (B, S, H, D) k / v -> (B, L, H * D). ``mask``: bool (B, 1, 1, S), True = attend; a masked
    attention (a padded batch) runs in PyTorch SDPA, an unmasked one through the fork's dispatcher."""
    k, v = k.to(q.dtype), v.to(q.dtype)
    if mask is not None or attn_mode in (None, "torch", "sdpa"):
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
        return out.transpose(1, 2).flatten(2)
    return dispatch_attention([q, k, v], attn_params=AttentionParams.create_attention_params(attn_mode, False))


def modulate(h: Tensor, values: tuple, span: Optional[tuple[int, int]]) -> Tensor:
    """h * value per token: values = (noisy,) for all tokens, or (noisy, clean) with only tokens [span[0]:span[1]]
    noisy."""
    if span is None:
        return h * values[0][:, None]
    noisy, clean = values
    s, e = span
    return torch.cat((h[:, :s] * clean[:, None], h[:, s:e] * noisy[:, None], h[:, e:] * clean[:, None]), 1)


class JointAttention(nn.Module):
    def __init__(self, dim: int, heads: int, eps: float):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(dim // heads, eps)
        self.k_norm = RMSNorm(dim // heads, eps)

    def forward(self, x: Tensor, rope, mask: Optional[Tensor], attn_mode: str) -> Tensor:
        q, k, v = self.qkv(x).unflatten(-1, (3, self.heads, -1)).unbind(2)
        q = apply_rope(self.q_norm(q), rope)
        k = apply_rope(self.k_norm(k), rope)
        return self.out(attend(q, k, v, attn_mode, mask))


class Attention(nn.Module):
    """No RoPE: the cross-attention to the text, and the Perceiver's self-attention."""

    def __init__(self, dim: int, heads: int, kv_dim: int, eps: float):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(kv_dim, 2 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.norm_q = RMSNorm(dim // heads, eps)
        self.norm_k = RMSNorm(dim // heads, eps)

    def forward(self, x: Tensor, context: Tensor, mask: Optional[Tensor], attn_mode: str) -> Tensor:
        q = self.norm_q(self.to_q(x).unflatten(-1, (self.heads, -1)))
        k, v = self.to_kv(context).unflatten(-1, (2, self.heads, -1)).unbind(2)
        return self.to_out(attend(q, self.norm_k(k), v, attn_mode, mask))


class FeedForward(nn.Module):
    """SwiGLU with w1 / w3 fused into w13 (the hidden width zero-padded to a multiple of 256 in the ComfyUI files)."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.w13(x).chunk(2, -1)
        return self.w2(F.silu(gate) * up)


class _CheckpointMixin:
    gradient_checkpointing = False
    activation_cpu_offloading = False
    attn_mode = "torch"
    # set by the trainable block-swap ring (full fine-tuning): a checkpointed block's backward recompute waits for its
    # weights as the forward does
    recompute_wait: Optional[Callable[[], None]] = None

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def _run(self, fn, *args):
        if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
            if self.activation_cpu_offloading:
                fn = create_cpu_offloading_wrapper(fn, self._weight_device())
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

    def _weight_device(self) -> torch.device:
        raise NotImplementedError


class Fibo2Block(_CheckpointMixin, nn.Module):
    """Z-Image's block; an injection block also cross-attends to its own text bundle (the LoRA target, with the
    Perceiver's layers)."""

    def __init__(self, p: Fibo2Params, injection: bool):
        super().__init__()
        dim, eps = p.dim, p.norm_eps
        self.attention = JointAttention(dim, p.n_heads, eps)
        self.feed_forward = FeedForward(dim, p.ffn_dim)
        self.attention_norm1 = RMSNorm(dim, eps)
        self.ffn_norm1 = RMSNorm(dim, eps)
        self.attention_norm2 = RMSNorm(dim, eps)
        self.ffn_norm2 = RMSNorm(dim, eps)
        self.adaLN_modulation = nn.Sequential(nn.Linear(ADALN_DIM, 4 * dim, bias=True))
        self.cross_attention = None
        if injection:
            self.cross_attn_norm1 = RMSNorm(dim, eps)
            self.cross_attention = Attention(dim, p.n_heads, p.cap_feat_dim, eps)
            self.cross_attn_norm2 = RMSNorm(dim, eps)
            self.cross_attn_gate = nn.Parameter(torch.zeros(1))

    def _weight_device(self) -> torch.device:
        return self.attention_norm1.weight.device

    def _forward(self, x, rope, adaln, span, mask=None, context=None, context_mask=None):
        # adaln: (B, 256) for all tokens, or (2B, 256) = [noisy; clean] with span
        mod = self.adaLN_modulation(adaln)
        parts = (mod,) if span is None else mod.chunk(2, 0)
        chunks = [m.chunk(4, -1) for m in parts]
        scale_msa = [1 + c[0] for c in chunks]
        gate_msa = [c[1].tanh() for c in chunks]
        scale_mlp = [1 + c[2] for c in chunks]
        gate_mlp = [c[3].tanh() for c in chunks]

        y = self.attention(modulate(self.attention_norm1(x), scale_msa, span), rope, mask, self.attn_mode)
        x = x + modulate(self.attention_norm2(y), gate_msa, span)
        if self.cross_attention is not None:
            y = self.cross_attention(self.cross_attn_norm1(x), context, context_mask, self.attn_mode)
            x = x + self.cross_attn_gate.tanh() * self.cross_attn_norm2(y)
        y = self.feed_forward(modulate(self.ffn_norm1(x), scale_mlp, span))
        return x + modulate(self.ffn_norm2(y), gate_mlp, span)

    def forward(self, x, rope, adaln, span, mask=None, context=None, context_mask=None):
        return self._run(self._forward, x, rope, adaln, span, mask, context, context_mask)


class Fibo2PerceiverLayer(_CheckpointMixin, nn.Module):
    def __init__(self, p: Fibo2Params):
        super().__init__()
        dim, eps = p.dim, p.norm_eps
        self.cross_attn_norm = RMSNorm(dim, eps)
        self.cross_attention = Attention(dim, p.n_heads, p.cap_feat_dim, eps)
        self.self_attn_norm = RMSNorm(dim, eps)
        self.self_attention = Attention(dim, p.n_heads, dim, eps)
        self.ffn_norm = RMSNorm(dim, eps)
        self.ffn = FeedForward(dim, p.ffn_dim)

    def _weight_device(self) -> torch.device:
        return self.ffn_norm.weight.device

    def _forward(self, x, context, context_mask, gist_mask):
        x = x + self.cross_attention(self.cross_attn_norm(x), context, context_mask, self.attn_mode)
        y = self.self_attn_norm(x)
        x = x + self.self_attention(y, y, gist_mask, self.attn_mode)
        return x + self.ffn(self.ffn_norm(x))

    def forward(self, x, context, context_mask, gist_mask):
        return self._run(self._forward, x, context, context_mask, gist_mask)


# an injection block's cross-attention: outside the part block swap moves (_SwapUnit), so every swapped unit has one layout
CROSS_ATTENTION_PARTS = ("cross_attn_norm1", "cross_attention", "cross_attn_norm2")


class _SwapUnit(nn.Module):
    """The part of a block that block swap moves: its self-attention, feed-forward, norms and modulation, the same tensors
    in all 44 blocks. The trainable ring (full fine-tuning) and the H2D-only stream copy every swapped block through one
    buffer layout, which the five injection blocks' extra cross-attention would break ("block N is incompatible with the
    trainable ring layout"); their cross-attention (0.16 GB each) stays on the GPU instead. A unit shares its modules with
    its block (not a submodule of the model: names, state dict and LoRA targets are the block's), and the forward runs the
    block through its unit, so the offloaders' backward hooks on the unit fire."""

    def __init__(self, block: nn.Module):
        super().__init__()
        for name, child in block.named_children():
            if name not in CROSS_ATTENTION_PARTS:
                self.add_module(name, child)
        object.__setattr__(self, "_block", block)  # not a child: the cross-attention would join the unit's layout

    def forward(self, *args):
        return self._block(*args)


class Perceiver(nn.Module):
    def __init__(self, p: Fibo2Params):
        super().__init__()
        self.latent_queries = nn.Parameter(torch.zeros(p.max_gist_tokens, p.dim))
        self.layers = nn.ModuleList([Fibo2PerceiverLayer(p) for _ in range(p.perceiver_layers)])
        self.norm_out = RMSNorm(p.dim, p.norm_eps)


class TimestepEmbedder(nn.Module):
    def __init__(self, freq_dim: int = 256, mid_dim: int = 1024, out_dim: int = ADALN_DIM):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, mid_dim, bias=True), nn.SiLU(), nn.Linear(mid_dim, out_dim, bias=True))

    def forward(self, t: Tensor) -> Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat((args.cos(), args.sin()), -1)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(dim, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(ADALN_DIM, dim, bias=True))

    def forward(self, x: Tensor, adaln: Tensor) -> Tensor:
        scale = 1 + self.adaLN_modulation(adaln)
        x = F.layer_norm(x, (x.shape[-1],), eps=1e-6)
        return self.linear(x * scale[:, None])


class Fibo2(nn.Module):
    def __init__(self, params: Optional[Fibo2Params] = None, attn_mode: str = "torch", split_attn: bool = False):
        super().__init__()
        p = params or Fibo2Params()
        if p.dim // p.n_heads != 2 * sum(d // 2 for d in p.axes_dims):
            raise ValueError("FIBO 2.0: the RoPE axes must cover the head dim")
        self.params = p
        self.attn_mode = "torch" if attn_mode in (None, "sdpa") else attn_mode
        self.split_attn = split_attn  # not used: padded batches run masked in PyTorch SDPA
        self.x_embedder = nn.Linear(p.in_channels, p.dim, bias=True)
        self.t_embedder = TimestepEmbedder()
        self.noise_refiner = nn.ModuleList([Fibo2Block(p, False) for _ in range(p.n_refiner_layers)])
        self.perceiver = Perceiver(p)
        self.layers = nn.ModuleList([Fibo2Block(p, i in p.injection_layers) for i in range(p.n_layers)])
        self.final_layer = FinalLayer(p.dim, p.in_channels)
        for block in self._blocks():
            block.attn_mode = self.attn_mode

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        self.blocks_to_swap = None
        self.offloader = None
        self.num_blocks = len(self.layers)
        self._swap_unit_list = None

    def _blocks(self) -> list:
        return list(self.noise_refiner) + list(self.perceiver.layers) + list(self.layers)

    @property
    def device(self):
        return self.x_embedder.weight.device

    @property
    def dtype(self):
        return self.x_embedder.weight.dtype  # never quantized

    # ------------------------------------------------------------------ memory
    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading
        for block in self._blocks():
            block.enable_gradient_checkpointing(activation_cpu_offloading)
        print(f"FIBO 2.0: Gradient checkpointing enabled. Activation CPU offloading: {activation_cpu_offloading}")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        for block in self._blocks():
            block.disable_gradient_checkpointing()
        print("FIBO 2.0: Gradient checkpointing disabled.")

    def enable_block_swap(self, num_blocks: int, config: BlockSwapConfig):
        """Swap ``num_blocks`` of the 44 blocks (at least two stay resident, Guardrail 9); the noise refiner and the
        Perceiver stay on the GPU."""
        assert 0 < num_blocks <= self.num_blocks - 2, f"FIBO 2.0: blocks_to_swap must be 1 to {self.num_blocks - 2}, got {num_blocks}"
        self.blocks_to_swap = num_blocks
        self.offloader = create_offloader("single", self.swap_units(), self.num_blocks, num_blocks, config)
        if getattr(self.offloader, "recompute_requires_wait", False):  # the trainable ring (full fine-tuning)
            for index, layer in enumerate(self.layers):
                layer.recompute_wait = partial(self.offloader.wait_for_block, index)
        print(f"FIBO 2.0: Block swap enabled. Swapping {num_blocks} of {self.num_blocks} blocks.")

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()
            print("FIBO 2.0: Block swap set to forward only.")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()
            print("FIBO 2.0: Block swap set to forward and backward.")

    def swap_units(self) -> list:
        """The 44 blocks as block swap sees them (_SwapUnit: one layout for all)."""
        if self._swap_unit_list is None:
            self._swap_unit_list = [_SwapUnit(block) for block in self.layers]
        return self._swap_unit_list

    def move_to_device_except_swap_blocks(self, device: torch.device):
        # the model is on the CPU: leave the blocks' swap units there (the offloader places them), but every injection
        # block's cross-attention goes to the device, as it never swaps
        if self.blocks_to_swap:
            saved = self.layers
            self.layers = nn.ModuleList()
        self.to(device)
        if self.blocks_to_swap:
            self.layers = saved
            for block in self.layers:
                for name in CROSS_ATTENTION_PARTS:
                    part = getattr(block, name, None)
                    if part is not None:
                        part.to(device)
                gate = block._parameters.get("cross_attn_gate")
                if gate is not None:
                    gate.data = gate.data.to(device)

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.prepare_block_devices_before_forward(self.swap_units())

    # ------------------------------------------------------------------ positions
    def _rope(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        """(L, 3) integer positions -> (cos, sin), each (1, L, 1, head_dim / 2) fp32; phases in fp64 like diffusers'
        precomputed tables."""
        angles = []
        for i, d in enumerate(self.params.axes_dims):
            freqs = 1.0 / (self.params.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float64, device=ids.device) / d))
            angles.append(torch.outer(ids[:, i].double(), freqs).float())
        angles = torch.cat(angles, -1)[None, :, None]
        return angles.cos(), angles.sin()

    @staticmethod
    def _grid(plane: int, height: int, width: int, device) -> Tensor:
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        return torch.stack((torch.full_like(h, plane), h, w), -1).flatten(0, 1)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Optional[Tensor] = None,
        ref_latents: Optional[List[Tensor]] = None,
    ) -> Tensor:
        """Velocity (noise - clean) of ``hidden_states`` (B, 128, H, W), the normalized FLUX.2 latents, at ``timestep``
        (B,) on the 0..1000 scale (1000 = pure noise), conditioned on ``encoder_hidden_states`` (B, T, 46080) with the
        (B, T) bool ``encoder_attention_mask`` of a padded batch, and on clean ``ref_latents`` (each (B, 128, h, w))."""
        p = self.params
        batch, channels, height, width = hidden_states.shape
        device, dtype = hidden_states.device, self.dtype
        num_image = height * width

        t = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1).expand(batch)
        adaln_noisy = self.t_embedder(t)
        adaln_both = torch.cat((adaln_noisy, self.t_embedder(torch.zeros_like(t))))

        # the image on rope plane max_gist + 1 whatever the gist count, reference k on the k-th plane after it
        plane = p.max_gist_tokens + 1
        tokens = [hidden_states.flatten(2).transpose(1, 2)]
        ids = [self._grid(plane, height, width, device)]
        for k, ref in enumerate(ref_latents or []):
            tokens.append(ref.to(device).flatten(2).transpose(1, 2))
            ids.append(self._grid(plane + 1 + k, ref.shape[-2], ref.shape[-1], device))
        x = self.x_embedder(torch.cat(tokens, 1).to(dtype))
        del tokens
        image_rope = self._rope(torch.cat(ids))
        span = (0, num_image) if ref_latents else None
        for block in self.noise_refiner:
            x = block(x, image_rope, adaln_both if span else adaln_noisy, span)
        x = x.to(device)  # back on the GPU after activation offloading

        context = encoder_hidden_states.to(device=device, dtype=dtype)
        mask = encoder_attention_mask
        if mask is not None:
            mask = mask.to(device=device, dtype=torch.bool)
            lengths = mask.sum(1).tolist()
            context_mask = None if all(n == mask.shape[1] for n in lengths) else mask[:, None, None]
        else:
            lengths, context_mask = [context.shape[1]] * batch, None
        bundles = context.chunk(1 + len(p.injection_layers), -1)
        del context

        counts = gist_counts(lengths, p)
        num_gist = max(counts)
        gist_mask = None
        if min(counts) < num_gist:
            gist_mask = torch.arange(num_gist, device=device) < torch.tensor(counts, device=device)[:, None]
        gist = self.perceiver.latent_queries[:num_gist].to(dtype).unsqueeze(0).expand(batch, -1, -1)
        for layer in self.perceiver.layers:
            gist = layer(gist, bundles[0], context_mask, None if gist_mask is None else gist_mask[:, None, None])
        gist = self.perceiver.norm_out(gist.to(device))
        gist_ids = torch.zeros(num_gist, 3, dtype=torch.long, device=device)
        gist_ids[:, 0] = torch.arange(1, num_gist + 1, device=device)
        gist_rope = self._rope(gist_ids)

        # gist first as trained; with padded gists last so the padding ends every row
        if gist_mask is None:
            x = torch.cat((gist, x), 1)
            rope = tuple(torch.cat(r, 1) for r in zip(gist_rope, image_rope))
            span, attn_mask = (num_gist, num_gist + num_image), None
        else:
            attn_mask = torch.cat((torch.ones(batch, x.shape[1], dtype=torch.bool, device=device), gist_mask), 1)[:, None, None]
            x = torch.cat((x, gist), 1)
            rope = tuple(torch.cat(r, 1) for r in zip(image_rope, gist_rope))
            span = (0, num_image)
        del gist, image_rope, gist_rope

        text = {layer_id: bundles[i + 1] for i, layer_id in enumerate(p.injection_layers)}
        units = self.swap_units() if self.blocks_to_swap else None
        for index, block in enumerate(self.layers):
            if units is None:
                x = block(x, rope, adaln_both, span, attn_mask, text.get(index), context_mask)
                continue
            # through the block's swap unit: the offloader's backward hooks sit on the units it swaps
            self.offloader.wait_for_block(index)
            x = units[index](x, rope, adaln_both, span, attn_mask, text.get(index), context_mask)
            self.offloader.submit_move_blocks_forward(units, index)

        x = self.final_layer(x[:, span[0] : span[1]].to(device), adaln_noisy)
        return x.transpose(1, 2).reshape(batch, channels, height, width)


def params_from_shapes(shapes: dict[str, tuple]) -> Fibo2Params:
    """The model's dimensions from a checkpoint's tensor shapes (ComfyUI layout; INT8 ConvRot keeps (out, in))."""
    max_gist, dim = shapes["perceiver.latent_queries"]
    keys = list(shapes)
    n_layers = 1 + max(int(k.split(".")[1]) for k in keys if k.startswith("layers."))
    return Fibo2Params(
        dim=dim,
        n_heads=dim // shapes["layers.0.attention.q_norm.weight"][0],
        n_layers=n_layers,
        n_refiner_layers=1 + max(int(k.split(".")[1]) for k in keys if k.startswith("noise_refiner.")),
        injection_layers=tuple(i for i in range(n_layers) if f"layers.{i}.cross_attn_gate" in shapes),
        cap_feat_dim=shapes["perceiver.layers.0.cross_attention.to_kv.weight"][1],
        perceiver_layers=1 + max(int(k.split(".")[2]) for k in keys if k.startswith("perceiver.layers.")),
        max_gist_tokens=max_gist,
        ffn_dim=shapes["layers.0.feed_forward.w13.weight"][0] // 2,
        in_channels=shapes["x_embedder.weight"][1],
    )
