"""MOSS-Image DiT (OpenMOSS-Team/MOSS-Image-Base / -SFT, 5B).

One stream: image tokens followed by text tokens through 42 blocks that share one AdaLN modulation; the text first goes
through a 2-block refiner. Ported from ai-toolkit's MOSS-Image transformer (ostris, a port of the vendor's
``pipeline_moss_image.py``) with the same math: the image's RoPE coordinates (0, text length, centered h, centered w),
text token i at (0, i, 0, 0), sin/cos of every axis' 16 bands laid out as [sin x 4 axes | cos x 4 axes], the gated
``situ`` activation, fp32 timestep sinusoid and RoPE phases, everything else in the compute dtype. Batches with text of
different lengths run padded with the padding masked out of every attention (identical to one forward per item).

Differences that do not change the outputs: each block's q / k / v are separate Linears (``proj.q`` / ``proj.k`` /
``proj.v``; the ComfyUI files fuse them into ``proj.qkv``, split on load), so a LoRA / DoRA trains one module per
projection under the names ComfyUI maps onto the fused weight's row slices (``lora_unet_layers_0_proj_q``; Guardrail 18:
fused modules would share one DoRA magnitude); the blocks swap to the CPU and checkpoint their activations.

Edit LoRAs (ai-toolkit b7cd8a1 / 88360c6, not part of the vendor model; ComfyUI's ``TextEncodeMOSSImageEdit``, edit_format
"ai-toolkit"): the clean latents of the reference images ride after the text tokens, modulated at t = 0, reference i on
the RoPE axis r = i + 1 (0 for the image and the text) at the image's frame coordinate with its own centered grid.
``isolate_refs`` (ai-toolkit's kv_cache) lets reference queries attend only to reference keys, so the references' keys
and values do not depend on the step and are computed once at inference (``precompute_ref_kv``); without padding the
isolation runs as two unmasked SDPA calls instead of a boolean mask, which would force SDPA off flash attention.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig, create_offloader
from musubi_tuner.utils.model_utils import create_cpu_offloading_wrapper

# Linear weights of the 42 blocks; embedders, the shared modulation, the text refiner and the output head stay in full
# precision (ai-toolkit's quantization exclusions)
FP8_OPTIMIZATION_TARGET_KEYS = ["layers"]
FP8_OPTIMIZATION_EXCLUDE_KEYS = ["norm", "video_embedder", "text_embedder", "time_embedding", "modulation", "final_linear"]

NUM_TRAIN_TIMESTEPS = 1000


@dataclass
class MossImageParams:
    dim: int = 3072
    ffn_dim: int = 8192
    num_heads: int = 24
    num_layers: int = 42
    text_dim: int = 4096
    freq_dim: int = 256
    latent_channels: int = 128
    rope_base: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        y = F.rms_norm(x.float(), (x.shape[-1],), eps=1e-6)
        return y.to(x.dtype) * self.weight


def situ(hidden: Tensor, gate: Tensor) -> Tensor:
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def rotate(x: Tensor, rope: Tensor) -> Tensor:
    """x (B, L, H, D), rope (B, L, D) = [sin | cos] halves of D / 2 each."""
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    rope: Tensor,
    heads: int,
    mask: Optional[Tensor],
    isolate_split: Optional[int] = None,
    kv_capture: Optional[list] = None,
    kv_cache: Optional[tuple[Tensor, Tensor]] = None,
) -> Tensor:
    """Joint attention. ``isolate_split``: tokens from there on are references that attend only to each other (the
    others attend to every key), as two unmasked SDPA calls. ``kv_capture`` (a list) receives this call's post-RoPE
    keys and values; ``kv_cache`` holds the references' keys and values, attended to after this call's own keys."""
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = rotate(q.reshape(shape), rope).transpose(1, 2)
    k = rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    if kv_capture is not None:
        # copies: views would pin the whole projection output for the rest of the sampling
        kv_capture.append((k.clone(memory_format=torch.contiguous_format), v.clone(memory_format=torch.contiguous_format)))
    if kv_cache is not None:
        cached_k, cached_v = kv_cache
        if cached_k.shape[0] != k.shape[0]:  # one set of references for the whole batch
            cached_k, cached_v = cached_k.expand(k.shape[0], -1, -1, -1), cached_v.expand(k.shape[0], -1, -1, -1)
        k = torch.cat((k, cached_k.to(device=k.device, dtype=k.dtype)), 2)
        v = torch.cat((v, cached_v.to(device=v.device, dtype=v.dtype)), 2)
    q, k = q.to(v.dtype), k.to(v.dtype)
    if isolate_split is not None:
        live = F.scaled_dot_product_attention(q[:, :, :isolate_split], k, v)
        ref = F.scaled_dot_product_attention(q[:, :, isolate_split:], k[:, :, isolate_split:], v[:, :, isolate_split:])
        out = torch.cat((live, ref), 2)
    else:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out.transpose(1, 2).flatten(2)


def rope_bands(device, head_dim: int, base: float) -> Tensor:
    # built on the CPU in fp32 like the vendor's training, then moved
    num_bands = head_dim // 8
    exponent = torch.arange(num_bands, dtype=torch.float32) / num_bands
    return (1 / (base**exponent)).to(device)


class _CheckpointMixin:
    gradient_checkpointing = False
    activation_cpu_offloading = False
    # set by the trainable block-swap ring (full fine-tuning): it binds a block's weights to a GPU slot only when asked,
    # so the backward recompute of a checkpointed block waits for its slot as the forward does
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
                fn = create_cpu_offloading_wrapper(fn, self.norm1.weight.device)
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


class MossTextBlock(_CheckpointMixin, nn.Module):
    """The text refiner's block (fused qkv with bias, as in every MOSS file; not a LoRA target)."""

    def __init__(self, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def _forward(self, x: Tensor, rope: Tensor, mask: Optional[Tensor]) -> Tensor:
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(attention(q, k, v, rope, self.heads, mask))
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))

    def forward(self, x: Tensor, rope: Tensor, mask: Optional[Tensor]) -> Tensor:
        return self._run(self._forward, x, rope, mask)


class MossTextProjection(nn.Module):
    def __init__(self, text_dim: int, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([MossTextBlock(dim, ffn_dim, heads) for _ in range(2)])


class MossImageBlock(_CheckpointMixin, nn.Module):
    """One of the 42 joint blocks (the LoRA target). ``modulation`` (B, 6, D): shift / scale / gate of the attention,
    then of the feed-forward."""

    def __init__(self, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.proj = nn.ModuleDict(
            {
                "q": nn.Linear(dim, dim, bias=False),
                "k": nn.Linear(dim, dim, bias=False),
                "v": nn.Linear(dim, dim, bias=False),
                "o": nn.Linear(dim, dim, bias=False),
                "ffn_up_gate": nn.Linear(dim, 2 * ffn_dim, bias=False),
                "ffn_down": nn.Linear(ffn_dim, dim, bias=False),
            }
        )

    def _forward(
        self,
        x: Tensor,
        rope: Tensor,
        modulation,
        mask: Optional[Tensor],
        isolate_split: Optional[int] = None,
        kv_capture: Optional[list] = None,
        kv_cache: Optional[tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        # modulation (B, 6, D), or (mod, ref_mod, split) for edit LoRAs: tokens from split on are clean references,
        # modulated at t = 0
        if isinstance(modulation, tuple):
            mod, ref_mod, split = modulation
            m, r = mod.unbind(1), ref_mod.unbind(1)

            def affine(h, scale, shift):
                return torch.cat(
                    (h[:, :split] * (1 + m[scale][:, None]) + m[shift][:, None], h[:, split:] * (1 + r[scale][:, None]) + r[shift][:, None]), 1
                )

            def gate(h, index):
                return torch.cat((h[:, :split] * m[index][:, None], h[:, split:] * r[index][:, None]), 1)

        else:
            m = modulation.unbind(1)

            def affine(h, scale, shift):
                return h * (1 + m[scale][:, None]) + m[shift][:, None]

            def gate(h, index):
                return h * m[index][:, None]

        y = affine(self.norm1(x), 1, 0)
        q = self.q_norm(self.proj["q"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj["k"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        v = self.proj["v"](y)
        x = x + gate(self.proj["o"](attention(q, k, v, rope, self.heads, mask, isolate_split, kv_capture, kv_cache)), 2)
        y = affine(self.norm2(x), 4, 3)
        hidden, ffn_gate = self.proj["ffn_up_gate"](y).chunk(2, -1)
        return x + gate(self.proj["ffn_down"](situ(hidden, ffn_gate)), 5)

    def forward(
        self,
        x: Tensor,
        rope: Tensor,
        modulation,
        mask: Optional[Tensor],
        isolate_split: Optional[int] = None,
        kv_capture: Optional[list] = None,
        kv_cache: Optional[tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        if kv_capture is not None or kv_cache is not None:  # inference only: a recompute would capture twice
            return self._forward(x, rope, modulation, mask, isolate_split, kv_capture, kv_cache)
        return self._run(self._forward, x, rope, modulation, mask, isolate_split)


class MossImage(nn.Module):
    def __init__(self, params: Optional[MossImageParams] = None, attn_mode: str = "torch", split_attn: bool = False):
        super().__init__()
        params = params or MossImageParams()
        if params.dim % params.num_heads or (params.dim // params.num_heads) % 8:
            raise ValueError("dim must divide into heads with head_dim divisible by 8")
        self.params = params
        self.attn_mode = attn_mode  # PyTorch SDPA, as the vendor (its masks rule out the flash kernels' fast paths)
        self.split_attn = split_attn
        dim, ffn_dim, heads = params.dim, params.ffn_dim, params.num_heads
        self.video_embedder = nn.Linear(params.latent_channels, dim)
        self.text_embedder = MossTextProjection(params.text_dim, dim, ffn_dim, heads)
        self.time_embedding = nn.Sequential(nn.Linear(params.freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([MossImageBlock(dim, ffn_dim, heads) for _ in range(params.num_layers)])
        self.final_norm_video = RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, params.latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        self.blocks_to_swap = None
        self.offloader = None
        self.num_blocks = len(self.layers)
        self._bands = {}

    @property
    def device(self):
        return self.final_norm_video.weight.device

    @property
    def dtype(self):
        return self.final_norm_video.weight.dtype

    # ------------------------------------------------------------------ memory
    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading
        for block in list(self.layers) + list(self.text_embedder.blocks):
            block.enable_gradient_checkpointing(activation_cpu_offloading)
        print(f"MOSS-Image: Gradient checkpointing enabled. Activation CPU offloading: {activation_cpu_offloading}")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        for block in list(self.layers) + list(self.text_embedder.blocks):
            block.disable_gradient_checkpointing()
        print("MOSS-Image: Gradient checkpointing disabled.")

    def enable_block_swap(self, num_blocks: int, config: BlockSwapConfig):
        """Swap ``num_blocks`` of the 42 blocks (at least two stay resident, Guardrail 9)."""
        assert 0 < num_blocks <= self.num_blocks - 2, f"MOSS-Image: blocks_to_swap must be 1 to {self.num_blocks - 2}, got {num_blocks}"
        self.blocks_to_swap = num_blocks
        self.offloader = create_offloader("single", self.layers, self.num_blocks, num_blocks, config)
        if getattr(self.offloader, "recompute_requires_wait", False):  # the trainable ring (full fine-tuning)
            for index, layer in enumerate(self.layers):
                layer.recompute_wait = partial(self.offloader.wait_for_block, index)
        print(f"MOSS-Image: Block swap enabled. Swapping {num_blocks} of {self.num_blocks} blocks.")

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()
            print("MOSS-Image: Block swap set to forward only.")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()
            print("MOSS-Image: Block swap set to forward and backward.")

    def move_to_device_except_swap_blocks(self, device: torch.device):
        # the model is on the CPU: leave the swapped blocks there
        if self.blocks_to_swap:
            saved = self.layers
            self.layers = nn.ModuleList()
        self.to(device)
        if self.blocks_to_swap:
            self.layers = saved

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.prepare_block_devices_before_forward(self.layers)

    # ------------------------------------------------------------------ embeddings
    def _rope_from_coords(self, coords: Tensor) -> Tensor:
        """(B, L, 4) fp32 (r, f, h, w) coordinates -> (B, L, head_dim) rope."""
        device = coords.device
        bands = self._bands.get(device)
        if bands is None:
            bands = rope_bands(device, self.params.dim // self.params.num_heads, self.params.rope_base)
            self._bands[device] = bands
        phase = coords.unsqueeze(-1) * bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2)

    @staticmethod
    def _image_coords(height: int, width: int, device) -> Tensor:
        h, w = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        coords = torch.zeros(height * width, 4, device=device)
        coords[:, 2] = (h - (height - 1) / 2).flatten()
        coords[:, 3] = (w - (width - 1) / 2).flatten()
        return coords

    def _rope(self, lengths: Tensor, height: int, width: int, text_len: int, device, image_at_zero: bool = False) -> Tensor:
        """(B, H*W + S, head_dim) fp32: image tokens at (0, item text length, centered h, centered w), text token i at
        (0, i, 0, 0). ``image_at_zero`` shifts both by minus the item's text length (image f = 0, the text before it), the
        layout of references whose keys were computed at f = 0 (precompute_ref_kv, ComfyUI); attention sees only the
        differences of positions, so the outputs are the same."""
        batch = lengths.shape[0]
        image = self._image_coords(height, width, device).unsqueeze(0).repeat(batch, 1, 1)
        image[..., 1] = lengths.float()[:, None]
        text = torch.zeros(batch, text_len, 4, device=device)
        text[..., 1] = torch.arange(text_len, device=device, dtype=torch.float32)
        if image_at_zero:
            image[..., 1] -= lengths.float()[:, None]
            text[..., 1] -= lengths.float()[:, None]
        return self._rope_from_coords(torch.cat((image, text), 1))

    def _pack_refs(self, ref_latents, frame: Tensor, device, dtype) -> tuple[Tensor, Tensor, Tensor]:
        """Per-sample lists of clean (C, h, w) reference latents -> right-padded tokens (B, R, C), their rope (B, R,
        head_dim) and the (B, R) mask of real tokens. Reference i sits on the RoPE axis r = i + 1 (0 for the image and the
        text) at frame coordinate ``frame[b]`` (the image's: the item's text length) with its own centered grid."""
        tokens, coords = [], []
        for refs, f in zip(ref_latents, frame.tolist()):
            sample_tokens, sample_coords = [], []
            for index, latent in enumerate(refs or ()):
                _, h, w = latent.shape
                sample_tokens.append(latent.flatten(1).transpose(0, 1))
                c = self._image_coords(h, w, device)
                c[:, 0] = index + 1
                c[:, 1] = f
                sample_coords.append(c)
            if sample_tokens:
                tokens.append(torch.cat(sample_tokens).to(device=device, dtype=dtype))
                coords.append(torch.cat(sample_coords))
            else:
                tokens.append(torch.zeros(0, self.params.latent_channels, device=device, dtype=dtype))
                coords.append(torch.zeros(0, 4, device=device))
        num_ref = max(t.shape[0] for t in tokens)
        valid = torch.stack([torch.arange(num_ref, device=device) < t.shape[0] for t in tokens])
        tokens = torch.stack([F.pad(t, (0, 0, 0, num_ref - t.shape[0])) for t in tokens])
        coords = torch.stack([F.pad(c, (0, 0, 0, num_ref - c.shape[0])) for c in coords])
        return tokens, self._rope_from_coords(coords), valid

    def _embed_refs(self, tokens: Tensor) -> Tensor:
        # a control adapter's widened patch embedder takes [latent | maps]: references carry no maps (ComfyUI pads too)
        extra = getattr(self.video_embedder, "in_features", tokens.shape[-1]) - tokens.shape[-1]
        if extra > 0:
            tokens = F.pad(tokens, (0, extra))
        return self.video_embedder(tokens.contiguous())

    @torch.no_grad()
    def precompute_ref_kv(self, ref_latents) -> dict:
        """Every block's keys and values of the references alone (isolated references at t = 0 depend on nothing else),
        for sampling: pass the result as ``ref_kv_cache``. The references sit at f = 0, so the forward that uses them puts
        the image at f = 0 as well (ComfyUI's ``preprocess_refs``); one cache serves both CFG branches, as in ComfyUI."""
        device, dtype = self.device, self.dtype
        frame = torch.zeros(len(ref_latents), device=device)
        tokens, rope, valid = self._pack_refs(ref_latents, frame, device, dtype)
        mod, _ = self._time_modulation(torch.zeros(len(ref_latents), device=device), dtype)
        mask = None if bool(valid.all()) else (valid[:, None, :, None] & valid[:, None, None, :]) | torch.eye(
            valid.shape[1], dtype=torch.bool, device=device
        )
        x = self._embed_refs(tokens)
        kv = {}
        for index, layer in enumerate(self.layers):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(index)
            slot = []
            x = layer(x, rope.to(dtype), mod, mask, kv_capture=slot)
            kv[index] = slot[0]
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.layers, index)
        return {"kv": kv, "mask": valid}

    def _time_modulation(self, times: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        half = self.params.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(dtype)).float()
        mod = self.shared_block_modulation(time.to(dtype)).view(times.shape[0], 6, -1)
        return mod, time

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Optional[Tensor] = None,
        skip_layers: Optional[Iterable[int]] = None,
        ref_latents=None,
        isolate_refs: bool = False,
        ref_kv_cache: Optional[dict] = None,
    ) -> Tensor:
        """Velocity (noise - clean) of ``hidden_states`` (B, 128, H, W) at ``timestep`` (B,) on the 0..1000 scale (1000 =
        pure noise), conditioned on ``encoder_hidden_states`` (B, S, 4096) with ``encoder_attention_mask`` (B, S) bool.
        ``skip_layers``: block indices left out (the negative branch of the vendor's CFG skips block 22).

        Edit LoRAs: ``ref_latents`` per sample a list of clean (128, h, w) reference latents, appended after the text at
        t = 0; ``isolate_refs`` keeps the reference queries on the reference keys (ai-toolkit's kv_cache);
        ``ref_kv_cache`` (inference, from precompute_ref_kv) replaces ``ref_latents`` with the references' keys and values."""
        skip = frozenset(skip_layers or ())
        reuse_refs = ref_kv_cache is not None
        if reuse_refs and ref_kv_cache.get("kv") is None:
            raise ValueError("ref_kv_cache must come from precompute_ref_kv")
        batch, _, height, width = hidden_states.shape
        num_image = height * width
        device = hidden_states.device
        dtype = self.dtype

        context = encoder_hidden_states
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(context.shape[:2], dtype=torch.bool, device=device)
        mask = mask.to(device=device, dtype=torch.bool)
        lengths = mask.sum(1)
        if bool((lengths == 0).any()):
            raise ValueError("each sample needs at least one valid text token")
        text_len = int(lengths.max())
        padded = bool((lengths < text_len).any())
        if padded or not bool(mask[:, :text_len].all()):
            # valid tokens first, in order (the vendor selects them by mask)
            order = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)
            context = torch.gather(context, 1, order[..., None].expand(-1, -1, context.shape[-1]))
        context = context[:, :text_len]
        valid = torch.arange(text_len, device=device)[None] < lengths[:, None]

        times = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1).expand(batch)
        mod, time = self._time_modulation(times, dtype)
        head = self.video_head_modulation(time.to(dtype)).view(batch, 2, -1)
        rope = self._rope(lengths, height, width, text_len, device, image_at_zero=reuse_refs).to(dtype)
        num_live = num_image + text_len

        num_ref, ref_valid, modulation, ref_tokens = 0, None, mod, None
        if reuse_refs:
            ref_valid = ref_kv_cache["mask"].to(device)
            if ref_valid.shape[0] != batch:  # one set of references for the whole batch
                ref_valid = ref_valid.expand(batch, -1)
        elif ref_latents is not None and any(len(refs or ()) > 0 for refs in ref_latents):
            if len(ref_latents) != batch:
                raise ValueError(f"ref_latents holds {len(ref_latents)} samples for a batch of {batch}")
            ref_tokens, ref_rope, ref_valid = self._pack_refs(ref_latents, lengths, device, dtype)
            num_ref = ref_tokens.shape[1]
            ref_mod, _ = self._time_modulation(torch.zeros_like(times), dtype)
            modulation = (mod, ref_mod, num_live)
            rope = torch.cat((rope, ref_rope.to(dtype)), 1)

        text_mask = joint_mask = isolate_split = None
        if padded:
            text_mask = valid[:, None, None, :]
        live_valid = torch.cat((torch.ones(batch, num_image, dtype=torch.bool, device=device), valid), 1)
        if num_ref and isolate_refs:
            key_valid = torch.cat((live_valid, ref_valid), 1)
            if bool(key_valid.all()):
                isolate_split = num_live  # no padding anywhere: two unmasked attention calls
            else:
                is_ref = torch.arange(num_live + num_ref, device=device) >= num_live
                allowed = key_valid[:, None, :] & (~is_ref[:, None] | is_ref[None, :])[None]
                # every query keeps at least itself (a padded row would be all masked)
                joint_mask = (allowed | torch.eye(num_live + num_ref, dtype=torch.bool, device=device))[:, None]
        elif num_ref or reuse_refs:
            # image and text queries see every real reference key (cached ones included)
            key_valid = torch.cat((live_valid, ref_valid), 1)
            if not bool(key_valid.all()):
                joint_mask = key_valid[:, None, None, :]
        elif padded:
            joint_mask = live_valid[:, None, None, :]

        video = self.video_embedder(hidden_states.flatten(2).transpose(1, 2).to(dtype).contiguous())
        text = self.text_embedder.proj(context.to(dtype))
        text_rope = rope[:, num_image:num_live]
        for block in self.text_embedder.blocks:
            text = block(text, text_rope, text_mask)
        # activation CPU offloading returns each checkpointed block's output on the CPU (the next block brings its
        # input back), so the refined text joins the image tokens on the GPU
        parts = [video, text.to(device)]
        if num_ref:
            parts.append(self._embed_refs(ref_tokens))
        x = torch.cat(parts, 1)
        del video, text, context, parts, ref_tokens

        for index, layer in enumerate(self.layers):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(index)
            if index not in skip:
                if reuse_refs:
                    x = layer(x, rope, modulation, joint_mask, kv_cache=ref_kv_cache["kv"][index])
                else:
                    x = layer(x, rope, modulation, joint_mask, isolate_split)
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.layers, index)

        x = x[:, :num_image].to(device)  # back on the GPU after activation offloading
        x = self.final_norm_video(x)
        x = x * (1 + head[:, 1, None]) + head[:, 0, None]
        x = self.final_linear_video(x)
        return x.transpose(1, 2).reshape(batch, -1, height, width)


def split_fused_qkv(state_dict: dict) -> dict:
    """ComfyUI / ai-toolkit layout ``layers.N.proj.qkv.weight`` (rows q, k, v) -> ``layers.N.proj.{q,k,v}.weight``.
    In place; the text refiner's ``qkv`` stays fused (it is fused in every MOSS file and not a LoRA target). In a ComfyUI
    INT8 ConvRot file the per-row ``weight_scale`` splits with the rows and q, k and v each get the ``comfy_quant`` spec:
    the rotation mixes input columns only, so the rows stay independent."""
    for suffix in (".weight", ".weight_scale", ".comfy_quant"):
        for key in [k for k in state_dict if k.startswith("layers.") and k.endswith(".proj.qkv" + suffix)]:
            base = key[: -len(".qkv" + suffix)]
            tensor = state_dict.pop(key)
            parts = (tensor, tensor.clone(), tensor.clone()) if suffix == ".comfy_quant" else tensor.chunk(3, dim=0)
            for name, part in zip(("q", "k", "v"), parts):
                state_dict[f"{base}.{name}{suffix}"] = part.contiguous()
    state_dict.pop("rope.bands", None)  # a derived constant, recomputed in fp32
    return state_dict


def fuse_qkv(state_dict: dict, params: Optional[MossImageParams] = None) -> dict:
    """The inverse of ``split_fused_qkv``: ``layers.N.proj.{q,k,v}.weight`` -> ``layers.N.proj.qkv.weight`` (rows q, k, v)
    and the ``rope.bands`` constant (FP32), i.e. the ComfyUI / vendor layout that ``load_dit``, ComfyUI and SwarmUI read.
    Returns a new dict; the tensors are not copied."""
    params = params or MossImageParams()
    out = {}
    for key, value in state_dict.items():
        if key.startswith("layers.") and key.endswith((".proj.k.weight", ".proj.v.weight")):
            continue
        if key.startswith("layers.") and key.endswith(".proj.q.weight"):
            base = key[: -len(".q.weight")]
            out[f"{base}.qkv.weight"] = torch.cat([state_dict[f"{base}.{name}.weight"] for name in ("q", "k", "v")], dim=0)
            continue
        out[key] = value
    out["rope.bands"] = rope_bands("cpu", params.dim // params.num_heads, params.rope_base)
    return out
