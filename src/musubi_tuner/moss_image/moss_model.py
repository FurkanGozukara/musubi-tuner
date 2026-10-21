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
fused modules would share one DoRA magnitude); the blocks swap to the CPU and checkpoint their activations. The
reference-image (edit) conditioning ai-toolkit adds is not part of the vendor model and is not ported.
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


def attention(q: Tensor, k: Tensor, v: Tensor, rope: Tensor, heads: int, mask: Optional[Tensor]) -> Tensor:
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = rotate(q.reshape(shape), rope).transpose(1, 2)
    k = rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    q, k = q.to(v.dtype), k.to(v.dtype)
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

    def _forward(self, x: Tensor, rope: Tensor, modulation: Tensor, mask: Optional[Tensor]) -> Tensor:
        m = modulation.unbind(1)
        y = self.norm1(x) * (1 + m[1][:, None]) + m[0][:, None]
        q = self.q_norm(self.proj["q"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj["k"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        v = self.proj["v"](y)
        x = x + self.proj["o"](attention(q, k, v, rope, self.heads, mask)) * m[2][:, None]
        y = self.norm2(x) * (1 + m[4][:, None]) + m[3][:, None]
        hidden, gate = self.proj["ffn_up_gate"](y).chunk(2, -1)
        return x + self.proj["ffn_down"](situ(hidden, gate)) * m[5][:, None]

    def forward(self, x: Tensor, rope: Tensor, modulation: Tensor, mask: Optional[Tensor]) -> Tensor:
        return self._run(self._forward, x, rope, modulation, mask)


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

    def _rope(self, lengths: Tensor, height: int, width: int, text_len: int, device) -> Tensor:
        """(B, H*W + S, head_dim) fp32: image tokens at (0, item text length, centered h, centered w), text token i at
        (0, i, 0, 0)."""
        batch = lengths.shape[0]
        image = self._image_coords(height, width, device).unsqueeze(0).repeat(batch, 1, 1)
        image[..., 1] = lengths.float()[:, None]
        text = torch.zeros(batch, text_len, 4, device=device)
        text[..., 1] = torch.arange(text_len, device=device, dtype=torch.float32)
        return self._rope_from_coords(torch.cat((image, text), 1))

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
    ) -> Tensor:
        """Velocity (noise - clean) of ``hidden_states`` (B, 128, H, W) at ``timestep`` (B,) on the 0..1000 scale (1000 =
        pure noise), conditioned on ``encoder_hidden_states`` (B, S, 4096) with ``encoder_attention_mask`` (B, S) bool.
        ``skip_layers``: block indices left out (the negative branch of the vendor's CFG skips block 22)."""
        skip = frozenset(skip_layers or ())
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
        rope = self._rope(lengths, height, width, text_len, device).to(dtype)

        text_mask = joint_mask = None
        if padded:
            text_mask = valid[:, None, None, :]
            live_valid = torch.cat((torch.ones(batch, num_image, dtype=torch.bool, device=device), valid), 1)
            joint_mask = live_valid[:, None, None, :]

        video = self.video_embedder(hidden_states.flatten(2).transpose(1, 2).to(dtype).contiguous())
        text = self.text_embedder.proj(context.to(dtype))
        text_rope = rope[:, num_image:]
        for block in self.text_embedder.blocks:
            text = block(text, text_rope, text_mask)
        # activation CPU offloading returns each checkpointed block's output on the CPU (the next block brings its
        # input back), so the refined text joins the image tokens on the GPU
        x = torch.cat((video, text.to(device)), 1)
        del video, text, context

        for index, layer in enumerate(self.layers):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(index)
            if index not in skip:
                x = layer(x, rope, mod, joint_mask)
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.layers, index)

        x = x[:, :num_image].to(device)  # back on the GPU after activation offloading
        x = self.final_norm_video(x)
        x = x * (1 + head[:, 1, None]) + head[:, 0, None]
        x = self.final_linear_video(x)
        return x.transpose(1, 2).reshape(batch, -1, height, width)


def split_fused_qkv(state_dict: dict) -> dict:
    """ComfyUI / ai-toolkit layout ``layers.N.proj.qkv.weight`` (rows q, k, v) -> ``layers.N.proj.{q,k,v}.weight``.
    In place; the text refiner's ``qkv`` stays fused (it is fused in every MOSS file and not a LoRA target)."""
    for key in [k for k in state_dict if k.startswith("layers.") and k.endswith(".proj.qkv.weight")]:
        base = key[: -len(".qkv.weight")]
        for name, part in zip(("q", "k", "v"), state_dict.pop(key).chunk(3, dim=0)):
            state_dict[f"{base}.{name}.weight"] = part.contiguous()
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
