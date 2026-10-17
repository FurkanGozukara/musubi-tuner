"""FLUX.3 [klein] image DiT (Black Forest Labs, checkpoint metadata ``f3-image-small``, ~5.4B parameters).

A joint single-sequence DiT. Each content mode (``image``: the latents being denoised, ``image_cond``: clean reference
images for editing) and the text context get their own input projection, ``depth`` early self-attention blocks and
modulation tables; the active modes and the text are then concatenated for ``depth_single_blocks`` joint blocks, and
the image tokens leave through their own final layer. Text is the stack of eight Qwen3-VL-4B hidden layers
(``context_in_dim`` 20480) and runs at timestep 0, as do the reference tokens; there is no pooled vector and no text
attention mask. Image tokens carry 4-axis RoPE ids ``(t, h * 0.5, w * 0.5, l)``, references sit at ``l = -1, -2, ...``
and text tokens at ``l = 0 .. L - 1``.

Module names follow BFL's checkpoint and ComfyUI's ``comfy/ldm/flux3`` model, so the weights load without renaming and
LoRA keys (``lora_unet_single_blocks_0_q_proj``) are the ones ComfyUI maps. Ported from ai-toolkit's flux3 transformer
(MIT, ostris) onto this fork's FLUX.2 building blocks (RoPE, QK RMSNorm, modulation, final layer and the shared attention
dispatcher, so every ``--attn_mode`` including ``flash_auto`` works). Two simplifications that do not change the image
outputs: a mode without tokens is skipped instead of running a two-token dummy (its stream never reaches the joint
blocks), and every token of a mode shares the mode's timestep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from musubi_tuner.flux_2.flux2_models import (
    EmbedND,
    LastLayer,
    MLPEmbedder,
    Modulation,
    QKNorm,
    SiLUActivation,
    attention,
    timestep_embedding,
)
from musubi_tuner.modules.attention import AttentionParams
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig, create_offloader
from musubi_tuner.utils.model_utils import create_cpu_offloading_wrapper

# Linear weights of the blocks; the small modulation / embedding / output layers stay in full precision
FP8_OPTIMIZATION_TARGET_KEYS = ["content_mode_blocks", "txt_mode_blocks", "single_blocks"]
FP8_OPTIMIZATION_EXCLUDE_KEYS = ["norm", "pe_embedder", "time_in", "_modulation", "emb_in", "txt_in", "final_layer"]

CONTENT_MODES = ("image", "image_cond")
IMAGE_ROPE_SCALE = 0.5


def image_ids(batch_size: int, height: int, width: int, index: int, device, rope_scale: float = IMAGE_ROPE_SCALE) -> Tensor:
    """(B, h * w, 4) RoPE ids (t, row * 0.5, col * 0.5, l) of an h x w token grid, row-major like the packed tokens;
    the target image uses l = 0, reference image i uses l = -1 - i."""
    ids = torch.zeros(height, width, 4, device=device, dtype=torch.float32)
    ids[..., 1] = torch.arange(height, device=device, dtype=torch.float32)[:, None] * rope_scale
    ids[..., 2] = torch.arange(width, device=device, dtype=torch.float32)[None, :] * rope_scale
    ids[..., 3] = index
    return ids.flatten(0, 1).unsqueeze(0).expand(batch_size, -1, -1)


def text_ids(batch_size: int, length: int, device) -> Tensor:
    """(B, L, 4) RoPE ids of the text tokens: (0, 0, 0, position)."""
    ids = torch.zeros(length, 4, device=device, dtype=torch.float32)
    ids[:, 3] = torch.arange(length, device=device, dtype=torch.float32)
    return ids.unsqueeze(0).expand(batch_size, -1, -1)


@dataclass
class Flux3Params:
    in_channels: int = 128  # packed FLUX.2 VAE latents (32 channels x 2 x 2), for image and image_cond
    context_in_dim: int = 20480  # 8 Qwen3-VL hidden layers x 2560
    hidden_size: int = 3072
    num_heads: int = 24
    depth: int = 5
    depth_single_blocks: int = 28
    axes_dim: list[int] = field(default_factory=lambda: [32, 32, 32, 32])
    theta: int = 10000
    mlp_ratio: float = 3.0
    image_rope_scale: float = IMAGE_ROPE_SCALE


class _Flux3BlockBase(nn.Module):
    """Parallel attention + SwiGLU MLP under one adaLN and one gate (the per-mode early blocks and the joint blocks
    share this layout and their parameter names). Each kind runs through ``forward`` (``__call__``), so the block-swap
    offloaders' backward hooks fire."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_in = nn.Linear(hidden_size, 2 * self.mlp_hidden_dim, bias=False)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_out = nn.Linear(self.mlp_hidden_dim, hidden_size, bias=False)

        self.norm = QKNorm(head_dim)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = SiLUActivation()

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def _attend(self, x_mod: Tensor, pe: Tensor, attn_params: AttentionParams) -> Tensor:
        q = rearrange(self.q_proj(x_mod), "B L (H D) -> B H L D", H=self.num_heads)
        k = rearrange(self.k_proj(x_mod), "B L (H D) -> B H L D", H=self.num_heads)
        v = rearrange(self.v_proj(x_mod), "B L (H D) -> B H L D", H=self.num_heads)
        mlp = self.mlp_in(x_mod)
        del x_mod
        q, k = self.norm(q, k, v)
        attn = attention([q, k, v], pe, attn_params)  # (B, L, H * D)
        del q, k, v
        return self.attn_out(attn) + self.mlp_out(self.mlp_act(mlp))

    def _checkpointed(self, fn, *args):
        if self.training and self.gradient_checkpointing:
            if self.activation_cpu_offloading:
                fn = create_cpu_offloading_wrapper(fn, self.q_proj.weight.device)
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)


class Flux3ModeBlock(_Flux3BlockBase):
    """Early block: self-attention within one stream (an image mode or the text)."""

    def _forward(self, x: Tensor, pe: Tensor, mod: tuple[Tensor, Tensor, Tensor], attn_params: AttentionParams) -> Tensor:
        shift, scale, gate = mod
        x_mod = (1 + scale) * self.pre_norm(x) + shift
        return x + gate * self._attend(x_mod, pe, attn_params)

    def forward(self, x: Tensor, pe: Tensor, mod, attn_params: AttentionParams) -> Tensor:
        return self._checkpointed(self._forward, x, pe, mod, attn_params)


class Flux3JointBlock(_Flux3BlockBase):
    """Joint block: the text and every active image mode in one attention sequence (text first), each stream
    modulated with its own table."""

    def _forward(
        self,
        img: Tensor,
        txt: Tensor,
        pe: Tensor,
        mod_img: tuple[Tensor, Tensor, Tensor],
        mod_txt: tuple[Tensor, Tensor, Tensor],
        attn_params: AttentionParams,
    ) -> tuple[Tensor, Tensor]:
        shift_img, scale_img, gate_img = mod_img
        shift_txt, scale_txt, gate_txt = mod_txt
        img_mod = (1 + scale_img) * self.pre_norm(img) + shift_img
        txt_mod = (1 + scale_txt) * self.pre_norm(txt) + shift_txt
        out = self._attend(torch.cat((txt_mod, img_mod), dim=1), pe, attn_params)
        del img_mod, txt_mod
        txt_out, img_out = torch.split(out, [txt.shape[1], img.shape[1]], dim=1)
        return img + gate_img * img_out, txt + gate_txt * txt_out

    def forward(self, img: Tensor, txt: Tensor, pe: Tensor, mod_img, mod_txt, attn_params: AttentionParams):
        return self._checkpointed(self._forward, img, txt, pe, mod_img, mod_txt, attn_params)


class Flux3(nn.Module):
    def __init__(self, params: Optional[Flux3Params] = None, attn_mode: str = "torch", split_attn: bool = False):
        super().__init__()
        params = params or Flux3Params()
        self.params = params
        self.attn_mode = attn_mode
        self.split_attn = split_attn
        self.in_channels = params.in_channels
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.depth = params.depth
        self.image_rope_scale = params.image_rope_scale
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(f"hidden_size {params.hidden_size} must be divisible by num_heads {params.num_heads}")
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"axes_dim {params.axes_dim} must add up to the head dim {pe_dim}")

        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=list(params.axes_dim))
        self.emb_in = nn.ModuleDict({m: nn.Linear(params.in_channels, params.hidden_size, bias=False) for m in CONTENT_MODES})
        self.txt_in = nn.Linear(params.context_in_dim, params.hidden_size, bias=False)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=params.hidden_size, disable_bias=True)

        def modulation() -> Modulation:
            return Modulation(params.hidden_size, double=False, disable_bias=True)

        self.early_stream_modulations = nn.ModuleDict({m: modulation() for m in CONTENT_MODES + ("txt",)})
        self.single_stream_modulations = nn.ModuleDict({m: modulation() for m in CONTENT_MODES + ("txt",)})

        def mode_block() -> Flux3ModeBlock:
            return Flux3ModeBlock(params.hidden_size, params.num_heads, params.mlp_ratio)

        self.content_mode_blocks = nn.ModuleDict({m: nn.ModuleList([mode_block() for _ in range(params.depth)]) for m in CONTENT_MODES})
        self.txt_mode_blocks = nn.ModuleList([mode_block() for _ in range(params.depth)])
        self.single_blocks = nn.ModuleList(
            [Flux3JointBlock(params.hidden_size, params.num_heads, params.mlp_ratio) for _ in range(params.depth_single_blocks)]
        )
        self.final_layer = nn.ModuleDict({"image": LastLayer(params.hidden_size, params.in_channels)})

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        self.blocks_to_swap = None
        self.offloader_single = None
        self.num_single_blocks = len(self.single_blocks)
        # text-to-image training never runs the reference stream: its blocks can stay on the CPU (keep_image_cond_on_cpu)
        self.image_cond_on_cpu = False

    def get_model_type(self) -> str:
        return "flux_3"

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    # ------------------------------------------------------------------ memory management
    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading
        self.time_in.enable_gradient_checkpointing()
        for block in self._all_blocks():
            block.enable_gradient_checkpointing(activation_cpu_offloading)
        print(f"FLUX.3: Gradient checkpointing enabled. Activation CPU offloading: {activation_cpu_offloading}")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        self.time_in.disable_gradient_checkpointing()
        for block in self._all_blocks():
            block.disable_gradient_checkpointing()
        print("FLUX.3: Gradient checkpointing disabled.")

    def _all_blocks(self) -> list[_Flux3BlockBase]:
        blocks = [b for m in CONTENT_MODES for b in self.content_mode_blocks[m]]
        return blocks + list(self.txt_mode_blocks) + list(self.single_blocks)

    def enable_block_swap(self, num_blocks: int, config: BlockSwapConfig):
        """Swap ``num_blocks`` of the 28 joint blocks (at least two stay resident, Guardrail 9)."""
        assert 0 < num_blocks <= self.num_single_blocks - 2, (
            f"FLUX.3: blocks_to_swap must be 1 to {self.num_single_blocks - 2}, got {num_blocks}"
        )
        self.blocks_to_swap = num_blocks
        self.offloader_single = create_offloader("single", self.single_blocks, self.num_single_blocks, num_blocks, config)
        print(f"FLUX.3: Block swap enabled. Swapping {num_blocks} of {self.num_single_blocks} joint blocks.")

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader_single.set_forward_only(True)
            self.prepare_block_swap_before_forward()
            print("FLUX.3: Block swap set to forward only.")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader_single.set_forward_only(False)
            self.prepare_block_swap_before_forward()
            print("FLUX.3: Block swap set to forward and backward.")

    def keep_image_cond_on_cpu(self, enabled: bool = True):
        """Text-to-image runs: the reference stream (five early blocks, ~1.2 GB in BF16) is never used, so it is not
        moved to the GPU by ``move_to_device_except_swap_blocks``."""
        self.image_cond_on_cpu = enabled

    def move_to_device_except_swap_blocks(self, device: torch.device):
        # the model is on the CPU: leave the swapped blocks (and an unused reference stream) there
        saved = {}
        if self.blocks_to_swap:
            saved["single_blocks"] = self.single_blocks
            self.single_blocks = nn.ModuleList()
        if self.image_cond_on_cpu:
            saved["image_cond"] = self.content_mode_blocks["image_cond"]
            self.content_mode_blocks["image_cond"] = nn.ModuleList()
        self.to(device)
        if "single_blocks" in saved:
            self.single_blocks = saved["single_blocks"]
        if "image_cond" in saved:
            self.content_mode_blocks["image_cond"] = saved["image_cond"]

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader_single.prepare_block_devices_before_forward(self.single_blocks)

    # ------------------------------------------------------------------ forward
    def _time_vec(self, timesteps: Tensor) -> Tensor:
        emb = timestep_embedding(timesteps.float(), 256)
        return self.time_in(emb.to(self.time_in.in_layer.weight.dtype))

    def forward(
        self,
        ctx: Tensor,
        image: Tensor,
        image_ids: Tensor,
        timesteps: Tensor,
        image_cond: Optional[Tensor] = None,
        image_cond_ids: Optional[Tensor] = None,
        ctx_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Velocity of the image tokens.

        ctx: (B, L, 20480) text context; image: (B, S, 128) noisy latent tokens with ``image_ids`` (B, S, 4);
        timesteps: (B,) flow time in [0, 1] (1 = noise, the reference convention: target = noise - clean);
        image_cond / image_cond_ids: optional clean reference tokens (B, S_ref, 128) / (B, S_ref, 4).
        Returns (B, S, 128).
        """
        batch_size = image.shape[0]
        if ctx_ids is None:
            ctx_ids = text_ids(batch_size, ctx.shape[1], ctx.device)

        vec_image = self._time_vec(timesteps)
        vec_zero = self._time_vec(torch.zeros_like(timesteps))  # text and references run at t = 0
        vec = {"image": vec_image, "image_cond": vec_zero}

        streams = {"image": self.emb_in["image"](image)}
        ids = {"image": image_ids}
        if image_cond is not None:
            if self.image_cond_on_cpu:
                raise RuntimeError("FLUX.3: reference images were passed but the reference stream was left on the CPU")
            streams["image_cond"] = self.emb_in["image_cond"](image_cond)
            ids["image_cond"] = image_cond_ids
        active = [m for m in CONTENT_MODES if m in streams]
        txt = self.txt_in(ctx)
        del ctx

        early_mod = {m: self.early_stream_modulations[m](vec[m])[0] for m in active}
        early_mod_txt = self.early_stream_modulations["txt"](vec_zero)[0]
        single_mod = {m: self.single_stream_modulations[m](vec[m])[0] for m in active}
        single_mod_txt = self.single_stream_modulations["txt"](vec_zero)[0]

        pe = {m: self.pe_embedder(ids[m]) for m in active}
        pe_ctx = self.pe_embedder(ctx_ids)
        attn_params = AttentionParams.create_attention_params(self.attn_mode, self.split_attn)  # no attention mask

        for d in range(self.depth):
            for m in active:
                streams[m] = self.content_mode_blocks[m][d](streams[m], pe[m], early_mod[m], attn_params)
            txt = self.txt_mode_blocks[d](txt, pe_ctx, early_mod_txt, attn_params)
        del early_mod, early_mod_txt

        lengths = [streams[m].shape[1] for m in active]
        img = torch.cat([streams[m] for m in active], dim=1)
        del streams
        pe_joint = torch.cat((pe_ctx, torch.cat([pe[m] for m in active], dim=2)), dim=2)
        del pe, pe_ctx
        if len(active) == 1:
            mod_img = single_mod["image"]  # (B, 1, D) tables broadcast over the image tokens
        else:  # each mode's rows get its own table
            mod_img = tuple(
                torch.cat([single_mod[m][i].expand(-1, n, -1) for m, n in zip(active, lengths)], dim=1) for i in range(3)
            )

        for block_idx, block in enumerate(self.single_blocks):
            if self.blocks_to_swap:
                self.offloader_single.wait_for_block(block_idx)
            img, txt = block(img, txt, pe_joint, mod_img, single_mod_txt, attn_params)
            if self.blocks_to_swap:
                self.offloader_single.submit_move_blocks_forward(self.single_blocks, block_idx)
        del txt, pe_joint, mod_img, single_mod_txt

        img = img[:, : lengths[0]].to(vec_image.device)  # the image tokens (first); back on the GPU after activation offloading
        return self.final_layer["image"](img, vec_image)
