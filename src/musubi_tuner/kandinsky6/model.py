from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from .configs import MODEL_CONFIGS
from .dit import DiffusionTransformer3D
from .piflow_dit import PiFlowDiffusionTransformer3D


TOKEN_TYPE_KEY = "visual_token_type_embeddings.weight"
REF_SLOT_KEY = "visual_ref_slot_embeddings.weight"
# Token-type row used for multi-reference (R2AV) frames; rows 0/1 are the released
# generated / first-frame types.
REFERENCE_TOKEN_TYPE = 2


def _stream_assign_safetensors(
    model: torch.nn.Module,
    path: str | Path,
    *,
    dtype: torch.dtype | None,
    device: str | torch.device,
    strict: bool,
    extend_reference: bool = False,
) -> tuple[list[str], list[str]]:
    """Assign one safetensors tensor at a time into a meta-instantiated module.

    With ``extend_reference`` a released checkpoint is loaded into a model built with
    reference slots: the token-type table gains rows (row ``REFERENCE_TOKEN_TYPE`` starts
    as a copy of the trained first-frame row, so references begin as "clean conditioning
    frames" the model already understands) and the absent slot table starts at zero.
    """
    device = torch.device(device)
    expected = model.state_dict()
    expected_keys = set(expected)
    with safe_open(path, framework="pt", device="cpu") as handle:
        file_keys = set(handle.keys())
        missing = sorted(expected_keys - file_keys)
        unexpected = sorted(file_keys - expected_keys)
        allowed_missing = {REF_SLOT_KEY} if extend_reference else set()
        if strict and (set(missing) - allowed_missing or unexpected):
            raise RuntimeError(f"State dict mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
        for key in sorted(expected_keys & file_keys):
            tensor = handle.get_tensor(key)
            expected_tensor = expected[key]
            if extend_reference and key == TOKEN_TYPE_KEY and tensor.shape != expected_tensor.shape:
                rows, width = tensor.shape
                if width != expected_tensor.shape[1] or rows < 2 or rows > expected_tensor.shape[0]:
                    raise RuntimeError(f"Cannot extend {key} from {tuple(tensor.shape)} to {tuple(expected_tensor.shape)}")
                extended = torch.empty(tuple(expected_tensor.shape), dtype=tensor.dtype)
                extended[:rows] = tensor
                extended[rows:] = tensor[1]  # new reference rows start as the first-frame condition row
                tensor = extended
            if tensor.shape != expected_tensor.shape:
                raise RuntimeError(
                    f"State dict shape mismatch for {key}: checkpoint={tuple(tensor.shape)}, model={tuple(expected_tensor.shape)}"
                )
            target_dtype = dtype if dtype is not None and tensor.is_floating_point() else tensor.dtype
            tensor = tensor.to(device=device, dtype=target_dtype)
            _assign_tensor(model, key, tensor)
        for key in sorted(set(missing) & allowed_missing):
            tensor = torch.zeros(tuple(expected[key].shape), dtype=dtype or torch.float32, device=device)
            _assign_tensor(model, key, tensor)
    return missing, unexpected


def _assign_tensor(model: torch.nn.Module, key: str, tensor: torch.Tensor) -> None:
    parent_name, _, leaf_name = key.rpartition(".")
    module = model.get_submodule(parent_name) if parent_name else model
    if leaf_name in module._parameters:
        old_parameter = module._parameters[leaf_name]
        requires_grad = old_parameter.requires_grad if old_parameter is not None else True
        module._parameters[leaf_name] = torch.nn.Parameter(tensor, requires_grad=requires_grad)
    elif leaf_name in module._buffers:
        module._buffers[leaf_name] = tensor
    else:
        raise RuntimeError(f"State dict key {key} does not resolve to a parameter or buffer")


def _resolve_weight_file(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        candidates = (path / "transformer" / "diffusion_pytorch_model.safetensors", path / "diffusion_pytorch_model.safetensors")
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"No Kandinsky 6 transformer safetensors found under {path}")
    return path


def inspect_reference_layout(path: str | Path) -> tuple[int, int]:
    """Return (token-type rows, reference slots) stored in a checkpoint; slots are 0 for released files."""
    path = _resolve_weight_file(path)
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        type_rows = f.get_slice(TOKEN_TYPE_KEY).get_shape()[0] if TOKEN_TYPE_KEY in keys else 0
        slots = f.get_slice(REF_SLOT_KEY).get_shape()[0] if REF_SLOT_KEY in keys else 0
    return int(type_rows), int(slots)


def _reference_config(path: Path, reference_slots: int | None) -> tuple[dict, bool]:
    """Config overrides for the reference layout and whether a released file must be extended."""
    type_rows, file_slots = inspect_reference_layout(path)
    if file_slots > 0:
        if reference_slots not in (None, 0, file_slots):
            raise ValueError(f"Checkpoint has {file_slots} reference slots, but {reference_slots} were requested")
        return {"visual_token_type_num_embeddings": type_rows, "reference_slots": file_slots}, False
    if reference_slots:
        return {
            "visual_token_type_num_embeddings": max(type_rows, REFERENCE_TOKEN_TYPE + 1),
            "reference_slots": reference_slots,
        }, True
    return {}, False


def inspect_checkpoint(path: str | Path) -> tuple[str, int | None]:
    """Return (lite/pro, PiFlow grid count or None) without loading weights."""
    path = _resolve_weight_file(path)
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        sentinel = "visual_embeddings.in_layer.weight"
        if sentinel not in keys:
            raise ValueError(f"Not a Kandinsky 6 checkpoint: missing {sentinel}")
        hidden_size = f.get_slice(sentinel).get_shape()[0]
        if hidden_size == 1792:
            variant = "lite"
        elif hidden_size == 4096:
            variant = "pro"
        else:
            raise ValueError(f"Unknown Kandinsky 6 hidden size {hidden_size}; expected Lite 1792 or Pro 4096")
        out_key = "out_layer.out_layer.weight"
        output_width = f.get_slice(out_key).get_shape()[0]
    base_width = 16 * 4
    if output_width % base_width:
        raise ValueError(f"Unexpected Kandinsky 6 output head width {output_width}")
    n_grid = output_width // base_width
    return variant, (n_grid if n_grid > 1 else None)


def load_dit(
    checkpoint_path: str | Path,
    model_variant: str = "auto",
    dtype: torch.dtype | None = torch.bfloat16,
    device: str | torch.device = "cpu",
    strict: bool = True,
    reference_slots: int | None = None,
    attention_engine: str | None = None,
) -> DiffusionTransformer3D:
    """Load a published Lite/Pro or distilled PiFlow Kandinsky 6 transformer.

    Checkpoints trained with reference slots are detected and loaded as such. Passing
    ``reference_slots`` for a released checkpoint adds the (untrained) reference extension.
    """
    path = _resolve_weight_file(checkpoint_path)
    detected_variant, n_grid = inspect_checkpoint(path)
    if model_variant == "auto":
        model_variant = detected_variant
    if model_variant not in MODEL_CONFIGS:
        raise ValueError(f"model_variant must be 'auto', 'lite', or 'pro', got {model_variant!r}")
    if model_variant != detected_variant:
        raise ValueError(f"Checkpoint is {detected_variant}, requested {model_variant}")
    config = dict(MODEL_CONFIGS[model_variant])
    if attention_engine is not None:
        config["attention_engine"] = attention_engine
    reference_overrides, extend_reference = _reference_config(path, reference_slots)
    if reference_overrides and n_grid is not None:
        raise ValueError("PiFlow distilled Kandinsky 6 checkpoints do not support reference slots")
    config.update(reference_overrides)
    with torch.device("meta"):
        if n_grid is None:
            model = DiffusionTransformer3D(**config)
        else:
            model = PiFlowDiffusionTransformer3D(
                n_grid=n_grid, out_visual_dim=config.pop("out_visual_dim"), out_audio_dim=config.pop("out_audio_dim"), **config
            )
    _stream_assign_safetensors(model, path, dtype=dtype, device=device, strict=strict, extend_reference=extend_reference)
    for module in model.modules():
        touched = False
        for name, buffer in list(module._buffers.items()):
            if buffer is not None and buffer.device.type == "meta":
                module._buffers[name] = torch.empty(buffer.shape, dtype=buffer.dtype, device=device)
                touched = True
        if touched:
            module.reset_parameters()
    model.eval()
    model.model_variant = model_variant
    model.piflow_n_grid = n_grid
    model.reference_extension_is_fresh = extend_reference
    return model


def load_dit_convrot_int8(
    checkpoint_path: str | Path,
    model_variant: str = "auto",
    device: str | torch.device = "cpu",
    quant_device: str | torch.device = "cuda",
    bwd_mode: str = "bf16",
    disable_numpy_memmap: bool = False,
    attention_engine: str | None = None,
) -> DiffusionTransformer3D:
    """Stream and quantize the frozen DiT base for memory-efficient LoRA training."""
    from musubi_tuner.modules.convrot_int8_utils import ConvRotInt8Quantizer, apply_convrot_int8_monkey_patch
    from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8

    path = _resolve_weight_file(checkpoint_path)
    detected_variant, n_grid = inspect_checkpoint(path)
    if model_variant == "auto":
        model_variant = detected_variant
    if model_variant != detected_variant or model_variant not in MODEL_CONFIGS:
        raise ValueError(f"Checkpoint is {detected_variant}, requested {model_variant}")
    config = dict(MODEL_CONFIGS[model_variant])
    if attention_engine is not None:
        config["attention_engine"] = attention_engine
    config.update(_reference_config(path, None)[0])
    with torch.device("meta"):
        if n_grid is None:
            model = DiffusionTransformer3D(**config)
        else:
            model = PiFlowDiffusionTransformer3D(
                n_grid=n_grid, out_visual_dim=config.pop("out_visual_dim"), out_audio_dim=config.pop("out_audio_dim"), **config
            )
    quantizer = ConvRotInt8Quantizer(
        target_layer_keys=[
            "visual_transformer_blocks.",
            "video_text_transformer_blocks.",
            "audio_text_transformer_blocks.",
        ],
        exclude_layer_keys=["modulation", "norm"],
        allowed_groupsizes=(256, 64),
        allow_unrotated=True,
    )
    state = load_safetensors_with_lora_and_fp8(
        model_files=[str(path)],
        lora_weights_list=None,
        lora_multipliers=None,
        fp8_optimization=False,
        calc_device=torch.device(quant_device),
        move_to_device=torch.device(device) == torch.device(quant_device),
        dit_weight_dtype=None,
        disable_numpy_memmap=disable_numpy_memmap,
        quantizer=quantizer,
    )
    apply_convrot_int8_monkey_patch(model, state, bwd_mode=bwd_mode, groupsize_map=quantizer.module_groupsizes)
    model.requires_grad_(False)
    for key, value in state.items():
        if value.dtype in (torch.float16, torch.float32):
            state[key] = value.to(torch.bfloat16) if not key.endswith(".scale_weight") else value.float()
        if torch.device(device).type != "cpu":
            state[key] = state[key].to(device)
    model.load_state_dict(state, strict=True, assign=True)
    for module in model.modules():
        touched = False
        for name, buffer in list(module._buffers.items()):
            if buffer is not None and buffer.device.type == "meta":
                module._buffers[name] = torch.empty(buffer.shape, dtype=buffer.dtype, device=device)
                touched = True
        if touched:
            module.reset_parameters()
    model.eval()
    model.model_variant = model_variant
    model.piflow_n_grid = n_grid
    return model


def load_dit_fp8(
    checkpoint_path: str | Path,
    model_variant: str = "auto",
    device: str | torch.device = "cpu",
    quant_device: str | torch.device = "cuda",
    disable_numpy_memmap: bool = False,
    attention_engine: str | None = None,
) -> DiffusionTransformer3D:
    """Stream and scale-quantize transformer-block Linears to FP8."""
    from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
    from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8

    path = _resolve_weight_file(checkpoint_path)
    detected_variant, n_grid = inspect_checkpoint(path)
    if model_variant == "auto":
        model_variant = detected_variant
    if model_variant != detected_variant or model_variant not in MODEL_CONFIGS:
        raise ValueError(f"Checkpoint is {detected_variant}, requested {model_variant}")
    config = dict(MODEL_CONFIGS[model_variant])
    if attention_engine is not None:
        config["attention_engine"] = attention_engine
    config.update(_reference_config(path, None)[0])
    with torch.device("meta"):
        if n_grid is None:
            model = DiffusionTransformer3D(**config)
        else:
            model = PiFlowDiffusionTransformer3D(
                n_grid=n_grid, out_visual_dim=config.pop("out_visual_dim"), out_audio_dim=config.pop("out_audio_dim"), **config
            )
    state = load_safetensors_with_lora_and_fp8(
        model_files=[str(path)],
        lora_weights_list=None,
        lora_multipliers=None,
        fp8_optimization=True,
        calc_device=torch.device(quant_device),
        move_to_device=torch.device(device) == torch.device(quant_device),
        dit_weight_dtype=None,
        target_keys=["visual_transformer_blocks.", "video_text_transformer_blocks.", "audio_text_transformer_blocks."],
        exclude_keys=["modulation", "norm"],
        disable_numpy_memmap=disable_numpy_memmap,
    )
    apply_fp8_monkey_patch(model, state, use_scaled_mm=False)
    model.requires_grad_(False)
    if torch.device(device).type != "cpu":
        state = {key: value.to(device) for key, value in state.items()}
    model.load_state_dict(state, strict=True, assign=True)
    for module in model.modules():
        touched = False
        for name, buffer in list(module._buffers.items()):
            if buffer is not None and buffer.device.type == "meta":
                module._buffers[name] = torch.empty(buffer.shape, dtype=buffer.dtype, device=device)
                touched = True
        if touched:
            module.reset_parameters()
    model.eval()
    model.model_variant = model_variant
    model.piflow_n_grid = n_grid
    return model
