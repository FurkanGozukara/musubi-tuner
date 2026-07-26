"""Online, differentiable-reward training for LTX-2.

This backend performs one differentiable denoising step from a re-noised, detached rollout latent,
then maximizes a ``kind == "differentiable"`` reward. ``--refl_renoise_samples`` repeats that
single-step estimate with fresh noise and averages the losses. Rewards may operate directly on the
latent or on media decoded by frozen video/audio decoders.

The optional reference term is mean-squared error between the policy and LoRA-disabled denoised
latents. Its coefficient reuses the historically named ``--nft_kl_beta`` flag; it is not a measured
KL divergence. Non-differentiable rewards remain available only to the policy-gradient update rules.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def compute_refl_loss(
    reward: torch.Tensor,
    fwd_x0: torch.Tensor,
    ref_x0: Optional[torch.Tensor],
    *,
    reward_weight: float,
    anchor_beta: float,
    additional_anchors: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Reward-maximization loss with an optional reference-x0 MSE term.

    Args:
        reward:   ``[K]`` grad-carrying per-sample reward (higher-is-better) from a differentiable reward.
        fwd_x0:   ``[K, ...]`` grad-carrying denoised latent from the policy (LoRA-on) forward.
        ref_x0:   ``[K, ...]`` DETACHED denoised latent from the frozen base (LoRA-off) forward, or None.
        reward_weight: scale on the (maximized) reward term.
        anchor_beta: scale on ``E||fwd_x0 - ref_x0||^2``.

    Returns ``(loss, info)``. The minimized value is the negative mean reward plus the weighted MSE.
    """
    reward_term = -float(reward_weight) * reward.float().mean()
    anchor_pairs = []
    if ref_x0 is not None:
        anchor_pairs.append((fwd_x0, ref_x0))
    anchor_pairs.extend(additional_anchors or [])
    if anchor_pairs and float(anchor_beta) != 0.0:
        modality_mses = []
        for policy, reference in anchor_pairs:
            reduce = tuple(range(1, policy.dim()))
            modality_mses.append(((policy.float() - reference.float()) ** 2).mean(dim=reduce).mean())
        anchor_mse = torch.stack(modality_mses).mean()
    else:
        anchor_mse = torch.zeros((), device=fwd_x0.device, dtype=torch.float32)
    loss = reward_term + float(anchor_beta) * anchor_mse
    info = {
        "policy": reward_term.detach(),
        "anchor_mse": anchor_mse.detach(),
        "reward": reward.detach().float().mean(),
    }
    return loss, info


def _final_step_sigmas(sigmas: torch.Tensor, grad_steps: int) -> torch.Tensor:
    """Return the requested smallest positive sigmas, clamped away from zero."""
    s = sigmas.reshape(-1).clamp_min(1e-4)
    k = max(1, min(int(grad_steps), s.numel()))
    return torch.topk(s, k, largest=False).values  # k smallest


def decode_latent_for_reward(net_trainer, vae, latent: torch.Tensor):
    """Decode model-space video latents to differentiable frames in ``[0, 1]``."""
    if vae is None:
        raise ValueError("Pixel-space ReFL rewards require --vae")
    if latent.dim() != 5:
        raise ValueError(f"Expected ReFL video latent [B,C,F,H,W], got {tuple(latent.shape)}")

    decoded = vae.decode([sample for sample in latent])
    if not isinstance(decoded, (list, tuple)) or len(decoded) != latent.shape[0]:
        raise RuntimeError("LTX-2 VAE decode returned an unexpected batch")
    frames = torch.stack(list(decoded), dim=0)
    return (frames / 2 + 0.5).clamp(0, 1)


def decode_audio_latent_for_reward(audio_decoder, vocoder, latent: torch.Tensor) -> torch.Tensor:
    """Decode audio latents to a grad-carrying waveform batch."""
    if audio_decoder is None or vocoder is None:
        raise ValueError("Waveform-space AV ReFL rewards require the LTX-2 audio decoder and vocoder")
    if latent.dim() != 4:
        raise ValueError(f"Expected ReFL audio latent [B,C,T,F], got {tuple(latent.shape)}")
    first_param = next(audio_decoder.parameters(), None)
    if first_param is not None:
        latent = latent.to(device=first_param.device, dtype=first_param.dtype)
    waveform = vocoder(audio_decoder(latent)).float()
    if waveform.dim() < 2 or waveform.shape[0] != latent.shape[0]:
        raise RuntimeError("LTX-2 audio decode returned an unexpected batch")
    return waveform


def run_refl(
    net_trainer,
    args,
    accelerator,
    transformer,
    network,
    optimizer,
    lr_scheduler,
    device,
    dit_dtype: torch.dtype,
    *,
    is_av: bool = False,
) -> None:
    """Online differentiable-reward training loop. Called from ``ltx2_train_rl`` after the shared
    model/LoRA/optimizer setup when ``--rl_loss refl``; the policy-gradient loop never runs then."""
    from musubi_tuner.ltx2_rewards import RewardStack, load_reward_plugins, parse_reward_spec
    from musubi_tuner.ltx2_rl_generate import build_generate_fn, make_sigma_schedule, prepare_sampling_args
    from musubi_tuner.ltx_2.utils import to_denoised
    from tqdm import tqdm

    refl_av = bool(getattr(args, "refl_av", False))
    if is_av and not refl_av:
        raise NotImplementedError("AV ReFL is opt-in: add --refl_av, or use --rl_loss nft/ppo for AV.")
    if refl_av and not is_av:
        raise ValueError("--refl_av requires --ltx2_mode av")

    net = accelerator.unwrap_model(network)
    unwrapped = accelerator.unwrap_model(transformer)
    blocks_to_swap = int(getattr(args, "blocks_to_swap", 0) or 0)

    grad_steps = int(getattr(args, "refl_grad_steps", 1) or 1)
    renoise_samples = int(getattr(args, "refl_renoise_samples", 1) or 1)
    reward_weight = float(getattr(args, "refl_reward_weight", 1.0))
    anchor_beta = float(getattr(args, "nft_kl_beta", 1e-4))
    if grad_steps != 1:
        raise ValueError("--refl_grad_steps currently supports only 1; multi-step differentiable denoising is not implemented")

    frame_rate = float(getattr(args, "frame_rate", 24.0))
    num_steps = int(getattr(args, "sample_steps", 20) or 20)

    # --- reward stack (must be fully differentiable) ---
    reward_kwargs: Dict[str, str] = {}
    for raw in getattr(args, "reward_args", None) or []:
        if "=" not in raw:
            raise ValueError(f"--reward_args entry '{raw}' must be key=value")
        key, val = raw.split("=", 1)
        reward_kwargs[key.strip()] = val
    if getattr(args, "reward_plugins", None):
        load_reward_plugins(args.reward_plugins)
    per_reward_args = {name: dict(reward_kwargs) for name in parse_reward_spec(args.reward_fn)}
    reward_stack = RewardStack.from_spec(args.reward_fn, device=device, reward_args=per_reward_args)
    reward_stack.assert_differentiable()  # fail loudly if a blackbox reward is in a refl spec
    needs_pixels = any("video" in getattr(r, "needs", frozenset()) for r in reward_stack._rewards.values())
    needs_audio_waveform = any("audio_waveform" in getattr(r, "needs", frozenset()) for r in reward_stack._rewards.values())
    if needs_audio_waveform and not is_av:
        raise ValueError("Waveform-space ReFL rewards require --ltx2_mode av --refl_av")

    # --- generation (no_grad) supplies the clean latent x0 to re-noise; no decode needed for x0 ---
    vae = None
    if needs_pixels:
        if not getattr(args, "vae", None):
            raise ValueError("Pixel-space ReFL rewards require --vae")
        from musubi_tuner.utils import model_utils

        vae_dtype = model_utils.str_to_dtype(args.vae_dtype)
        vae = net_trainer.load_vae(args, vae_dtype=vae_dtype, vae_path=args.vae)
        vae.to_device(device)
        vae.to_dtype(vae_dtype)
        vae.eval()
        vae.requires_grad_(False)
    audio_decoder = None
    vocoder = None
    if needs_audio_waveform:
        from musubi_tuner.utils import model_utils

        audio_dtype = model_utils.str_to_dtype(args.vae_dtype)
        audio_decoder, vocoder = net_trainer._load_audio_components(
            args,
            audio_dtype,
            args.ltx2_checkpoint,
            device=device,
        )
        audio_decoder.requires_grad_(False)
        vocoder.requires_grad_(False)
    prepare_sampling_args(args)
    sigma_schedule = make_sigma_schedule(num_steps)
    te_dtype = net_trainer._build_text_encoder(args, accelerator)
    gen_fn = build_generate_fn(
        net_trainer,
        args,
        accelerator,
        transformer,
        vae,
        dit_dtype,
        device,
        num_steps=num_steps,
        needs_media=False,
        sigma_schedule=sigma_schedule,
        te_dtype=te_dtype,
        media_needs=frozenset(),
    )
    with open(args.rl_prompts, encoding="utf-8") as f:
        prompts = [s for ln in f if (s := ln.strip()) and not s.startswith("#")]
    if not prompts:
        raise ValueError(f"--rl_prompts {args.rl_prompts!r} contains no prompts")

    group_size = int(getattr(args, "rl_group_size", 8) or 8)
    max_steps = int(getattr(args, "rl_max_steps", 0)) or (len(prompts) * renoise_samples)
    seed_base = int(getattr(args, "seed", 0) or 0)

    tb_writer = None
    if accelerator.is_main_process and getattr(args, "logging_dir", None):
        from torch.utils.tensorboard import SummaryWriter

        tb_writer = SummaryWriter(os.path.join(args.logging_dir, args.output_name or "ltx2_refl"))

    logger.info(
        "ReFL: grad_steps=%d renoise=%d reward_weight=%.4g anchor_beta=%.4g reward=%s video_pixels=%s audio_waveform=%s",
        grad_steps,
        renoise_samples,
        reward_weight,
        anchor_beta,
        args.reward_fn,
        needs_pixels,
        needs_audio_waveform,
    )

    global_step = 0
    progress = tqdm(total=max_steps, desc="RL (ReFL)")
    prompt_i = 0
    while global_step < max_steps:
        prompt = prompts[prompt_i % len(prompts)]
        seeds = [seed_base + prompt_i * group_size + j for j in range(group_size)]
        prompt_i += 1

        # 1) generate clean rollout latents under no_grad (the differentiable-reward starting point)
        samples = gen_fn(prompt, seeds)
        x0 = torch.stack([s["video_x0"] for s in samples], dim=0).to(device=device, dtype=dit_dtype)  # [K,C,F,H,W]
        audio_x0 = None
        if is_av:
            audio_x0 = torch.stack([s["audio_x0"] for s in samples], dim=0).to(device=device, dtype=dit_dtype)
        v_ctx = samples[0]["v_ctx"].to(device=device, dtype=dit_dtype)
        if v_ctx.dim() == 2:
            v_ctx = v_ctx.unsqueeze(0)
        v_ctx = v_ctx.expand(x0.shape[0], *v_ctx.shape[1:]) if v_ctx.shape[0] == 1 else v_ctx
        v_mask = samples[0].get("v_mask")
        if v_mask is not None:
            v_mask = v_mask.to(device=device)
            if v_mask.dim() == 1:
                v_mask = v_mask.unsqueeze(0)
            if v_mask.shape[0] == 1:
                v_mask = v_mask.expand(x0.shape[0], *v_mask.shape[1:])
        final_sigmas = _final_step_sigmas(samples[0]["sigmas"].to(device), grad_steps).to(torch.float32)
        k = x0.shape[0]

        def _forward(xt: torch.Tensor, model_ts: torch.Tensor, xt_audio: Optional[torch.Tensor] = None):
            transformer_options = {}
            if getattr(args, "ltx2_causal_temporal_attention", False):
                transformer_options["causal_temporal_attention"] = True
            if is_av and getattr(args, "ltx2_soft_av_alignment", False):
                transformer_options["soft_av_alignment_sigma"] = float(getattr(args, "ltx2_soft_av_alignment_sigma", 1.0))
            model_input = [xt.to(dit_dtype), xt_audio.to(dit_dtype)] if is_av else xt.to(dit_dtype)
            fa, fk = net_trainer.prepare_forward_inputs(
                transformer,
                args,
                model_input=model_input,
                model_timesteps=model_ts,
                text_embeds=v_ctx,
                text_mask=v_mask,
                frame_rate=frame_rate,
                audio_timestep=model_ts if is_av else None,
                transformer_options=transformer_options,
            )
            with accelerator.autocast():
                out = transformer(*fa, **fk)
            if is_av:
                if not isinstance(out, (list, tuple)) or len(out) != 2:
                    raise ValueError("AV ReFL forward must return video and audio predictions")
                return out[0], out[1]
            return out[0] if isinstance(out, (list, tuple)) else out

        # 2) re-noise to a final-step sigma, one grad forward, decode/score the reward, KL, backward.
        acc_loss = None
        acc_info = {"policy": 0.0, "anchor_mse": 0.0, "reward": 0.0}
        for it in range(renoise_samples):
            sigma = final_sigmas[it % final_sigmas.numel()].expand(k)  # [K]
            sigma_b = sigma.view(k, *([1] * (x0.dim() - 1)))
            model_ts = sigma.view(k, 1).to(dtype=dit_dtype)
            noise = torch.randn_like(x0)
            xt = (1.0 - sigma_b) * x0 + sigma_b * noise  # rectified-flow noising to the final step
            xt_audio = None
            sigma_audio_b = None
            if is_av:
                sigma_audio_b = sigma.view(k, *([1] * (audio_x0.dim() - 1)))
                audio_noise = torch.randn_like(audio_x0)
                xt_audio = (1.0 - sigma_audio_b) * audio_x0 + sigma_audio_b * audio_noise

            # Frozen-base (LoRA-off) reference for the reference-x0 MSE term.
            if blocks_to_swap > 0:
                unwrapped.switch_block_swap_for_inference()
            with torch.no_grad():
                if blocks_to_swap > 0:
                    unwrapped.prepare_block_swap_before_forward()
                net.set_enabled(False)
                ref_out = _forward(xt, model_ts, xt_audio)
                if is_av:
                    ref_x0 = to_denoised(xt, ref_out[0], sigma_b).detach()
                    ref_audio_x0 = to_denoised(xt_audio, ref_out[1], sigma_audio_b).detach()
                else:
                    ref_x0 = to_denoised(xt, ref_out, sigma_b).detach()
                    ref_audio_x0 = None
                net.set_enabled(True)
            if blocks_to_swap > 0:
                unwrapped.switch_block_swap_for_training()
                unwrapped.prepare_block_swap_before_forward()

            # policy (LoRA-on) forward — grad
            fwd_out = _forward(xt, model_ts, xt_audio)
            if is_av:
                fwd_x0 = to_denoised(xt, fwd_out[0], sigma_b)
                fwd_audio_x0 = to_denoised(xt_audio, fwd_out[1], sigma_audio_b)
            else:
                fwd_x0 = to_denoised(xt, fwd_out, sigma_b)
                fwd_audio_x0 = None

            # build per-sample media dicts for the reward: latent always; pixels if any reward needs them
            media: List[Dict[str, Any]] = [{"video_x0": fwd_x0[j], "prompt": prompt} for j in range(k)]
            if is_av:
                for j in range(k):
                    media[j]["audio_x0"] = fwd_audio_x0[j]
            if needs_pixels:
                pixels = decode_latent_for_reward(net_trainer, vae, fwd_x0)  # [K,C,T,H,W] grad (GPU-verify)
                for j in range(k):
                    media[j]["video"] = pixels[j]
            if needs_audio_waveform:
                waveforms = decode_audio_latent_for_reward(audio_decoder, vocoder, fwd_audio_x0)
                for j in range(k):
                    media[j]["audio_waveform"] = waveforms[j]

            reward, r_info = reward_stack.score_grad(media)  # [K] grad
            additional_anchors = [(fwd_audio_x0, ref_audio_x0)] if is_av else None
            loss, info = compute_refl_loss(
                reward,
                fwd_x0,
                ref_x0,
                reward_weight=reward_weight,
                anchor_beta=anchor_beta,
                additional_anchors=additional_anchors,
            )
            acc_loss = loss if acc_loss is None else acc_loss + loss
            for key in acc_info:
                acc_info[key] += float(info[key])

        acc_loss = acc_loss / max(1, renoise_samples)
        accelerator.backward(acc_loss)
        if args.max_grad_norm:
            accelerator.clip_grad_norm_(net.trainable_lora_params(), args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1
        progress.update(1)
        post = {kk: vv / max(1, renoise_samples) for kk, vv in acc_info.items()}
        post["loss"] = float(acc_loss.detach())
        progress.set_postfix(**post)
        if tb_writer is not None:
            for kk, vv in post.items():
                tb_writer.add_scalar(f"refl/{kk}", vv, global_step)
            tb_writer.add_scalar("refl/lr", lr_scheduler.get_last_lr()[0], global_step)

    progress.close()
    if tb_writer is not None:
        tb_writer.close()
    net_trainer._cleanup_text_encoder(accelerator)
    if accelerator.is_main_process:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"{args.output_name or 'ltx2_refl_lora'}.safetensors")
        net.save_weights(save_path, torch.float16, None)
        logger.info("Saved ReFL LoRA to %s (global_step=%d)", save_path, global_step)
