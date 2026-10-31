"""--timestep_sampling truncated_logit_normal (kohya sd-scripts 188c03c, the MOSS-Image pre-training schedule) in both
training loops, on the shift-aware grid, CPU only."""

import argparse
import math

import pytest
import torch

from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from musubi_tuner.training.timesteps import get_sigmas, sample_truncated_logit_normal_indices


def _kohya_indices(batch_size, num_timesteps, logit_mean, logit_std, cutoff, uniform_ratio):
    """kohya sd-scripts cb70b96 library/flux_train_utils.sample_truncated_logit_normal_indices (no offset), copied."""
    mean = torch.full((batch_size,), float(logit_mean), dtype=torch.float64)
    std = float(logit_std)
    if cutoff < 1.0:
        z_lo = -math.log(cutoff / (1.0 - cutoff))
        p_lo = torch.special.ndtr((z_lo - mean) / std)
    else:
        p_lo = torch.zeros_like(mean)
    p = p_lo + torch.rand(batch_size, dtype=torch.float64) * (1.0 - p_lo)
    z = mean + std * torch.special.ndtri(p.clamp(max=1.0 - 1e-12))
    s = torch.sigmoid(-z) / cutoff
    indices = (num_timesteps - torch.ceil(s * num_timesteps).long()).clamp(0, num_timesteps - 1)
    if uniform_ratio > 0.0:
        use_uniform = torch.rand(batch_size) < uniform_ratio
        indices = torch.where(use_uniform, torch.randint(0, num_timesteps, (batch_size,)), indices)
    return indices


def test_the_sampler_is_kohyas():
    torch.manual_seed(7)
    ours = sample_truncated_logit_normal_indices(4096, 1000, -1.09, 0.95, 0.95, 0.01)
    torch.manual_seed(7)
    theirs = _kohya_indices(4096, 1000, -1.09, 0.95, 0.95, 0.01)
    assert torch.equal(ours, theirs) and ours.min() >= 0 and ours.max() <= 999


def test_the_truncated_distribution_follows_the_logit_normal_cdf():
    """Without the uniform share, P(sigma <= x) = P(S <= x * cutoff | S <= cutoff) for S = sigmoid(-N(mean, std))."""
    torch.manual_seed(0)
    mean, std, cutoff = -1.09, 0.95, 0.5
    indices = sample_truncated_logit_normal_indices(200_000, 1000, mean, std, cutoff, 0.0)
    sigmas = (1000 - indices).double() / 1000  # the unshifted grid: index i holds sigma (1000 - i) / 1000

    def cdf_s(value):  # P(sigmoid(-Z) <= value) = P(Z >= -logit(value))
        return 1.0 - 0.5 * (1.0 + math.erf((-math.log(value / (1.0 - value)) - mean) / (std * math.sqrt(2.0))))

    for x in (0.2, 0.5, 0.8):
        expected = cdf_s(x * cutoff) / cdf_s(cutoff)
        assert abs(float((sigmas <= x).double().mean()) - expected) < 0.01


@pytest.mark.parametrize("cutoff, ratio", [(0.0, 0.01), (1.5, 0.01), (0.95, -0.1), (0.95, 1.5)])
def test_out_of_range_parameters_are_refused(cutoff, ratio):
    with pytest.raises(ValueError):
        sample_truncated_logit_normal_indices(4, 1000, 0.0, 1.0, cutoff, ratio)


@pytest.mark.parametrize("trainer_kind", ["lora", "full"])
def test_both_loops_draw_grid_timesteps_the_moss_dit_sees_as_1000_sigma(trainer_kind):
    """The extended loop (LoRA) and trainer_base's loop (full fine-tuning) both sample on the shift-aware grid; MOSS's DiT
    gets t = 1000 * sigma of the noise it sees."""
    if trainer_kind == "lora":
        from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer as Trainer
    else:
        from musubi_tuner.moss_image_train import MossImageTrainer as Trainer
    scheduler = FlowMatchDiscreteScheduler(shift=3.0, reverse=True, solver="euler")
    args = argparse.Namespace(
        timestep_sampling="truncated_logit_normal", logit_mean=-1.09, logit_std=0.95, timestep_cutoff=0.95,
        timestep_uniform_ratio=0.01, weighting_scheme="none", mode_scale=1.29, min_timestep=None, max_timestep=None,
    )
    torch.manual_seed(3)
    latents = torch.randn(64, 8, 2, 2)
    noise = torch.randn_like(latents)
    noisy, timesteps = Trainer().get_noisy_model_input_and_timesteps(args, noise, latents, None, scheduler, torch.device("cpu"), torch.float32)
    sigmas = get_sigmas(scheduler, timesteps, "cpu", n_dim=4)
    assert torch.allclose(noisy, sigmas * noise + (1 - sigmas) * latents)
    assert torch.allclose(timesteps, sigmas.flatten() * 1000.0, atol=1e-3)
    assert bool(torch.isin(timesteps, scheduler.timesteps).all())


def test_the_common_parser_takes_the_schedule():
    from musubi_tuner.training.parser_common import setup_parser_common

    args = setup_parser_common().parse_args(
        ["--timestep_sampling", "truncated_logit_normal", "--timestep_cutoff", "0.9", "--timestep_uniform_ratio", "0.05"]
    )
    assert (args.timestep_sampling, args.timestep_cutoff, args.timestep_uniform_ratio) == ("truncated_logit_normal", 0.9, 0.05)
    defaults = setup_parser_common().parse_args([])
    assert (defaults.timestep_cutoff, defaults.timestep_uniform_ratio) == (0.95, 0.01)
