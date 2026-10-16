"""A training sample's VAE decode falls back to the CPU in float32 when the GPU runs out of memory."""

import pytest
import torch

from musubi_tuner.training.sample_decode import decode_sample


class _Vae(torch.nn.Module):
    def __init__(self, failures: int = 0, error=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
        self.failures, self.error, self.calls = failures, error or torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 9 GiB"), []

    def decode(self, latents):
        self.calls.append((latents.device.type, latents.dtype, self.weight.dtype))
        if self.failures:
            self.failures -= 1
            raise self.error
        return latents * 2


def test_a_decode_that_fits_runs_as_called():
    vae = _Vae()
    latents = torch.ones(1, 4, dtype=torch.bfloat16)
    assert torch.equal(decode_sample(vae, vae.decode, latents), latents * 2)
    assert vae.calls == [("cpu", torch.bfloat16, torch.bfloat16)]


def test_out_of_memory_decodes_on_the_cpu_in_float32_from_then_on():
    vae = _Vae(failures=1)
    latents = torch.ones(1, 4, dtype=torch.bfloat16)
    pixels = decode_sample(vae, vae.decode, latents)
    assert pixels.dtype == torch.float32 and torch.equal(pixels, torch.full((1, 4), 2.0))
    assert vae.calls[1] == ("cpu", torch.float32, torch.float32)  # the retry: VAE and latents in float32 on the CPU
    assert vae.weight.dtype == torch.bfloat16 and vae.weight.device.type == "cpu"  # its own precision again, on the CPU
    decode_sample(vae, vae.decode, latents)
    assert len(vae.calls) == 3 and vae.calls[2] == ("cpu", torch.float32, torch.float32)  # no second GPU attempt


@pytest.mark.parametrize("message", ["CUDA out of memory. Tried to allocate", "cuDNN error: CUDNN_STATUS_ALLOC_FAILED"])
def test_out_of_memory_reported_as_runtime_error_is_retried(message):
    vae = _Vae(failures=1, error=RuntimeError(message))
    assert decode_sample(vae, vae.decode, torch.ones(1, 4)).dtype == torch.float32


def test_other_errors_are_not_hidden():
    vae = _Vae(failures=1, error=ValueError("bad latent shape"))
    with pytest.raises(ValueError, match="bad latent shape"):
        decode_sample(vae, vae.decode, torch.ones(1, 4))


def test_a_vae_wrapper_with_its_own_to_works_like_a_module():
    class Wrapper:  # Wan's WanVAE: not a Module, to(device, dtype) and a dtype attribute
        def __init__(self):
            self.model, self.dtype, self.moves = _Vae(failures=1), torch.bfloat16, []

        def to(self, device_or_dtype, dtype=None):
            self.moves.append((device_or_dtype, dtype))
            if isinstance(device_or_dtype, torch.dtype):
                self.dtype = device_or_dtype
            if dtype is not None:
                self.dtype = dtype
            self.model.to(device_or_dtype, dtype) if dtype is not None else self.model.to(device_or_dtype)

        def decode(self, latents):
            return [self.model.decode(latents)]

    vae = Wrapper()
    assert decode_sample(vae, vae.decode, torch.ones(1, 4, dtype=torch.bfloat16))[0].dtype == torch.float32
    assert vae.moves == [("cpu", torch.float32), (torch.bfloat16, None)] and vae.dtype == torch.bfloat16
