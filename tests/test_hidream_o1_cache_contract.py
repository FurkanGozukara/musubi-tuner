"""HiDream-O1 text-encoder cache contract (reference images for the i2i task)."""

import inspect

import torch
from safetensors.torch import load_file

from musubi_tuner.dataset import cache_io, image_video_dataset
from musubi_tuner.dataset.image_video_dataset import ItemInfo


def test_hidream_o1_text_cache_writers_accept_reference_pixels():
    # hidream_o1_cache_text_encoder_outputs.py passes pixel_values= for control/reference datasets
    for writer in (cache_io.save_text_encoder_output_cache_hidream_o1, image_video_dataset.save_text_encoder_output_cache_hidream_o1):
        parameters = inspect.signature(writer).parameters
        assert "pixel_values" in parameters and "image_grid_thw" in parameters
    assert image_video_dataset.save_text_encoder_output_cache_hidream_o1 is cache_io.save_text_encoder_output_cache_hidream_o1


def test_hidream_o1_text_cache_round_trip(tmp_path):
    item = ItemInfo("sample", "a photo of ohwx", (64, 64))
    item.text_encoder_output_cache_path = str(tmp_path / "sample_ho1_te.safetensors")
    input_ids = torch.arange(12, dtype=torch.int64)
    pixel_values = torch.zeros(4, 8, dtype=torch.bfloat16)
    image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int64)
    cache_io.save_text_encoder_output_cache_hidream_o1(item, input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)
    saved = load_file(item.text_encoder_output_cache_path)
    assert any(key.startswith("varlen_input_ids_") for key in saved)
    assert any(key.startswith("varlen_pixel_values_") for key in saved)
    assert any(key.startswith("varlen_image_grid_thw_") for key in saved)