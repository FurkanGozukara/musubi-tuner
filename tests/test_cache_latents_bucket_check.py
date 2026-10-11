import torch
from safetensors.torch import load_file, safe_open, save_file

from musubi_tuner.cache_latents import latent_cache_matches_bucket
from musubi_tuner.dataset.cache_io import save_latent_cache_common
from musubi_tuner.dataset.image_video_dataset import ItemInfo


def _item(path, bucket):
    item = ItemInfo("img.png", "caption", (1024, 1024), bucket)
    item.latent_cache_path = str(path)
    return item


def test_new_caches_record_their_bucket_and_compare_exactly(tmp_path):
    # image caches are named by the *source* size (img_1024x1024_ho1.safetensors) whatever the training bucket
    item = _item(tmp_path / "img_1024x1024_ho1.safetensors", (1024, 1024))
    save_latent_cache_common(item, {"latents_1x32x32_bfloat16": torch.zeros(32, 32, 8, dtype=torch.bfloat16)}, "hidream_o1")
    with safe_open(item.latent_cache_path, framework="pt") as handle:
        assert handle.metadata()["bucket_size"] == "1024x1024"
    assert latent_cache_matches_bucket(item)
    # 32x32 patches also fit 512 px at factor 16: only the recorded bucket can tell them apart
    assert not latent_cache_matches_bucket(_item(item.latent_cache_path, (512, 512)))
    assert load_file(item.latent_cache_path)["latents_1x32x32_bfloat16"].shape == (32, 32, 8)


def test_older_caches_fall_back_to_the_latent_grid(tmp_path):
    path = tmp_path / "img_1024x1024_qi21.safetensors"
    save_file({"latents_1x84x82_bfloat16": torch.zeros(64, 1, 84, 82, dtype=torch.bfloat16)}, str(path))
    assert latent_cache_matches_bucket(_item(path, (1312, 1344)))  # 82*16 x 84*16
    assert not latent_cache_matches_bucket(_item(path, (768, 768)))
    assert not latent_cache_matches_bucket(_item(tmp_path / "missing.safetensors", (768, 768)))
    ernie = tmp_path / "b_1024x1024_ei.safetensors"
    save_file({"latents_64x64_bfloat16": torch.zeros(128, 64, 64, dtype=torch.bfloat16)}, str(ernie))
    assert latent_cache_matches_bucket(_item(ernie, (1024, 1024)))
    assert not latent_cache_matches_bucket(_item(ernie, (768, 768)))


def test_unknown_layout_keeps_the_exists_test(tmp_path):
    path = tmp_path / "c_1024x1024_xx.safetensors"
    save_file({"something_else": torch.zeros(2)}, str(path))
    assert latent_cache_matches_bucket(_item(path, (512, 512)))
    garbage = tmp_path / "d_1024x1024_xx.safetensors"
    garbage.write_bytes(b"not a safetensors file")
    assert latent_cache_matches_bucket(_item(garbage, (512, 512)))
    assert latent_cache_matches_bucket(_item(path, None))
